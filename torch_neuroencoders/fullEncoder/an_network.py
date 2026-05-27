"""
Neuroencoders: LSTMandSpikeNetwork
an_network module for training and managing LSTM and spiking neural networks in pure PyTorch.
"""

import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

# Get utility functions from native Torch modules
from torch_neuroencoders import nnUtils
from torch_neuroencoders.nnUtils_torch import (
    TorchBatchDataset,
    batch_examples,
    build_tfrecord_description,
    create_indices_torch,
    import_true_pos_torch,
    maybe_shuffle_examples,
    load_tfrecord_examples,
    write_tfrecord_examples,
    parse_serialized_sequence_torch,
)
from torch_neuroencoders.importData.epochs_management import (
    get_epochs_mask,
    inEpochsMask,
)
from torch_neuroencoders.utils.global_classes import (
    DEFAULT_GRIDSIZE,
    Params,
    Project,
    SpatialConstraintsMixin,
)


class LSTMandSpikeNetwork(nn.Module, SpatialConstraintsMixin):
    """
    LSTMandSpikeNetwork class, the main ANN Class implemented in native PyTorch.
    """

    def __init__(
        self,
        projectPath: Project,
        params: Params,
        deviceName: str = "cpu",
        debug: bool = False,
        phase: Optional[str] = None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        grid_size = getattr(params, "GaussianGridSize", DEFAULT_GRIDSIZE)

        # Handle formatting of device name string
        self.deviceName = self._normalize_device_string(deviceName)
        self.device = torch.device(self.deviceName)

        if getattr(params, "usingMixedPrecision", False):
            print(f"Using mixed precision (float16) on device {self.device}")
            self.dtype = torch.float16
        else:
            print(f"Using float32 on device {self.device}")
            self.dtype = torch.float32

        if kwargs.get("linearizer", None) is not None:
            self.Linearizer = kwargs["linearizer"]
            self.fix_linearizer(self.Linearizer.mazePoints, self.Linearizer.tsProj)
            self.maze_params = self.Linearizer.maze_params
        else:
            self.maze_points = None
            self.ts_proj = None
            self.mazePoints_tensor = None
            self.tsProjTensor = None
            self.maze_params = None

        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=self.maze_params, **kwargs
        )

        self.projectPath = projectPath
        self.params = params
        self.debug = debug
        self.target = params.target
        self.phase = phase
        self.suffix = "_" + str(phase) if phase is not None else ""

        self._setup_folders()
        self._setup_feature_description()

        self.preprocess_normalization = kwargs.get(
            "normalize_in_pipeline",
            getattr(self.params, "normalize_in_pipeline", True),
        )
        self.normalization_stats = None
        self.learnable_contrastive_temperature = kwargs.get(
            "learnable_contrastive_temperature",
            getattr(self.params, "learnable_contrastive_temperature", True),
        )

        self.max_nb_spikes = kwargs.get(
            "max_nb_spikes", getattr(self.params, "max_nb_spikes", 400)
        )
        if self.max_nb_spikes is None:
            self.max_nb_spikes = 400

        self.max_spikes_per_group = kwargs.get("max_spikes_per_group", None)
        if self.max_spikes_per_group is None:
            self.max_spikes_per_group = int(self.max_nb_spikes / self.params.nGroups)

        self._parse_target_structure()

        self.zeroForGather = torch.zeros(
            [1, self.params.nFeatures], dtype=self.dtype, device=self.device
        )

        if params.denseweight:
            if kwargs.get("behaviorData", None) is None:
                warnings.warn(
                    '"behaviorData" not provided, using default setup WITHOUT Dense Weight.'
                )
            else:
                self.setup_dynamic_dense_loss(**kwargs)
        else:
            self.setup_training_data(**kwargs)

        if (
            getattr(params, "GaussianHeatmap", False)
            or getattr(params, "OversamplingResampling", False)
            or getattr(params, "contrastive_loss", False)
        ):
            assert not params.denseweight, (
                "Cannot use both GaussianHeatmap and DenseWeight"
            )
            if kwargs.get("behaviorData", None) is None:
                warnings.warn(
                    '"behaviorData" not provided, using default setup WITHOUT Gaussian Heatmap layering.'
                )
            else:
                self.lfunction_layer_params = {
                    "maze_points": self.maze_points,
                    "ts_proj": self.ts_proj,
                    "device": self.deviceName,
                }
                self.setup_gaussian_heatmap(**kwargs)
        else:
            self.gaussian_heatmap_params = None
            self.lfunction_layer_params = None

        self._build_model(**kwargs)

    def _normalize_device_string(self, dev_str: str) -> str:
        if "CPU" in dev_str or "cpu" in dev_str:
            return "cpu"
        if "GPU" in dev_str or "gpu" in dev_str or "cuda" in dev_str:
            if ":" in dev_str:
                gpu_num = dev_str.split(":")[-1]
                return f"cuda:{gpu_num}"
            return "cuda:0"
        return "cpu"

    def set_input_normalization_mode(self, enabled: bool):
        for spike_net in getattr(self, "spikeNets", []):
            spike_net.apply_input_normalization = enabled

    def _setup_folders(self):
        self.folderResult = self.projectPath.folderResult
        try:
            self.folderResultSleep = self.projectPath.folderResultSleep
        except AttributeError:
            self.folderResultSleep = os.path.join(
                self.projectPath.experimentPath, "results_Sleep"
            )
            self.projectPath.folderResultSleep = self.folderResultSleep
        self.folderModels = os.path.join(self.projectPath.experimentPath, "models")
        os.makedirs(self.folderResult, exist_ok=True)
        os.makedirs(self.folderResultSleep, exist_ok=True)
        os.makedirs(self.folderModels, exist_ok=True)

    def _setup_feature_description(self):
        self.featDesc = {
            "pos_index": "int",
            "pos": "float",
            "length": "int",
            "groups": "int",
            "time": "float",
            "time_behavior": "float",
            "indexInDat": "int",
        }
        for g in range(self.params.nGroups):
            self.featDesc[f"group{g}"] = "float"
        self.trainLosses = {}

    def _parse_target_structure(self):
        target = self.params.target.lower()
        use_heatmap = getattr(self.params, "GaussianHeatmap", False)

        scaled_sigmoid_activation = nnUtils.ScaledSigmoid(
            high=getattr(self.params, "high_rad", 2 * np.pi)
        )

        pos_dim_out = 2
        if use_heatmap:
            pos_dim_out = (
                self.params.GaussianGridSize[0] * self.params.GaussianGridSize[1]
            )

        structure = {}

        if target == "pos":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "sigmoid"
                if not use_heatmap
                else getattr(self.params, "featureActivation", "linear"),
            }
        elif target in ["lin", "linear"]:
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "relu"}
        elif target == "linandthigmo":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "relu"}
            structure["thigmo"] = {"dim": 1, "slice": (1, 2), "activation": "sigmoid"}
        elif target == "linanddirection":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "relu"}
            structure["direction"] = {
                "dim": 1,
                "slice": (1, 2),
                "activation": "sigmoid",
            }
        elif target == "direction":
            structure["direction"] = {
                "dim": 1,
                "slice": (0, 1),
                "activation": "sigmoid",
            }
        elif target == "linandheaddirection":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "relu"}
            structure["hd"] = {
                "dim": 1,
                "slice": (1, 2),
                "activation": scaled_sigmoid_activation,
            }
        elif target == "linandspeed":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "relu"}
            structure["speed"] = {"dim": 1, "slice": (1, 2), "activation": "relu"}
        elif target == "posanddirection":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "sigmoid"
                if not use_heatmap
                else getattr(self.params, "featureActivation", "linear"),
            }
            structure["direction"] = {
                "dim": 1,
                "slice": (2, 3),
                "activation": "sigmoid",
            }
        elif target == "posandheaddirection":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "sigmoid"
                if not use_heatmap
                else getattr(self.params, "featureActivation", "linear"),
            }
            structure["hd"] = {
                "dim": 1,
                "slice": (2, 3),
                "activation": scaled_sigmoid_activation,
            }
        elif target == "posandspeed":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "sigmoid"
                if not use_heatmap
                else getattr(self.params, "featureActivation", "linear"),
            }
            structure["speed"] = {"dim": 1, "slice": (2, 3), "activation": "relu"}
        else:
            structure["main_pred"] = {
                "dim": self.params.dimOutput,
                "slice": (0, self.params.dimOutput),
                "activation": getattr(self.params, "featureActivation", "linear"),
            }

        if getattr(self.params, "contrastive_loss", False):
            all_slices = [s["slice"] for s in structure.values()]
            if all_slices:
                start_idx = min(s[0] for s in all_slices)
                end_idx = max(s[1] for s in all_slices)
                dim = getattr(self.params, "contrastive_dim", 128)
                structure["latent"] = {
                    "dim": dim,
                    "slice": (start_idx, end_idx),
                    "activation": "linear",
                }

        self.target_structure = structure

    def _build_model(self, **kwargs):
        conv_dim = 2 if getattr(self.params, "use_conv2d", False) else 1
        spikeNetClass = (
            nnUtils.SpikeNet2DTorch if conv_dim == 2 else nnUtils.SpikeNet1DTorch
        )

        self.spikeNets = nn.ModuleList(
            [
                spikeNetClass(
                    nChannels=self.params.nChannelsPerGroup[group],
                    device=self.deviceName,
                    nFeatures=self.params.nFeatures,
                    number=str(group),
                    batch_normalization=False,
                    reduce_dense=getattr(self.params, "reduce_dense", False),
                    no_cnn=getattr(self.params, "no_cnn", False),
                    apply_input_normalization=not self.preprocess_normalization,
                    name=f"spikeNet_{group}",
                )
                for group in range(self.params.nGroups)
            ]
        )

        self.spike_encoder = nnUtils.SpikeEncoder(
            self.spikeNets,
            self.params,
            self.max_nb_spikes,
            self.max_spikes_per_group,
            conv_dim,
        )
        self.spike_sequence_processor = nnUtils.SpikeSequenceProcessor(
            spike_encoder=self.spike_encoder,
            n_groups=self.params.nGroups,
            n_features=self.params.nFeatures,
            max_spikes_per_group=self.max_spikes_per_group,
            max_nb_spikes=self.max_nb_spikes,
            device=self.deviceName,
            name="spike_sequence_processor",
        )

        self.dropoutLayer = nn.Dropout(kwargs.get("dropoutCNN", self.params.dropoutCNN))
        self.lstmdropOutLayer = nn.Dropout(
            kwargs.get("dropoutLSTM", self.params.dropoutLSTM)
        )

        self.isTransformer = kwargs.get(
            "isTransformer", getattr(self.params, "isTransformer", False)
        )

        if (
            not hasattr(self.params, "sequence_output_dim")
            or self.params.sequence_output_dim is None
        ):
            if self.isTransformer:
                self.params.sequence_output_dim = getattr(
                    self.params, "nFeatures", 64
                ) * getattr(self.params, "dim_factor", 1)
            else:
                self.params.sequence_output_dim = getattr(
                    self.params, "lstmSize", getattr(self.params, "nFeatures", 64)
                )

        if not self.isTransformer:
            self.sequence_model = nn.LSTM(
                input_size=self.params.nFeatures,
                hidden_size=self.params.lstmSize,
                batch_first=True,
            )
        else:
            self.dim_factor = getattr(self.params, "dim_factor", 1)
            self.sequence_model = nnUtils.TransformerEncoderBlock(
                d_model=self.params.sequence_output_dim,
                num_heads=getattr(self.params, "num_heads", 8),
                ff_dim1=getattr(self.params, "ff_dim", 256),
                dropout_rate=kwargs.get("dropoutLSTM", self.params.dropoutLSTM),
                device=self.deviceName,
            )

        self.output_heads = nn.ModuleDict()
        for name, spec in self.target_structure.items():
            if name == "pos_2d" and getattr(self.params, "GaussianHeatmap", False):
                out_dim = int(np.prod(self.params.GaussianGridSize))
            else:
                out_dim = int(spec["dim"])

            # Map dynamic activations to native modules
            act_name = spec["activation"]
            if isinstance(act_name, str):
                if act_name == "sigmoid":
                    act_layer = nn.Sigmoid()
                elif act_name == "relu":
                    get_activation = nn.ReLU()
                else:
                    act_layer = nn.Identity()
            else:
                act_layer = act_name

            self.output_heads[name] = nn.Sequential(
                nn.Linear(self.params.sequence_output_dim, out_dim), act_layer
            )

        if self.learnable_contrastive_temperature:
            self.contrastive_temperature_layer = nnUtils.LearnableTemperature(
                initial_temperature=getattr(self.params, "temperature", 0.07),
                min_temperature=getattr(
                    self.params, "contrastive_temperature_floor", 0.03
                ),
                max_temperature=getattr(
                    self.params, "contrastive_temperature_max", 0.2
                ),
            )
        else:
            self.contrastive_temperature_layer = None

        if getattr(self.params, "project_transformer", True) and self.isTransformer:
            self.transformer_projection_layer = nn.Linear(
                self.params.sequence_output_dim,
                self.params.nFeatures * self.dim_factor,
            )

        self.ProjectionInMazeLayer = nnUtils.UMazeProjectionLayer(
            grid_size=kwargs.get(
                "grid_size", getattr(self.params, "GaussianGridSize", DEFAULT_GRIDSIZE)
            ),
            maze_params=self.maze_params,
        )
        self.to(self.device)

    def forward(self, batch_inputs):
        # Flatten batch keys into sorted parameters list for sequence processor
        processor_inputs = [
            batch_inputs[f"group{g}"] for g in range(self.params.nGroups)
        ]
        processor_inputs += [
            batch_inputs[f"indices{g}"] for g in range(self.params.nGroups)
        ]
        processor_inputs.append(batch_inputs["groups"])

        masked_features, mymask, sum_features, all_features = (
            self.spike_sequence_processor(processor_inputs)
        )
        features = self.dropoutLayer(masked_features)

        if not self.isTransformer:
            hidden, _ = self.sequence_model(features)
            latent_output = hidden[:, -1, :]  # Global sequence reduction pooling step
        else:
            latent_output = self.sequence_model(features, mask=mymask)
            if len(latent_output.shape) > 2:
                latent_output = torch.mean(latent_output, dim=1)

        outputs = {}
        for name, head in self.output_heads.items():
            pred = head(latent_output)
            if (
                name == "pos_2d"
                and "pos" in self.params.target.lower()
                and not getattr(self.params, "GaussianHeatmap", False)
            ):
                pred = self.ProjectionInMazeLayer(pred)
            outputs[name] = pred

        if self.contrastive_temperature_layer is not None and "latent" in outputs:
            temp = self.contrastive_temperature_layer(latent_output)
            outputs["latent"] = torch.cat([outputs["latent"], temp], dim=-1)

        outputs["latent_output"] = latent_output
        return outputs

    def train_model(self, dataset_dict, epochs=10, lr=1e-3, **kwargs):
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=getattr(self.params, "weight_decay", 1e-4),
        )

        # Native execution loop maps criterion targets directly from target_structure logic
        loss_dict = {}
        for name, spec in self.target_structure.items():
            if name == "pos_2d" and getattr(self.params, "GaussianHeatmap", False):
                loss_dict[name] = nnUtils.GaussianHeatmapLosses(
                    l_function_layer_params=self.lfunction_layer_params,
                    grid_size=self.params.GaussianGridSize,
                    maze_params=self.maze_params,
                    device=self.deviceName,
                )
            elif name == "latent":
                behav_structure = {
                    k: v for k, v in self.target_structure.items() if k != "latent"
                }
                loss_dict[name] = nnUtils.ContrastiveRegressionLoss(
                    target_structure=behav_structure,
                    temperature=getattr(self.params, "temperature", 0.07),
                    sigma=getattr(self.params, "sigma_contrastive", 0.05),
                    l_function_params=self.lfunction_layer_params,
                    learnable_temperature=self.learnable_contrastive_temperature,
                )
            elif name == "hd":
                loss_dict[name] = nnUtils.CyclicMAE(
                    high=getattr(self.params, "high_rad", 2 * np.pi)
                )
            else:
                loss_dict[name] = nn.L1Loss()

        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(dataset_dict["train"], desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = {
                    k: torch.as_tensor(v, device=self.device) for k, v in inputs.items()
                }
                targets = {
                    k: torch.as_tensor(v, device=self.device)
                    for k, v in targets.items()
                }

                optimizer.zero_grad()
                outputs = self(inputs)

                total_loss = 0.0
                for name, criterion in loss_dict.items():
                    if name in outputs and name in targets:
                        total_loss += criterion(outputs[name], targets[name]).mean()

                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
                pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})

    def fix_linearizer(self, mazePoints, tsProj):
        self.maze_points = mazePoints
        self.ts_proj = tsProj
        self.mazePoints_tensor = torch.tensor(
            mazePoints[None, :], dtype=self.dtype, device=self.device
        )
        self.tsProjTensor = torch.tensor(
            tsProj[None, :], dtype=self.dtype, device=self.device
        )

    def setup_training_data(self, **kwargs):
        behaviorData = kwargs.get("behaviorData", None)
        if behaviorData is None:
            return
        speedMask = behaviorData["Times"]["speedFilter"]
        epochMask = inEpochsMask(
            behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
        )
        self.training_data = behaviorData["Positions"][speedMask * epochMask, :2]

    def setup_dynamic_dense_loss(self, **kwargs):
        self.dense_loss_processor = nnUtils.DenseLossProcessor(
            maze_points=self.maze_points,
            ts_proj=self.ts_proj,
            alpha=kwargs.get("alpha", 1.3),
            device=self.deviceName,
        )
        self.setup_training_data(**kwargs)
        self.dense_loss_processor.fit_dense_weight_model(self.training_data)

    def setup_gaussian_heatmap(self, **kwargs):
        behaviorData = kwargs.get("behaviorData", None)
        self.GaussianHeatmap = nnUtils.GaussianHeatmapLayer(
            training_positions=self.training_data,
            grid_size=getattr(self.params, "GaussianGridSize", DEFAULT_GRIDSIZE),
            sigma=getattr(self.params, "GaussianSigma", 0.03),
            maze_params=self.maze_params,
        )

    def saveResults(
        self, test_output, folderName=36, sleep=False, sleepName="Sleep", phase=None
    ):
        folderToSave = os.path.join(
            self.folderResultSleep if sleep else self.folderResult, str(folderName)
        )
        if sleep:
            folderToSave = os.path.join(folderToSave, sleepName)
        os.makedirs(folderToSave, exist_ok=True)
        suffix = f"_{phase}" if phase else self.suffix

        import pandas as pd

        for k, v in test_output.items():
            if isinstance(v, np.ndarray) and v.ndim <= 2:
                pd.DataFrame(v).to_csv(os.path.join(folderToSave, f"{k}{suffix}.csv"))
