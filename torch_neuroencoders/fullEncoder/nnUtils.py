"""
Neuroencoders: nnUtils
Native PyTorch layers, loss profiles, metrics, and augmentation utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from denseweight import DenseWeight

from torch_neuroencoders.utils.global_classes import (
    DEFAULT_GRIDSIZE,
    Params,
    SpatialConstraintsMixin,
)


def standardize_channelwise_tensor(
    tensor: torch.Tensor,
    mean: Union[np.ndarray, torch.Tensor],
    variance: Union[np.ndarray, torch.Tensor],
    axis: int = 1,
    preserve_zero_rows: bool = True,
) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    rank = tensor.ndim
    axis = axis if axis >= 0 else rank + axis

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.sqrt(
        torch.as_tensor(variance, dtype=tensor.dtype, device=tensor.device) + 1e-8
    )

    broadcast_shape = [1] * rank
    broadcast_shape[axis] = mean.shape[0]
    mean = mean.reshape(broadcast_shape)
    std = std.reshape(broadcast_shape)

    standardized = (tensor - mean) / std
    if not preserve_zero_rows:
        return standardized

    spike_mask = tensor.ne(0.0).any(dim=tuple(range(1, rank)), keepdim=True)
    return torch.where(spike_mask, standardized, torch.zeros_like(standardized))


class ChannelwiseFixedNormalization(nn.Module):
    def __init__(self, channels: int, axis: int = 1, epsilon: float = 1e-6):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(channels, dtype=torch.float32))
        self.register_buffer("variance", torch.ones(channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = x.ndim
        axis = self.axis if self.axis >= 0 else rank + self.axis
        broadcast_shape = [1] * rank
        broadcast_shape[axis] = self.mean.shape[0]

        std = torch.sqrt(self.variance.reshape(broadcast_shape) + self.epsilon)
        return (x - self.mean.reshape(broadcast_shape)) / std


class AddNullSpike(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        batch_size = e.shape[0]
        null_spike = torch.zeros(
            (batch_size, 1, self.n_features), dtype=e.dtype, device=e.device
        )
        return torch.cat([null_spike, e], dim=1)


class GlobalSequenceGather(nn.Module):
    def __init__(
        self, n_groups: int, group_dim: int, offsets: List[int], max_nb_spikes: int
    ):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = group_dim
        self.max_nb_spikes = max_nb_spikes
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.long))
        self.group_embeddings = nn.Parameter(torch.empty(self.n_groups, self.group_dim))
        nn.init.xavier_uniform_(self.group_embeddings)
        self.register_buffer("null_identity", torch.zeros(1, self.group_dim))

    def forward(
        self,
        pool: torch.Tensor,
        indices_list: List[torch.Tensor],
        group_sequence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = group_sequence.shape[:2]
        stacked_indices = torch.stack(indices_list, dim=0)
        options = stacked_indices + self.offsets[:, None, None]

        safe_group_seq = torch.where(
            group_sequence.eq(-1), torch.zeros_like(group_sequence), group_sequence
        )
        group_mask = F.one_hot(safe_group_seq, num_classes=self.n_groups).permute(
            2, 0, 1
        )
        global_indices = torch.sum(options * group_mask, dim=0).to(torch.long)

        batch_offsets = (
            torch.arange(batch_size, device=pool.device, dtype=torch.long)
            * pool.shape[1]
        )
        flat_global_indices = (global_indices + batch_offsets[:, None]).reshape(-1)

        flat_pool = pool.reshape(-1, self.group_dim)
        sequence_features = flat_pool.index_select(0, flat_global_indices).reshape(
            batch_size, seq_len, self.group_dim
        )

        lookup_table = torch.cat(
            [self.group_embeddings, self.null_identity.to(self.group_embeddings.dtype)],
            dim=0,
        )
        safe_ids = torch.where(
            group_sequence.eq(-1),
            torch.full_like(group_sequence, self.n_groups),
            group_sequence,
        )
        identities = lookup_table.index_select(0, safe_ids.reshape(-1)).reshape(
            batch_size, seq_len, self.group_dim
        )

        return sequence_features + identities


class SpikeNet1DTorch(nn.Module):
    def __init__(
        self,
        nChannels=4,
        device="cpu",
        nFeatures=128,
        number="",
        dropout_rate=0.2,
        apply_input_normalization=True,
        **kwargs,
    ):
        super().__init__()
        self.nFeatures = nFeatures
        self.apply_input_normalization = apply_input_normalization
        self.no_cnn = kwargs.get("no_cnn", False)

        self.input_normalization = (
            ChannelwiseFixedNormalization(channels=nChannels)
            if apply_input_normalization
            else None
        )
        self.conv1 = nn.Conv1d(nChannels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense_out = nn.Linear(64, nFeatures)
        self.no_cnn_head = nn.LazyLinear(nFeatures)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.input_normalization and self.apply_input_normalization:
            x = self.input_normalization(x)

        if self.no_cnn:
            x = self.no_cnn_head(x.flatten(1))
            return self.dropout(x)

        x = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dense_out(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class SpikeNet2DTorch(nn.Module):
    def __init__(
        self,
        nChannels=4,
        device="cpu",
        nFeatures=128,
        number="",
        apply_input_normalization=True,
        **kwargs,
    ):
        super().__init__()
        self.nFeatures = nFeatures
        self.apply_input_normalization = apply_input_normalization
        self.no_cnn = kwargs.get("no_cnn", False)

        self.input_normalization = (
            ChannelwiseFixedNormalization(channels=nChannels)
            if apply_input_normalization
            else None
        )
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(2, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(2, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(2, 3), padding=(1, 1))
        self.dropout = nn.Dropout(0.2)
        self.dense_out = nn.Linear(32, nFeatures)
        self.no_cnn_head = nn.LazyLinear(nFeatures)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.input_normalization and self.apply_input_normalization:
            x = self.input_normalization(x)

        if self.no_cnn:
            x = self.no_cnn_head(x.flatten(1))
            return self.dropout(x)

        x = x.unsqueeze(1)  # Add single channel dim
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(1, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(1, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(1, 2))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dense_out(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class SpikeEncoder(nn.Module):
    def __init__(
        self,
        spikeNets: nn.ModuleList,
        params: Params,
        max_nb_spikes: int,
        max_spikes_per_group: int,
        conv_dim: int,
    ):
        super().__init__()
        self.spikeNets = spikeNets
        self.params = params
        self.max_spikes_per_group = max_spikes_per_group

    def forward(
        self, inputs: List[torch.Tensor], masks: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        encoded_groups = []
        for g, net in enumerate(self.spikeNets):
            group_input = inputs[g]
            n_ch = self.params.nChannelsPerGroup[g]

            # Collapse batch and spike dimension to process simultaneously
            x = group_input.reshape(-1, n_ch, 32)
            g_mask = masks[g].reshape(-1) if masks is not None else None

            out = net(x, mask=g_mask)
            encoded_groups.append(
                out.reshape(
                    group_input.shape[0],
                    self.max_spikes_per_group,
                    self.params.nFeatures,
                )
            )
        return encoded_groups


class SpikeSequenceProcessor(nn.Module):
    def __init__(
        self,
        spike_encoder: SpikeEncoder,
        n_groups: int,
        n_features: int,
        max_spikes_per_group: int,
        max_nb_spikes: int,
        device="cpu",
        name="",
    ):
        super().__init__()
        self.spike_encoder = spike_encoder
        self.n_groups = n_groups
        self.n_features = n_features
        self.max_spikes_per_group = max_spikes_per_group
        self.max_nb_spikes = max_nb_spikes

        offsets = []
        curr_offset = 0
        for g in range(self.n_groups):
            offsets.append(curr_offset)
            curr_offset += self.max_spikes_per_group + 1

        self.add_null_spike_layers = nn.ModuleList(
            [AddNullSpike(n_features) for _ in range(n_groups)]
        )
        self.sequence_reconstructor = GlobalSequenceGather(
            n_groups, n_features, offsets, max_nb_spikes
        )

    def forward(
        self, inputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_to_spike_nets = inputs[: self.n_groups]
        indices = inputs[self.n_groups : 2 * self.n_groups]
        input_groups = inputs[-1]

        all_group_masks = [
            group_data.ne(0.0).any(dim=-1).any(dim=-1)
            for group_data in inputs_to_spike_nets
        ]
        group_latents_raw = self.spike_encoder(
            inputs_to_spike_nets, masks=all_group_masks
        )

        all_group_latents = [
            self.add_null_spike_layers[g](latent)
            for g, latent in enumerate(group_latents_raw)
        ]
        pool = torch.cat(all_group_latents, dim=1)
        all_features = self.sequence_reconstructor(pool, indices, input_groups)

        mymask = input_groups.ne(-1)
        masked_features = torch.where(
            mymask.unsqueeze(-1), all_features, torch.zeros_like(all_features)
        )
        sum_features = masked_features.sum(dim=1)

        return masked_features, mymask, sum_features, all_features


class LinearizationLayer(nn.Module):
    def __init__(self, maze_points, ts_proj, device="cpu", name=""):
        super().__init__()
        self.register_buffer(
            "maze_points", torch.as_tensor(maze_points, dtype=torch.float32)
        )
        self.register_buffer("ts_proj", torch.as_tensor(ts_proj, dtype=torch.float32))

    def forward(
        self, euclidean_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        euclidean_expanded = euclidean_data.unsqueeze(1)  # [B, 1, 2]
        maze_expanded = self.maze_points.unsqueeze(0)  # [1, J, 2]

        distances = torch.sum((maze_expanded - euclidean_expanded) ** 2, dim=-1)
        best_points = torch.argmin(distances, dim=1)

        return self.maze_points[best_points], self.ts_proj[best_points]


class UMazeProjectionLayer(nn.Module, SpatialConstraintsMixin):
    def __init__(self, grid_size, smoothing_factor=0.01, maze_params=None, **kwargs):
        super().__init__()
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        self.smoothing_factor = smoothing_factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, y = inputs[..., 0], inputs[..., 1]

        gap_x_min = self.maze_params_dict["gap_x_min"]
        gap_x_max = self.maze_params_dict["gap_x_max"]
        gap_y_min = self.maze_params_dict["gap_y_min"]

        inside_soft = (
            torch.sigmoid((gap_x_max - x) / self.smoothing_factor)
            * torch.sigmoid((x - gap_x_min) / self.smoothing_factor)
            * torch.sigmoid((gap_y_min - y) / self.smoothing_factor)
        )

        # Soft boundary alignment handling
        x_proj = torch.clamp(
            x, self.maze_params_dict["x_min"], self.maze_params_dict["x_max"]
        )
        y_proj = torch.clamp(
            y, self.maze_params_dict["y_min"], self.maze_params_dict["y_max"]
        )

        proj = torch.stack(
            [
                torch.where(inside_soft > 0.5, gap_x_min, x_proj),
                torch.where(inside_soft > 0.5, gap_y_min, y_proj),
            ],
            dim=-1,
        )

        return torch.cat([proj, inputs[..., 2:]], dim=-1)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model=64, num_heads=8, ff_dim1=256, dropout_rate=0.5, device="cpu"
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ff_layer1 = nn.Linear(d_model, ff_dim1)
        self.ff_layer2 = nn.Linear(ff_dim1, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_norm = self.norm1(x)

        attn_mask = None
        if mask is not None:
            # Attention padding masks look up inverse boolean allocations in PyTorch MHA
            attn_mask = ~mask

        attn_output, _ = self.mha(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        x = x + self.dropout1(attn_output)

        x_norm2 = self.norm2(x)
        ff_output = self.ff_layer2(F.gelu(self.ff_layer1(x_norm2)))
        return x + ff_output


class LearnableTemperature(nn.Module):
    def __init__(
        self, initial_temperature=0.07, min_temperature=0.03, max_temperature=0.2
    ):
        super().__init__()
        self.min_temp = min_temperature
        self.max_temp = max_temperature
        inv_softplus = np.log(np.exp(initial_temperature) - 1.0)
        self.log_temperature = nn.Parameter(
            torch.tensor(inv_softplus, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temp = F.softplus(self.log_temperature)
        temp = torch.clamp(temp, self.min_temp, self.max_temp)
        return torch.full(
            (inputs.shape[0], 1), temp, dtype=inputs.dtype, device=inputs.device
        )


class CyclicMAE(nn.Module):
    def __init__(self, high=2 * np.pi):
        super().__init__()
        self.high = high

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        delta = y_pred - y_true
        half = self.high / 2.0
        return torch.abs(torch.remainder(delta + half, self.high) - half)


class ScaledSigmoid(nn.Module):
    def __init__(self, high=2 * np.pi):
        super().__init__()
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.high


class GaussianHeatmapLayer(SpatialConstraintsMixin):
    def __init__(self, training_positions, grid_size, sigma=0.03, maze_params=None):
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        self.sigma = sigma
        self.GRID_H, self.GRID_W = grid_size


class GaussianHeatmapLosses(nn.Module, SpatialConstraintsMixin):
    def __init__(
        self, l_function_layer_params, grid_size=DEFAULT_GRIDSIZE, maze_params=None
    ):
        super().__init__()
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        self.GRID_H, self.GRID_W = grid_size
        self.allowed_mask = torch.as_tensor(
            self.get_allowed_mask(use_tensorflow=False), dtype=torch.float32
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.ndim == 2:
            y_pred = y_pred.reshape(-1, self.GRID_H, self.GRID_W)

        # Safe KL Implementation on distribution masks
        mask = self.allowed_mask.to(y_pred.device)
        log_q = F.log_softmax(y_pred.reshape(y_pred.shape[0], -1), dim=-1).reshape(
            y_pred.shape
        )

        loss = -torch.sum(y_true * log_q * mask, dim=(1, 2))
        return loss.mean()


class ContrastiveRegressionLoss(nn.Module):
    def __init__(
        self,
        target_structure: Dict,
        temperature: float = 0.1,
        sigma: float = 0.1,
        l_function_params: Optional[Dict] = None,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        self.target_structure = target_structure
        self.temperature = temperature
        self.sigma = sigma
        self.learnable_temperature = learnable_temperature
        self.l_function = (
            LinearizationLayer(**l_function_params) if l_function_params else None
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        N = y_pred.shape[0]
        if self.learnable_temperature:
            temp = F.softplus(y_pred[:, -1:]) + 1e-3
            z = F.normalize(y_pred[:, :-1], p=2, dim=1)
        else:
            temp = self.temperature
            z = F.normalize(y_pred, p=2, dim=1)

        # Build soft anchor coordinates match map using positions matching target structure slices
        with torch.no_grad():
            pos_slice = self.target_structure["pos_2d"]["slice"]
            pos = y_true[:, pos_slice[0] : pos_slice[1]]
            if self.l_function:
                _, lin_pos = self.l_function(pos)
                pos = lin_pos.unsqueeze(-1)

            d2 = torch.cdist(pos, pos, p=2) ** 2
            w_norm = torch.exp(-0.5 * d2 / (self.sigma**2))
            w_norm.fill_diagonal_(0.0)
            w_norm = w_norm / (w_norm.sum(dim=1, keepdim=True) + 1e-8)

        logits = torch.matmul(z, z.t()) / temp
        logits.fill_diagonal_(-1e4)

        log_prob = F.log_softmax(logits, dim=1)
        return -torch.sum(w_norm * log_prob, dim=1).mean()


class DenseLossProcessor:
    def __init__(self, maze_points, ts_proj, alpha=1.0, device="cpu"):
        self.maze_points = maze_points
        self.ts_proj = ts_proj
        self.alpha = alpha
        self.linearizer = LinearizationLayer(maze_points, ts_proj)
        self.fitted_dw = None

    def fit_dense_weight_model(self, full_training_positions):
        training_pos_t = torch.as_tensor(full_training_positions, dtype=torch.float32)
        _, lin_output = self.linearizer(training_pos_t)

        self.fitted_dw = DenseWeight(alpha=self.alpha)
        self.fitted_dw.fit(lin_output.cpu().numpy())
