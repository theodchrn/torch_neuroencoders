import os
from unittest.mock import MagicMock

import numpy as np
from torch_neuroencoders.utils.backend import pd
import pytest

from torch_neuroencoders.fullEncoder.an_network import LSTMandSpikeNetwork as TFNet
from torch_neuroencoders.fullEncoder.nnUtils_torch import load_tfrecord_examples


@pytest.fixture
def mock_project(tmp_path):
    # Using tmp_path fixture from pytest
    project = MagicMock()
    project.folder = str(tmp_path)
    project.folderModels = os.path.join(project.folder, "models")
    os.makedirs(project.folderModels, exist_ok=True)
    return project


@pytest.fixture
def mock_params():
    params = MagicMock()
    params.batch_size = 2
    params.nGroups = 1
    params.nChannelsPerGroup = [5]
    params.stride = 36
    params.windowSizeMS = 100
    params.dimOutput = 2
    params.GaussianHeatmap = False
    params.usingMixedPrecision = False
    params.max_nb_spikes = 512
    params.max_nb_spikes_per_group = 512
    return params


def test_save_datasets_isolated(mock_params, mock_project):
    # This test verifies the saving methods in isolation
    # We create a mock network object to avoid complex __init__
    model_obj = MagicMock(spec=TFNet)
    # Re-bind the real methods to the mock object
    model_obj._save_datasets_to_tfrec = TFNet._save_datasets_to_tfrec.__get__(
        model_obj, TFNet
    )
    model_obj._save_datasets_to_parquet = TFNet._save_datasets_to_parquet.__get__(
        model_obj, TFNet
    )
    model_obj.convert_tfrec_to_pandas = TFNet.convert_tfrec_to_pandas.__get__(
        model_obj, TFNet
    )
    model_obj.params = mock_params

    # Mock data with matching dimensions for samples
    # For one sample: pos (2,), groups (N,), group0 (N*32*32)
    data = {
        "pos": np.random.rand(2).astype(np.float32),
        "group0": np.random.rand(10 * 5 * 32).astype(np.float32),
        "pos_index": np.asarray(1, dtype=np.int64),
        "groups": np.asarray([0] * 10, dtype=np.int64),
        "indexInDat": np.asarray([123] * 10, dtype=np.int64),
    }

    datasets = {"train": [data]}

    base_tfrec = os.path.join(mock_project.folder, "isolated_saved")
    base_parquet = os.path.join(mock_project.folder, "isolated_saved")

    # Test TFRecord saving
    model_obj._save_datasets_to_tfrec(datasets, base_tfrec)
    assert os.path.exists(f"{base_tfrec}_train.tfrec")

    # Verify TFRecord can be read back
    parsed_dataset = list(
        load_tfrecord_examples(
            f"{base_tfrec}_train.tfrec", {"pos": "float", "group0": "float"}
        )
    )
    assert len(parsed_dataset) == 1
    assert "pos" in parsed_dataset[0]
    assert "group0" in parsed_dataset[0]

    # Test Parquet saving
    model_obj._save_datasets_to_parquet(datasets, base_parquet)
    assert os.path.exists(f"{base_parquet}_train.parquet")

    # Verify Parquet content
    df = pd.read_parquet(f"{base_parquet}_train.parquet")
    assert "pos" in df.columns
    assert len(df) == 1
    # Verify flattening (group0 should be a 1D array/list in the cell)
    assert len(df["group0"].iloc[0]) == 10 * 5 * 32


def test_load_parsed_dataset(mock_params, mock_project):
    # This test verifies the torch TFRecord round-trip behavior.
    model_obj = MagicMock(spec=TFNet)
    model_obj._save_datasets_to_tfrec = TFNet._save_datasets_to_tfrec.__get__(
        model_obj, TFNet
    )
    model_obj.params = mock_params
    model_obj.params.nGroups = 2  # Test with 2 groups
    model_obj.params.nChannelsPerGroup = [18, 5]  # define for both groups
    model_obj.max_nb_spikes = 512

    # Mock data for 2 groups
    data = {
        "pos_index": np.asarray(1, dtype=np.int64),
        "pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),  # 3D position
        "length": np.asarray(5, dtype=np.int64),
        "groups": np.asarray([0, 1, 0, 1, 0], dtype=np.int64),
        "time": np.asarray(100.5, dtype=np.float32),
        "time_behavior": np.asarray(100.6, dtype=np.float32),
        "indexInDat": np.asarray([10, 20, 30, 40, 50], dtype=np.int64),
        "group0": np.random.rand(3 * 18 * 32).astype(np.float32),
        "group1": np.random.rand(2 * 5 * 32).astype(np.float32),
    }

    datasets = {"train": [data]}

    base_path = os.path.join(mock_project.folder, "loading_test")

    # Save it first
    model_obj._save_datasets_to_tfrec(datasets, base_path)

    loaded = list(
        load_tfrecord_examples(
            f"{base_path}_train.tfrec",
            {
                "pos_index": "int",
                "pos": "float",
                "length": "int",
                "groups": "int",
                "time": "float",
                "time_behavior": "float",
                "indexInDat": "int",
                "group0": "float",
                "group1": "float",
            },
        )
    )

    assert len(loaded) == 1
    batch = loaded[0]
    assert batch["pos"].shape == (3,)
    assert np.allclose(batch["pos"], [0.1, 0.2, 0.3])
    assert batch["length"].shape == (1,)
    assert batch["groups"].shape[0] == 5
    assert batch["group0"].shape[0] == 3 * 18 * 32
    assert batch["group1"].shape[0] == 2 * 5 * 32
