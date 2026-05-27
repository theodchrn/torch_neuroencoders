import numpy as np
import pytest
import torch

from torch_neuroencoders.fullEncoder.nnUtils_torch import parse_serialized_sequence_torch


def create_dummy_parsed_tensors(params):
    tensors = {
        "pos": np.random.rand(params.dimOutput).astype(np.float32),
        "groups": np.array([0, 1, 0], dtype=np.int64),
        "indexInDat": np.array([10, 20, 30], dtype=np.int64),
    }

    for g in range(params.nGroups):
        # Create non-zero spike for group g
        num_spikes = 2
        spike_data = (
            np.random.rand(num_spikes * params.nChannelsPerGroup[g] * 32)
            .astype(np.float32)
            .reshape(-1)
        )

        tensors[f"group{g}"] = spike_data

    return tensors


def test_parse_serialized_sequence(mock_params):
    tensors = create_dummy_parsed_tensors(mock_params)

    # Call the torch-native parser
    parsed = parse_serialized_sequence_torch(mock_params, tensors, count_spikes=True)

    # Check pos
    assert isinstance(parsed["pos"], torch.Tensor)
    assert parsed["pos"].shape == (3,)

    # Check groups
    assert parsed["groups"].shape == (mock_params.max_nb_spikes,)

    # Check group tensors
    for g in range(mock_params.nGroups):
        group_key = f"group{g}"
        assert group_key in parsed
        # parse_serialized_sequence reshapes to [-1, channels, 32] and filters non-zeros
        # Since we added non-zero data, there should be 2 spikes
        assert len(parsed[group_key].shape) == 3
        assert parsed[group_key].shape[0] == mock_params.max_nb_spikes_per_group
        assert parsed[group_key].shape[1] == mock_params.nChannelsPerGroup[g]
        assert parsed[group_key].shape[2] == 32

        # Check spike counts
        count_key = f"group{g}_spikes_count"
        assert count_key in parsed
        assert parsed[count_key].numpy() == 2
