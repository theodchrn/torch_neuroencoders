import pytest
import torch

from torch_neuroencoders.fullEncoder.nnUtils import (
    GroupAttentionFusion,
    SpikeNet1DTorch,
)


def test_spike_net_1d():
    nChannels = 4
    nFeatures = 64
    batch_size = 8
    seq_len = 10

    layer = SpikeNet1DTorch(nChannels=nChannels, nFeatures=nFeatures)

    # Input shape: (Batch * SeqLen, Channels, Time)
    # With TimeDistributed, the layer sees (Batch, Channels, Time) if wrapped,
    # but SpikeNet1D expects (Batch, Channels, Time) and handles it.

    x = torch.randn(batch_size * seq_len, nChannels, 32)
    output = layer(x)

    assert output.shape == (batch_size * seq_len, nFeatures)


def test_group_attention_fusion():
    nGroups = 4
    nFeatures = 64
    batch_size = 8
    seq_len = 10

    layer = GroupAttentionFusion(n_groups=nGroups, embed_dim=nFeatures)

    # Input: list of tensors, each (Batch, SeqLen, Features)
    inputs = [torch.randn(batch_size, seq_len, nFeatures) for _ in range(nGroups)]

    # Mask: (Batch, SeqLen, nGroups)
    mask = torch.rand(batch_size, seq_len, nGroups) > 0.5

    output = layer(inputs, mask=mask)

    # Output should be (Batch, SeqLen, Groups * Features)
    assert output.shape == (batch_size, seq_len, nGroups * nFeatures)


def test_spike_net_1d_with_time_distributed():
    nChannels = 4
    nFeatures = 64
    batch_size = 8
    seq_len = 10

    layer = SpikeNet1DTorch(nChannels=nChannels, nFeatures=nFeatures)

    # Input: (Batch, SeqLen, Channels, Time)
    x = torch.randn(batch_size, seq_len, nChannels, 32)
    output = layer(x.reshape(batch_size * seq_len, nChannels, 32)).reshape(
        batch_size, seq_len, nFeatures
    )

    assert output.shape == (batch_size, seq_len, nFeatures)


if __name__ == "__main__":
    pytest.main([__file__])
