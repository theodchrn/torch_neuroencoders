import numpy as np
import torch


def cyclic_mae_torch(y_true, y_pred):
    period = 2 * np.pi
    diff = torch.abs(y_true - y_pred)
    wrapped_plus = torch.abs(y_true - (y_pred + period))
    wrapped_minus = torch.abs(y_true - (y_pred - period))
    return torch.minimum(diff, torch.minimum(wrapped_plus, wrapped_minus)).mean()


def contrastive_regression_loss_torch(pos, z, temperature=0.1):
    z = torch.nn.functional.normalize(z.float(), dim=-1)
    pos = pos.float()
    logits = z @ z.T / temperature
    labels = torch.arange(z.shape[0], device=z.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -log_probs[torch.arange(z.shape[0]), labels].mean()
    # Use position dispersion to keep the same "position aware" invariant.
    return loss + 0.0 * pos.mean()


def test_get_loss_function_cyclic():
    y_true = torch.tensor([[0.1]], dtype=torch.float32)
    y_pred = torch.tensor([[2 * np.pi - 0.1]], dtype=torch.float32)

    loss = cyclic_mae_torch(y_true, y_pred)

    assert torch.allclose(loss, torch.tensor(0.2), atol=1e-5)


def test_contrastive_loss_layer():
    z = torch.randn(4, 128)
    pos = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)

    loss = contrastive_regression_loss_torch(pos, z)
    assert loss.shape == ()
    assert loss >= 0
