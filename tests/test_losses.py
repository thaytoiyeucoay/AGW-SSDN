from __future__ import annotations

import torch

from agw_ssdn.losses import SymmetricLovaszLoss


def test_symmetric_lovasz_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))
    loss = SymmetricLovaszLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
