from __future__ import annotations

import pytest
import torch
from torch import nn

from agw_ssdn.inference import (
    sliding_window_logits,
    sliding_window_predict,
    window_positions,
)


class PointwiseDummyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[:, :1]
        return torch.cat((channel, channel + 1, channel - 1), dim=1)


@pytest.mark.parametrize(
    ("height", "width"),
    [(720, 960), (300, 400), (512, 512), (513, 777)],
)
def test_sliding_window_preserves_resolution(height: int, width: int) -> None:
    model = PointwiseDummyModel().eval()
    image = torch.randn(1, 3, height, width)
    logits = sliding_window_logits(model, image)
    prediction = sliding_window_predict(model, image)

    assert logits.shape == (1, 3, height, width)
    assert prediction.shape == (height, width)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits, model(image), atol=1e-6)


def test_window_positions_cover_far_edge() -> None:
    assert window_positions(960, 512, 384) == [0, 384, 448]
    assert window_positions(512, 512, 384) == [0]


def test_window_positions_reject_gaps() -> None:
    with pytest.raises(ValueError, match="leave gaps"):
        window_positions(1024, 512, 600)
