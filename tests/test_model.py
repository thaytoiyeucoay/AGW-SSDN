from __future__ import annotations

import torch

from agw_ssdn.models import AGWSSDN


def test_model_evaluation_output_shape() -> None:
    model = AGWSSDN().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 64, 64)


def test_model_training_outputs() -> None:
    model = AGWSSDN().train()
    output = model(torch.randn(2, 3, 64, 64))
    assert isinstance(output, dict)
    assert set(output) == {
        "main",
        "trend",
        "detail",
        "aux1",
        "aux2",
        "aux3",
        "boundary",
    }
    assert output["main"].shape == (2, 3, 64, 64)
    assert output["boundary"].shape == (2, 1, 64, 64)
