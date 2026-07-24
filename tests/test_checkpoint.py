from __future__ import annotations

import torch

from agw_ssdn.models import AGWSSDN
from agw_ssdn.utils import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    model = AGWSSDN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        best_mean_iou=0.5,
        config={"test": True},
    )

    restored = AGWSSDN()
    metadata = load_checkpoint(
        checkpoint_path, restored, device=torch.device("cpu")
    )

    assert metadata["epoch"] == 1
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            model.state_dict().values(), restored.state_dict().values()
        )
    )
