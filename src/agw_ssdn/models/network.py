"""AGW-SSDN semantic segmentation network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers import (
    CSSA,
    ConvBNAct,
    EdgeGatedAttention,
    GaborConv2d,
    LightweightDenseBlock,
    TransitionLayer,
    UpConvBlock,
)


class Encoder(nn.Module):
    """Adaptive Gabor stem, lightweight dense encoder and CSSA bottleneck."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.agw_stem = nn.Sequential(
            GaborConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = LightweightDenseBlock(64, num_layers=6)
        self.transition1 = TransitionLayer(
            self.dense1.out_channels, 128, pool=True
        )
        self.low_level_projection = nn.Conv2d(128, 48, kernel_size=1, bias=False)

        self.dense2 = LightweightDenseBlock(128, num_layers=12)
        self.transition2 = TransitionLayer(
            self.dense2.out_channels, 256, pool=True
        )
        self.dense3 = LightweightDenseBlock(256, num_layers=12)
        self.transition3 = TransitionLayer(
            self.dense3.out_channels, 512, pool=False
        )
        self.dense4 = LightweightDenseBlock(512, num_layers=8, dilation=2)
        self.transition4 = TransitionLayer(
            self.dense4.out_channels, 640, pool=False
        )
        self.cssa = CSSA(640, out_channels=256)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        agw_skip = self.agw_stem(x)
        x = self.stem_pool(agw_skip)

        x = self.dense1(x)
        x, dense1_skip = self.transition1(x)
        dense1_skip = self.low_level_projection(dense1_skip)

        x = self.dense2(x)
        x, _ = self.transition2(x)
        x = self.dense3(x)
        x, _ = self.transition3(x)
        x = self.dense4(x)
        x, _ = self.transition4(x)
        features, trend, detail = self.cssa(x)
        return features, dense1_skip, agw_skip, trend, detail


class AuxiliaryHead(nn.Module):
    """Training-only semantic segmentation head."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        logits = self.head(x)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=True
        )


class BoundaryHead(nn.Module):
    """Training-only binary boundary prediction head."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_channels, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        logits = self.head(x)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=True
        )


class Decoder(nn.Module):
    """Two-stage Edge-Guided Aggregation decoder."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.upsample1 = UpConvBlock(256, 256, scale_factor=4)
        self.gate1 = EdgeGatedAttention(256, skip_channels=48)
        self.refine1 = nn.Sequential(
            ConvBNAct(256 + 48, 128, kernel_size=3),
            ConvBNAct(128, 128, kernel_size=3),
        )

        self.upsample2 = UpConvBlock(128, 64, scale_factor=2)
        self.gate2 = EdgeGatedAttention(64, skip_channels=64)
        self.refine2 = ConvBNAct(64 + 64, 64, kernel_size=3)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _resize_skip(skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] == target.shape[-2:]:
            return skip
        return F.interpolate(
            skip, size=target.shape[-2:], mode="bilinear", align_corners=True
        )

    def forward(
        self,
        features: torch.Tensor,
        dense1_skip: torch.Tensor,
        agw_skip: torch.Tensor,
        *,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.upsample1(features)
        dense1_skip = self._resize_skip(dense1_skip, x)
        x = self.refine1(torch.cat((self.gate1(x, dense1_skip), dense1_skip), dim=1))
        features_h4 = x

        x = self.upsample2(x)
        agw_skip = self._resize_skip(agw_skip, x)
        x = self.refine2(torch.cat((self.gate2(x, agw_skip), agw_skip), dim=1))
        features_h2 = x

        logits = F.interpolate(
            self.classifier(x), scale_factor=2, mode="bilinear", align_corners=True
        )
        if return_intermediate:
            return logits, features_h4, features_h2
        return logits


class AGWSSDN(nn.Module):
    """Lightweight Gabor-guided spectral-spatial segmentation network."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        use_auxiliary_heads: bool = True,
        use_boundary_head: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_auxiliary_heads = use_auxiliary_heads
        self.use_boundary_head = use_boundary_head
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(num_classes)

        if use_auxiliary_heads:
            self.auxiliary_heads = nn.ModuleList(
                (
                    AuxiliaryHead(256, num_classes),
                    AuxiliaryHead(128, num_classes),
                    AuxiliaryHead(64, num_classes),
                )
            )
        if use_boundary_head:
            self.boundary_head = BoundaryHead(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        output_size = x.shape[-2:]
        features, dense1_skip, agw_skip, trend, detail = self.encoder(x)

        if not self.training:
            return self.decoder(features, dense1_skip, agw_skip)

        main, features_h4, features_h2 = self.decoder(
            features, dense1_skip, agw_skip, return_intermediate=True
        )
        outputs = {"main": main, "trend": trend, "detail": detail}
        if self.use_auxiliary_heads:
            outputs.update(
                {
                    "aux1": self.auxiliary_heads[0](features, output_size),
                    "aux2": self.auxiliary_heads[1](features_h4, output_size),
                    "aux3": self.auxiliary_heads[2](features_h2, output_size),
                }
            )
        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(features_h4, output_size)
        return outputs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class predictions and probabilities for an input tensor."""
        self.eval()
        logits = self(x)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Expected tensor output in evaluation mode")
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.argmax(dim=1), probabilities


# Backward-compatible name used by the original research script.
MSTMModel = AGWSSDN
