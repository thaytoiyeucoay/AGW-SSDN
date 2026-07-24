"""Reusable neural-network layers used by AGW-SSDN."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int | None = None,
    ) -> None:
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        scale = self.avg_pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale


class LightweightDenseLayer(nn.Module):
    """BN-ReLU bottleneck, depthwise convolution, pointwise projection and SE."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        bottleneck_width: int = 128,
        dilation: int = 1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_width, kernel_size=1, bias=False),
        )
        self.depthwise = nn.Sequential(
            nn.BatchNorm2d(bottleneck_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_width,
                bottleneck_width,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=bottleneck_width,
                bias=False,
            ),
        )
        self.pointwise = nn.Sequential(
            nn.BatchNorm2d(bottleneck_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_width, growth_rate, kernel_size=1, bias=False),
        )
        self.se = SEBlock(growth_rate) if use_se else nn.Identity()
        self.out_channels = in_channels + growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pointwise(self.depthwise(self.bottleneck(x)))
        features = self.se(features)
        return torch.cat((x, features), dim=1)


class LightweightDenseBlock(nn.Module):
    """Stack of densely connected lightweight layers."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int = 32,
        bottleneck_width: int = 128,
        dilation: int = 1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(num_layers):
            layer = LightweightDenseLayer(
                channels,
                growth_rate=growth_rate,
                bottleneck_width=bottleneck_width,
                dilation=dilation,
                use_se=use_se,
            )
            layers.append(layer)
            channels = layer.out_channels
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """Channel compression with optional 2x spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.projection = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_pool = self.projection(x)
        return self.pool(pre_pool), pre_pool


class CBAMBlock(nn.Module):
    """Convolutional block attention module."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = torch.sigmoid(
            self.channel_mlp(self.avg_pool(x)) + self.channel_mlp(self.max_pool(x))
        )
        x = x * channel_attention
        spatial_features = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1
        )
        return x * torch.sigmoid(self.spatial(spatial_features))


class GaborConv2d(nn.Module):
    """Differentiable 2D Gabor convolution with one kernel per channel pair."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        shape = (out_channels, in_channels)
        self.sigma = nn.Parameter(torch.rand(shape) + 3.0)
        self.theta = nn.Parameter(torch.rand(shape) * math.pi)
        self.wavelength = nn.Parameter(torch.rand(shape) + 3.0)
        self.gamma = nn.Parameter(torch.rand(shape) * 0.5 + 0.5)
        self.phase = nn.Parameter(torch.rand(shape) * math.pi)
        self.amplitude = nn.Parameter(torch.ones(shape))

        coordinates = torch.arange(kernel_size) - (kernel_size - 1) / 2
        y_grid, x_grid = torch.meshgrid(coordinates, coordinates, indexing="ij")
        # meshgrid may return zero-stride views that cannot be checkpoint-loaded
        # in place. Clone them into independent, contiguous buffers.
        self.register_buffer("x_grid", x_grid.clone())
        self.register_buffer("y_grid", y_grid.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma[..., None, None]
        theta = self.theta[..., None, None]
        wavelength = self.wavelength[..., None, None]
        gamma = self.gamma[..., None, None]
        phase = self.phase[..., None, None]

        x_prime = self.x_grid * torch.cos(theta) + self.y_grid * torch.sin(theta)
        y_prime = -self.x_grid * torch.sin(theta) + self.y_grid * torch.cos(theta)
        envelope = torch.exp(
            -(x_prime.square() + gamma.square() * y_prime.square())
            / (2 * sigma.square())
        )
        carrier = torch.cos(2 * math.pi * x_prime / wavelength + phase)
        kernels = envelope * carrier * self.amplitude[..., None, None]
        return F.conv2d(x, kernels, stride=self.stride, padding=self.padding)


class AFNO2d(nn.Module):
    """Adaptive Fourier neural operator for global frequency-domain mixing."""

    def __init__(
        self, channels: int, num_blocks: int = 8, sparsity_threshold: float = 0.01
    ) -> None:
        super().__init__()
        if channels % num_blocks != 0:
            raise ValueError("channels must be divisible by num_blocks")
        self.channels = channels
        self.num_blocks = num_blocks
        self.block_size = channels // num_blocks
        self.sparsity_threshold = sparsity_threshold
        scale = 0.02

        weight_shape = (num_blocks, self.block_size, self.block_size)
        bias_shape = (num_blocks, self.block_size, 1, 1)
        self.w1_real = nn.Parameter(scale * torch.randn(weight_shape))
        self.w1_imag = nn.Parameter(scale * torch.randn(weight_shape))
        self.w2_real = nn.Parameter(scale * torch.randn(weight_shape))
        self.w2_imag = nn.Parameter(scale * torch.randn(weight_shape))
        self.b1 = nn.Parameter(scale * torch.randn(bias_shape))
        self.b2 = nn.Parameter(scale * torch.randn(bias_shape))

    @staticmethod
    def _multiply(
        inputs: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor
    ) -> torch.Tensor:
        real = torch.einsum("bnihw,nio->bnohw", inputs.real, weights_real)
        real -= torch.einsum("bnihw,nio->bnohw", inputs.imag, weights_imag)
        imag = torch.einsum("bnihw,nio->bnohw", inputs.imag, weights_real)
        imag += torch.einsum("bnihw,nio->bnohw", inputs.real, weights_imag)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        frequency = torch.fft.rfft2(x, norm="ortho")
        frequency = frequency.reshape(
            batch, self.num_blocks, self.block_size, height, frequency.shape[-1]
        )
        frequency = self._multiply(frequency, self.w1_real, self.w1_imag) + self.b1
        frequency = F.relu(frequency.real) + 1j * F.relu(frequency.imag)
        frequency = self._multiply(frequency, self.w2_real, self.w2_imag) + self.b2

        if self.sparsity_threshold > 0:
            magnitude = F.relu(torch.abs(frequency) - self.sparsity_threshold)
            frequency = magnitude * torch.exp(1j * torch.angle(frequency))

        frequency = frequency.reshape(batch, channels, height, frequency.shape[-1])
        output = torch.fft.irfft2(frequency, s=(height, width), norm="ortho")
        return output + residual


class EfficientAttention(nn.Module):
    """Multi-head attention with optional spatially reduced keys and values."""

    def __init__(
        self, channels: int, num_heads: int = 4, reduction_ratio: int = 1
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        if reduction_ratio > 1:
            self.spatial_reduction: nn.Module | None = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=reduction_ratio,
                    stride=reduction_ratio,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            )
            self.norm: nn.Module | None = nn.LayerNorm(channels)
        else:
            self.spatial_reduction = None
            self.norm = None

        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(channels, channels * 2)
        self.projection = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        query = self.query(tokens)
        query = query.reshape(
            batch, -1, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        if self.spatial_reduction is not None:
            reduced = self.spatial_reduction(x).flatten(2).transpose(1, 2)
            key_value_tokens = self.norm(reduced)
        else:
            key_value_tokens = tokens

        key, value = self.key_value(key_value_tokens).chunk(2, dim=-1)
        key = key.reshape(batch, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        value = value.reshape(batch, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(
            batch, height * width, channels
        )
        output = self.projection(output)
        return output.transpose(1, 2).reshape(batch, channels, height, width)


class AttentionBlock(nn.Module):
    """Channel projection followed by efficient attention."""

    def __init__(
        self, in_channels: int, out_channels: int, reduction_ratio: int
    ) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.attention = EfficientAttention(
            out_channels, num_heads=4, reduction_ratio=reduction_ratio
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(self.reduce(x)))


class CSSA(nn.Module):
    """Cascaded Spectral-Spatial Aggregator."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.local_branch = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.medium_branch = AttentionBlock(
            in_channels + out_channels, out_channels, reduction_ratio=2
        )
        self.large_branch = AttentionBlock(
            in_channels + 2 * out_channels, out_channels, reduction_ratio=4
        )
        self.global_projection = ConvBNAct(
            in_channels + 3 * out_channels, out_channels, kernel_size=1
        )
        self.global_branch = AFNO2d(out_channels)
        self.fusion = nn.Sequential(
            CBAMBlock(4 * out_channels),
            ConvBNAct(4 * out_channels, out_channels, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detail = self.local_branch(x)
        medium = self.medium_branch(torch.cat((x, detail), dim=1))
        large = self.large_branch(torch.cat((x, detail, medium), dim=1))
        trend = self.global_projection(torch.cat((x, detail, medium, large), dim=1))
        trend = self.global_branch(trend)
        fused = self.fusion(torch.cat((detail, medium, large, trend), dim=1))
        return fused, trend, detail


class EdgeGatedAttention(nn.Module):
    """Gate decoder features using a spatial mask derived from a skip feature."""

    def __init__(self, channels: int, skip_channels: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(skip_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(
                skip, size=x.shape[-2:], mode="bilinear", align_corners=True
            )
        return self.refine(self.feature(x) * self.gate(skip)) + x


class UpConvBlock(nn.Sequential):
    """Bilinear upsampling followed by depthwise-separable convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=True
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
