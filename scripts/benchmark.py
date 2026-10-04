"""Measure the inference cost of AGW-SSDN.

Reports parameters, MACs, peak GPU memory, and latency/FPS for a single
512 x 512 patch (batch size 1), plus the latency of full sliding-window
inference at the requested image sizes. Auxiliary and boundary heads are
disabled, so the measured graph is the deployed inference model.

MACs require the optional ``fvcore`` package (``pip install fvcore``); the
values in the paper were counted with fvcore. ``thop`` is used as a fallback.
"""

from __future__ import annotations

import argparse
import time

import torch

from agw_ssdn.inference import sliding_window_logits, window_positions
from agw_ssdn.models import AGWSSDN
from agw_ssdn.utils import count_parameters, resolve_device

# Native test-image sizes (height, width) of the five datasets in the paper.
DEFAULT_SIZES = {
    "peanut": (720, 960),
    "bonirob": (966, 1296),
    "rice": (1024, 912),
    "carrot": (966, 1296),
    "mustard_greens": (1296, 1296),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--full-image-runs", type=int, default=10)
    return parser.parse_args()


@torch.inference_mode()
def latency_ms(fn, device: torch.device, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    size = args.patch_size

    # Memory already held on the GPU is subtracted from the peak.
    baseline_memory = torch.cuda.memory_allocated() if device.type == "cuda" else 0
    model = AGWSSDN(3, 3, use_auxiliary_heads=False, use_boundary_head=False)
    model = model.to(device).eval()
    patch = torch.randn(1, 3, size, size, device=device)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else device
    print(f"Device: {name}")
    print(f"Params: {count_parameters(model) / 1e6:.2f}M")
    cpu_model = AGWSSDN(3, 3, use_auxiliary_heads=False, use_boundary_head=False).eval()
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(cpu_model, patch.cpu())
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        with torch.no_grad():
            macs = analysis.total()
        print(f"MACs (fvcore): {macs / 1e9:.2f}G")
    except ImportError:
        try:
            from thop import profile

            macs, _ = profile(cpu_model, inputs=(patch.cpu(),), verbose=False)
            print(f"MACs (thop): {macs / 1e9:.2f}G")
        except ImportError:
            print("MACs: skipped (install fvcore or thop)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            model(patch)
        peak = torch.cuda.max_memory_allocated(device) - baseline_memory
        print(f"Peak memory: {peak / 1024**2:.2f} MB")

    patch_ms = latency_ms(lambda: model(patch), device, args.warmup, args.runs)
    print(f"Patch {size}x{size}: {patch_ms:.2f} ms ({1000 / patch_ms:.2f} FPS)")

    for name, (height, width) in DEFAULT_SIZES.items():
        image = torch.randn(1, 3, height, width, device=device)
        windows = len(window_positions(max(height, size), size, args.stride)) * len(
            window_positions(max(width, size), size, args.stride)
        )
        full_ms = latency_ms(
            lambda image=image: sliding_window_logits(
                model, image, patch_size=size, stride=args.stride, num_classes=3
            ),
            device,
            warmup=2,
            runs=args.full_image_runs,
        )
        print(
            f"Full image {name} ({width}x{height}, {windows} windows): "
            f"{full_ms:.2f} ms ({1000 / full_ms:.2f} images/s)"
        )


if __name__ == "__main__":
    main()
