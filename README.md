# AGW-SSDN

Official research implementation of **AGW-SSDN: A Lightweight Gabor-Guided
Spectral-Spatial Network for Crop-Weed Segmentation**.

AGW-SSDN is a compact three-class semantic segmentation model for background,
crop, and weed pixels. It combines:

- a learnable Adaptive Gabor Wavelet (AGW) stem;
- a lightweight dense encoder with depthwise separable convolutions and SE;
- a Cascaded Spectral-Spatial Aggregator (CSSA) with efficient attention and
  AFNO;
- an Edge-Guided Aggregation (EGA) decoder;
- auxiliary, boundary, and feature-decorrelation supervision during training.

At inference time, the repository preserves the original image resolution. It
uses `512 x 512` sliding windows with stride `384` and averages logits in
overlapping regions before selecting the final class.

## Repository layout

```text
agw-ssdn/
├── configs/
│   └── default.yaml
├── datasets/
│   └── mustard_greens/
│       ├── images/
│       ├── annotations/
│       ├── masks/
│       └── split.csv
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/agw_ssdn/
│   ├── models/
│   │   ├── layers.py
│   │   └── network.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── inference.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Installation

Python 3.10 or newer is required. A CUDA-enabled PyTorch installation is
recommended for training.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If a specific CUDA build of PyTorch is required, install it first using the
command provided by the PyTorch project, then install this package.

## Datasets

### Included: Mustard Greens

The Mustard Greens dataset used in the manuscript is included under
`datasets/mustard_greens`. It contains 100 smartphone images split once with
seed 42 into 80 training images and 20 test images. The default configuration
uses this dataset directly:

```text
datasets/mustard_greens/
├── images/
│   ├── train/
│   │   ├── 001_image.png
│   │   └── ...
│   └── test/
│       ├── 002_image.png
│       └── ...
└── annotations/
    ├── train/
    │   ├── 001_annotation.png
    │   └── ...
    └── test/
        ├── 002_annotation.png
        └── ...
```

The `masks/` directory contains the intermediate binary vegetation masks used
during annotation, while `split.csv` records the reproducible 80/20 split.
Training uses the RGB annotations only.

Masks are RGB images using the following palette:

| Class | Index | RGB |
|---|---:|---|
| Background | 0 | `(0, 0, 0)` |
| Crop | 1 | `(0, 255, 0)` |
| Weed | 2 | `(255, 0, 0)` |

File-name tokens and paths can be changed in
[`configs/default.yaml`](configs/default.yaml).

### External benchmark datasets

The other four datasets evaluated in the paper are not distributed in this
repository. Add their download links below, download them separately, and point
a copied YAML configuration to their image and annotation directories.

- **Peanut:** [dataset link](https://github.com/ptdkhoa/Peanut-dataset)
- **BoniRob:** [dataset link](https://www.ipb.uni-bonn.de/data/sugarbeets2016/)
- **Rice:** [dataset link](https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830)
- **Carrot:** [dataset link](https://github.com/cwfid)

Each external dataset must ultimately follow the same three-class RGB palette.
Its directory names do not need to match the included dataset because all paths
and filename tokens are configurable.

## Configuration

The default experiment reproduces the protocol described in the manuscript:

- training patches: `512 x 512`;
- weed-biased patch probability: `0.8`;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `4` with two-step gradient accumulation;
- Dice-CE for epochs 1-15, then symmetric Lovasz-Softmax;
- original-resolution sliding-window evaluation with stride `384`.

On the first training run, RGB mean and standard deviation are calculated from
the training images. The resolved configuration, including these values, is
saved as `outputs/resolved_config.yaml`. Use that file for standalone evaluation
or prediction so preprocessing remains reproducible.

## Training

Run commands from the repository root:

```bash
python scripts/train.py --config configs/default.yaml --device cuda
```

The best validation checkpoint is written to
`outputs/agw_ssdn_best.pt`. It contains the model, optimizer, scheduler, epoch,
best mIoU, and the complete experiment configuration.

## Evaluation

Evaluation keeps every test image at its native resolution and uses one image
per DataLoader batch:

```bash
python scripts/evaluate.py \
  --config outputs/resolved_config.yaml \
  --checkpoint outputs/agw_ssdn_best.pt \
  --device cuda \
  --output outputs/test_metrics.json
```

The command reports pixel accuracy, per-class accuracy, per-class IoU, and
mIoU.

## Prediction

```bash
python scripts/predict.py \
  --config outputs/resolved_config.yaml \
  --checkpoint outputs/agw_ssdn_best.pt \
  --image path/to/image.png \
  --output outputs/prediction.png \
  --device cuda
```

The saved prediction is a colorized RGB mask with exactly the same height and
width as the input image.

## Python API

```python
import torch
from agw_ssdn import AGWSSDN
from agw_ssdn.inference import sliding_window_predict

model = AGWSSDN(num_classes=3).cuda().eval()
image = torch.randn(1, 3, 720, 960, device="cuda")

prediction = sliding_window_predict(
    model,
    image,
    patch_size=512,
    stride=384,
    num_classes=3,
)
assert prediction.shape == (720, 960)
```

## Tests

```bash
pytest
```

The test suite checks model outputs, original-resolution dataset behavior, edge
coverage, padding, and overlapping-window averaging.

## Citation

If this repository supports your research, please cite:

```bibtex
@article{bui_agw_ssdn,
  title   = {AGW-SSDN: A Lightweight Gabor-Guided Spectral-Spatial Network
             for Crop-Weed Segmentation},
  author  = {Bui, Khanh Duy and Nguyen, Huu Du},
  journal = {Computers and Electronics in Agriculture}
}
```
