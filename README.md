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
│   ├── default.yaml          # Mustard Greens (same as mustard_greens.yaml)
│   ├── peanut.yaml
│   ├── bonirob.yaml
│   ├── rice.yaml
│   ├── carrot.yaml
│   └── mustard_greens.yaml
├── datasets/
│   └── mustard_greens/
│       ├── images/
│       ├── annotations/
│       ├── masks/
│       └── split.csv
├── splits/                   # fixed train/val/test lists and statistics
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── make_split.py
│   └── benchmark.py
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
repository. Download them from the links below and place each one under
`datasets/<name>/images/{train,test}` and `datasets/<name>/annotations/{train,test}`
(`<name>` = `peanut`, `bonirob`, `rice`, `carrot`), or edit the paths in
`configs/<name>.yaml`. The train/test partition follows the one used in the
paper; the image lists of every split are given in `splits/<name>.json`.

- **Peanut:** [dataset link](https://github.com/ptdkhoa/Peanut-dataset)
- **BoniRob:** [dataset link](https://www.ipb.uni-bonn.de/data/sugarbeets2016/)
- **Rice:** [dataset link](https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830)
- **Carrot:** [dataset link](https://github.com/cwfid)

Each external dataset must ultimately follow the same three-class RGB palette.
Its directory names do not need to match the included dataset because all paths
and filename tokens are configurable.

## Experimental protocol

The per-dataset configurations reproduce the protocol described in the
manuscript:

- **Splits:** 20% of the images in each original training folder are held out
  for validation (random seed `42`); the original test folder is unchanged.
  The exact image lists are stored in `splits/<dataset>.json` and read through
  `data.split_file`.
- **Checkpoint selection:** the checkpoint with the highest validation mIoU is
  kept; the test set is evaluated once with that checkpoint after training.
- **Preprocessing:** RGB normalization statistics and median-frequency class
  weights are computed on the training subset only (validation images
  excluded). The resulting values are also listed in each split file.
- **Patch sampling:** random `512 x 512` training patches; with probability
  `0.8` the patch is centred near a randomly chosen weed pixel.
- **Optimization:** AdamW, learning rate `1e-3`, weight decay `1e-4`, cosine
  annealing over 50 epochs, batch size `4` with two-step gradient
  accumulation (effective batch size 8), seed `42`.
- **Loss:** Dice-CE for epochs 1-15, then symmetric Lovasz-Softmax, plus the
  auxiliary, boundary, and feature-decorrelation terms (weights in
  `configs/*.yaml`).
- **Inference:** validation and test images are processed at native
  resolution with `512 x 512` sliding windows, stride `384`, and logit
  averaging over overlapping regions.

A split file can be regenerated from a local copy of a dataset with:

```bash
python scripts/make_split.py --config configs/peanut.yaml --output splits/peanut.json
```

During training, the resolved configuration, including the normalization
statistics, is saved as `outputs/<dataset>/resolved_config.yaml`. Use that file
for standalone evaluation or prediction so preprocessing remains reproducible.

## Training

Run commands from the repository root:

```bash
python scripts/train.py --config configs/peanut.yaml --device cuda
```

Replace `peanut` with `bonirob`, `rice`, `carrot`, or `mustard_greens`. The
best validation checkpoint is written to
`outputs/<dataset>/agw_ssdn_best.pt`. It contains the model, optimizer, scheduler, epoch,
best mIoU, and the complete experiment configuration.

## Evaluation

Evaluation keeps every test image at its native resolution and uses one image
per DataLoader batch:

```bash
python scripts/evaluate.py \
  --config outputs/peanut/resolved_config.yaml \
  --checkpoint outputs/peanut/agw_ssdn_best.pt \
  --device cuda \
  --output outputs/peanut/test_metrics.json
```

The command reports pixel accuracy, per-class accuracy, per-class IoU, and
mIoU.

## Inference cost

```bash
pip install fvcore   # optional, needed for MACs
python scripts/benchmark.py --device cuda
```

The script reports parameters, MACs, peak GPU memory, and latency/FPS for one
`512 x 512` patch with batch size 1, and the latency of full sliding-window
inference at the native image size of each dataset. Auxiliary and boundary
heads are removed, so the measured graph is the deployed model.

## Baselines

The baselines were trained and evaluated with the same splits, patch
sampling, optimizer, learning-rate schedule, effective batch size of 8
(DMSCN: batch 2 with four-step accumulation because of GPU memory), and
sliding-window evaluation as AGW-SSDN, using median-frequency weighted
Dice-CE for all 50 epochs.

| Model | Source | Backbone / initialization |
|---|---|---|
| FCN | [torchvision](https://github.com/pytorch/vision) `fcn_resnet50` | ResNet-50, COCO |
| U-Net | [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | ResNet-34, ImageNet |
| PSPNet | [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | ResNet-50, ImageNet |
| DeepLabV3+ | [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | ResNet-50, ImageNet |
| SegFormer-B2 | [Hugging Face Transformers](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) | MiT-B2, ADE20K |
| Swin-UNet | [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) U-Net, `tu-swin_tiny_patch4_window7_224` encoder | Swin-T, ImageNet |
| SegNet | Re-implemented following [Badrinarayanan et al. (2017)](https://doi.org/10.1109/TPAMI.2016.2644615) | random |
| CED-Net | Re-implemented following Khan et al. (2020) | random |
| DMSCN | Re-implemented following the original DMSCN paper | random |

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
