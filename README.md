# Arthropod Object Detection — Faster R-CNN

Object detection and classification of arthropods on the **ArTaxOr** dataset using Faster R-CNN with a MobileNetV3-Large FPN backbone (`torchvision`).

Includes data preparation, training via PyTorch Lightning, metric evaluation, and standalone checkpoint verification with bounding box visualizations.

## Project Structure

```text
Lab1_ZN/
├── main.py
├── verify_checkpoint.py
├── data/
│   └── ArTaxOr/
│       ├── <class_1>/
│       │   ├── annotations/
│       │   │   ├── <annotation_1>.json
│       │   │   └── ...
│       │   ├── <image_1>.jpg
│       │   └── ...
│       └── ...
├── lightning_logs/
│   └── version_*/
│       └── checkpoints/
└── verification_results/
    ├── metrics.json
    └── visualizations/
```

Image paths are resolved from `asset.name` inside the VoTT JSON files, matching the directory layout expected by `ArTaxOrDataModule` in `main.py`.

## Setup

**Requirements:**
- Python 3.10+
- Compatible PyTorch & Torchvision (NVIDIA GPU + CUDA recommended)
- `lightning`, `torchmetrics`, `pillow`, `pycocotools`

**Environment & Installation (Windows PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch with CUDA 12.6
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

# Remaining dependencies
python -m pip install lightning torchmetrics pillow pycocotools
```

*(For other CUDA or Python versions, check the [PyTorch installation selector](https://pytorch.org/get-started/locally/).)*

**Verify CUDA:**

```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

If your PyTorch version ends with `+cpu`, reinstall the CUDA build.

## Dataset Preparation

1. Download the [Arthropod Taxonomy Orders Object Detection Dataset](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset) from Kaggle.
2. Extract it into the `data/` directory.
3. Ensure each JSON annotation contains `asset.name` and `regions`.

During preprocessing, VoTT coordinates (`x, y, width, height`) are converted to `[x_min, y_min, x_max, y_max]`. Class labels start at `1` (`0` is reserved for background).

## Training

Run training (example for 2 epochs):

```powershell
python .\main.py --data_dir data --max_epochs 2 --batch_size 2 --num_workers 2
```

**CLI Arguments:**

| Argument | Description | Example / Default |
|---|---|---:|
| `--data_dir` | Dataset directory | `data` |
| `--max_epochs` | Training epochs | `2` |
| `--batch_size` | Batch size | `2` |
| `--num_workers` | DataLoader workers | `2` |
| `--val_split` | Validation split ratio | `0.1` |
| `--seed` | Random seed for data split | `42` |
| `--lr` | Learning rate | `0.0001` |

Lightning logs metrics and saves the best checkpoint based on `val_map_50` inside `lightning_logs/`:

```text
lightning_logs/version_1/checkpoints/frcnn-epoch=01-val_map_50=0.7316.ckpt
```

## Checkpoint Verification

Run standalone evaluation and generate prediction visualizations without retraining:

```powershell
python .\verify_checkpoint.py `
  --data_dir data `
  --checkpoint ".\lightning_logs\version_1\checkpoints\frcnn-epoch=01-val_map_50=0.7316.ckpt"
```

**Verification Arguments:**

| Argument | Description | Default |
|---|---|---:|
| `--output_dir` | Output folder for metrics and images | `verification_results` |
| `--batch_size` | Batch size | `2` |
| `--num_workers` | DataLoader workers | `2` |
| `--val_split` | Validation split ratio (must match training) | `0.1` |
| `--seed` | Split seed (must match training) | `42` |
| `--score_threshold` | Confidence threshold for drawing boxes | `0.5` |
| `--max_batches` | Max batches to evaluate (`0` = full set) | `0` |
| `--num_visualizations` | Number of sample images to save | `12` |

Outputs are saved to `verification_results/metrics.json` and `verification_results/visualizations/`.

## Experiment Results

Trained for **2 epochs** and evaluated across the full validation set (**1,537 images**) on a CUDA GPU.

| Metric | Before Training | After Training |
|---|---:|---:|
| **mAP@0.5:0.95** (`map`) | 0.0014 | **0.5065** |
| **mAP@0.5** (`map_50`) | 0.0062 | **0.7281** |
| **mAP@0.75** (`map_75`) | — | **0.5839** |
| **mAR@100** (`mar_100`) | — | **0.6437** |
| mAP medium (`map_medium`) | — | 0.0040 |
| mAP large (`map_large`) | — | 0.5117 |
| mAR@1 (`mar_1`) | — | 0.5718 |
| mAR@10 (`mar_10`) | — | 0.6383 |

**Notes:**
- Lightning logged `val_map_50 = 0.7316` at the end of epoch 1, while standalone verification scored `0.7281` across the full validation set.
- `map_small = -1.0` and `mar_small = -1.0` simply indicate that no ground-truth objects in the validation split fell into the COCO "small" area threshold.

## Troubleshooting

- **CUDA out of memory:** Lower the batch size (`--batch_size 1`).
- **Missing images:** Verify that `asset.name` in the VoTT JSON matches the actual image filename in the class directory.
- **DataLoader warnings:** Warnings about `persistent_workers` or `pin_memory` are performance hints, not errors. You can enable them in `DataLoader` if needed.
- **Reproducibility:** Keep `seed=42`, `val_split=0.1`, and package versions identical when comparing runs.

## References

- [ArTaxOr Dataset (Kaggle)](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset)
- [Torchvision Faster R-CNN](https://pytorch.org/vision/stable/models/faster_rcnn.html)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [TorchMetrics MeanAveragePrecision](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)