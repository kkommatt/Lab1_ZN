"""
Verify a trained Faster R-CNN checkpoint and save validation visualizations.

Place this file next to main.py and run it from the project's virtual environment.

Example:
    python verify_checkpoint.py --data_dir data --checkpoint "lightning_logs/version_1/checkpoints/frcnn-epoch=01-val_map_50=0.7316.ckpt"

The script:
  1. Reuses ArTaxOrDataModule and FasterRCNNLit from main.py.
  2. Loads the selected Lightning checkpoint.
  3. Evaluates the validation split using TorchMetrics mAP.
  4. Saves annotated validation images and a metrics.json file.

It does not train or modify the checkpoint.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from main import ArTaxOrDataModule, FasterRCNNLit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Faster R-CNN checkpoint and visualize detections."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("verification_results"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="0 evaluates the complete validation set; otherwise evaluates only this many batches.",
    )
    parser.add_argument("--num_visualizations", type=int, default=12)
    return parser.parse_args()


def accepted_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Pass only keyword arguments supported by the project's class constructor."""
    signature = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer the classifier output size from the Faster R-CNN checkpoint."""
    candidate_keys = (
        "model.roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.weight",
    )
    for key in candidate_keys:
        if key in state_dict:
            return int(state_dict[key].shape[0])

    for key, tensor in state_dict.items():
        if key.endswith("roi_heads.box_predictor.cls_score.weight"):
            return int(tensor.shape[0])

    raise RuntimeError(
        "Could not infer the number of classes from the checkpoint. "
        "Check that this is a checkpoint created by FasterRCNNLit."
    )


def get_class_names(data_module: Any) -> dict[int, str]:
    """Try common class-name attributes; fall back to numeric labels."""
    for attr in ("class_names", "classes", "order_names"):
        value = getattr(data_module, attr, None)
        if isinstance(value, (list, tuple)):
            return {i + 1: str(name) for i, name in enumerate(value)}

    mapping = getattr(data_module, "class_to_idx", None)
    if isinstance(mapping, dict):
        # Dataset labels are commonly one-based because 0 is background.
        return {int(index): str(name) for name, index in mapping.items()}

    return {}


def to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def draw_detections(
    image_tensor: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    class_names: dict[int, str],
    output_path: Path,
    score_threshold: float,
) -> None:
    image = to_pil(image_tensor)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        class_id = int(label)
        class_name = class_names.get(class_id, f"class_{class_id}")
        caption = f"{class_name}: {float(score):.2f}"

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_box = draw.textbbox((x1, max(0, y1 - 19)), caption, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((x1, max(0, y1 - 19)), caption, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score_threshold must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the checkpoint on CPU first so that GPU memory is not occupied
    # while constructing the data module.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    num_classes = infer_num_classes(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dm_kwargs = {
        "data_dir": str(args.data_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_split": args.val_split,
        "seed": args.seed,
    }
    data_module = ArTaxOrDataModule(
        **accepted_kwargs(ArTaxOrDataModule, dm_kwargs)
    )
    data_module.setup("validate")

    model_kwargs = {"num_classes": num_classes, "lr": 1e-4}
    lightning_model = FasterRCNNLit(
        **accepted_kwargs(FasterRCNNLit, model_kwargs)
    )

    incompatible = lightning_model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Warning: missing checkpoint keys: {incompatible.missing_keys[:10]}")
    if incompatible.unexpected_keys:
        print(
            "Warning: unexpected checkpoint keys: "
            f"{incompatible.unexpected_keys[:10]}"
        )

    model = lightning_model.model.to(device)
    model.eval()

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    class_names = get_class_names(data_module)
    val_loader = data_module.val_dataloader()

    saved_visualizations = 0
    processed_batches = 0
    processed_images = 0

    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(val_loader):
            if args.max_batches and batch_index >= args.max_batches:
                break

            images_on_device = [image.to(device) for image in images]
            predictions = model(images_on_device)

            predictions_cpu = [
                {key: value.detach().cpu() for key, value in prediction.items()}
                for prediction in predictions
            ]
            targets_cpu = [
                {key: value.detach().cpu() for key, value in target.items()}
                for target in targets
            ]

            metric.update(predictions_cpu, targets_cpu)

            for image, prediction in zip(images, predictions_cpu):
                if saved_visualizations >= args.num_visualizations:
                    break
                output_path = (
                    args.output_dir
                    / "visualizations"
                    / f"validation_{saved_visualizations + 1:03d}.jpg"
                )
                draw_detections(
                    image,
                    prediction,
                    class_names,
                    output_path,
                    args.score_threshold,
                )
                saved_visualizations += 1

            processed_images += len(images)
            processed_batches += 1

            if processed_batches % 50 == 0:
                print(
                    f"Processed {processed_images} images "
                    f"({processed_batches} batches)..."
                )

    if processed_images == 0:
        raise RuntimeError(
            "No validation images were processed. Check the dataset split "
            "and DataModule configuration."
        )

    computed = metric.compute()
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "num_classes_including_background": num_classes,
        "processed_images": processed_images,
        "processed_batches": processed_batches,
        "score_threshold_for_visualizations": args.score_threshold,
        "max_batches": args.max_batches,
        "metrics": {
            key: (
                value.tolist()
                if isinstance(value, torch.Tensor) and value.numel() > 1
                else float(value.item())
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in computed.items()
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nValidation results")
    print("------------------")
    for name in ("map", "map_50", "map_75", "mar_100"):
        value = computed.get(name)
        if isinstance(value, torch.Tensor):
            print(f"{name}: {float(value):.6f}")

    print(f"\nImages evaluated: {processed_images}")
    print(f"Visualizations saved: {saved_visualizations}")
    print(f"Metrics saved to: {metrics_path.resolve()}")
    print(
        "Visualization folder: "
        f"{(args.output_dir / 'visualizations').resolve()}"
    )


if __name__ == "__main__":
    main()
