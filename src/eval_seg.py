"""Evaluate DeepLabV3-MobileNetV3 checkpoints on Oxford-IIIT Pet segmentation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DATA_DIR,
    RESULT_DIR,
    SEED,
    SEGMENTATION_BATCH_SIZE,
    SEGMENTATION_IMAGE_SIZE,
    SEG_NUM_CLASSES,
    ensure_output_dirs,
    set_seed,
)
from data_loaders import Split, build_segmentation_dataloader
from models_seg import (
    build_segmentation_model,
    get_segmentation_checkpoint_path,
    normalize_segmentation_model_name,
)
from segmentation_utils import build_binary_segmentation_loss, logits_to_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a binary segmentation checkpoint.")
    parser.add_argument("--model", default="deeplabv3_mobilenet", choices=["deeplabv3_mobilenet"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=SEGMENTATION_BATCH_SIZE)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--binary-mask",
        dest="binary_mask",
        action="store_true",
        default=True,
        help="Use the project scheme-B binary masks. Enabled by default.",
    )
    parser.add_argument(
        "--no-binary-mask",
        dest="binary_mask",
        action="store_false",
        help="Debug only: keep raw trimaps. This is invalid for two-class evaluation.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise TypeError(f"Unsupported checkpoint format at {path}")


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_size = target.shape[-2:]
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def extract_main_logits(output: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    if isinstance(output, dict):
        if "out" not in output:
            raise KeyError("TorchVision segmentation output is missing key 'out'.")
        output = output["out"]
    return resize_logits_to_target(output, target)


class SegmentationMetricAccumulator:
    """Accumulate binary segmentation metrics over pixels."""

    def __init__(self) -> None:
        self.total_pixels = 0
        self.correct_pixels = 0
        self.intersections = [0, 0]
        self.unions = [0, 0]
        self.foreground_intersection = 0
        self.foreground_pred_total = 0
        self.foreground_target_total = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().long().reshape(-1).cpu()
        target = target.detach().long().reshape(-1).cpu()

        pred_values = set(int(value) for value in torch.unique(prediction).tolist())
        target_values = set(int(value) for value in torch.unique(target).tolist())
        if not pred_values.issubset({0, 1}):
            raise ValueError(f"Prediction must contain only 0/1 labels, got {sorted(pred_values)}")
        if not target_values.issubset({0, 1}):
            raise ValueError(f"Target must contain only 0/1 labels, got {sorted(target_values)}")

        self.total_pixels += int(target.numel())
        self.correct_pixels += int((prediction == target).sum().item())

        for class_id in (0, 1):
            pred_class = prediction == class_id
            target_class = target == class_id
            self.intersections[class_id] += int(torch.logical_and(pred_class, target_class).sum().item())
            self.unions[class_id] += int(torch.logical_or(pred_class, target_class).sum().item())

        foreground_pred = prediction == 1
        foreground_target = target == 1
        self.foreground_intersection += int(torch.logical_and(foreground_pred, foreground_target).sum().item())
        self.foreground_pred_total += int(foreground_pred.sum().item())
        self.foreground_target_total += int(foreground_target.sum().item())

    def compute(self) -> dict[str, float | int]:
        if self.total_pixels == 0:
            raise ValueError("Cannot compute metrics without any pixels.")

        background_iou = 1.0 if self.unions[0] == 0 else self.intersections[0] / self.unions[0]
        foreground_iou = 1.0 if self.unions[1] == 0 else self.intersections[1] / self.unions[1]
        foreground_total = self.foreground_pred_total + self.foreground_target_total
        dice = 1.0 if foreground_total == 0 else (2 * self.foreground_intersection) / foreground_total

        return {
            "pixel_accuracy": float(self.correct_pixels / self.total_pixels),
            "background_iou": float(background_iou),
            "foreground_iou": float(foreground_iou),
            "miou": float((background_iou + foreground_iou) / 2),
            "dice": float(dice),
            "valid_pixels": int(self.total_pixels),
            "background_intersection": int(self.intersections[0]),
            "foreground_intersection": int(self.intersections[1]),
            "background_union": int(self.unions[0]),
            "foreground_union": int(self.unions[1]),
            "foreground_pred_pixels": int(self.foreground_pred_total),
            "foreground_target_pixels": int(self.foreground_target_total),
        }


def _dataset_metadata(loader: torch.utils.data.DataLoader, start: int, count: int) -> list[dict[str, Any]]:
    dataset = loader.dataset
    indices = getattr(dataset, "indices", None)
    source_dataset = getattr(dataset, "dataset", None)
    image_paths = getattr(source_dataset, "_images", None)
    mask_paths = getattr(source_dataset, "_segs", None)

    metadata: list[dict[str, Any]] = []
    for offset in range(count):
        item = start + offset
        source_index = int(indices[item]) if indices is not None else item
        metadata.append(
            {
                "dataset_index": source_index,
                "image_path": str(Path(image_paths[source_index])) if image_paths is not None else "",
                "mask_path": str(Path(mask_paths[source_index])) if mask_paths is not None else "",
            }
        )
    return metadata


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    accumulator = SegmentationMetricAccumulator()
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0
    seen = 0

    for images, masks in loader:
        batch_size = images.size(0)
        batch_metadata = _dataset_metadata(loader, seen, batch_size)

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        output = model(images)
        logits = extract_main_logits(output, masks)
        loss = criterion(logits, masks)
        predictions = logits_to_prediction(logits)

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        accumulator.update(predictions, masks)

        for item_idx in range(batch_size):
            sample_accumulator = SegmentationMetricAccumulator()
            sample_prediction = predictions[item_idx : item_idx + 1]
            sample_mask = masks[item_idx : item_idx + 1]
            sample_accumulator.update(sample_prediction, sample_mask)
            sample_metrics = sample_accumulator.compute()
            rows.append(
                {
                    **batch_metadata[item_idx],
                    "pixel_accuracy": sample_metrics["pixel_accuracy"],
                    "background_iou": sample_metrics["background_iou"],
                    "foreground_iou": sample_metrics["foreground_iou"],
                    "miou": sample_metrics["miou"],
                    "dice": sample_metrics["dice"],
                    "valid_pixels": sample_metrics["valid_pixels"],
                    "foreground_pred_pixels": sample_metrics["foreground_pred_pixels"],
                    "foreground_target_pixels": sample_metrics["foreground_target_pixels"],
                }
            )

        seen += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, accumulator.compute(), rows


def build_metrics(
    *,
    model_name: str,
    checkpoint_path: Path,
    split: Split,
    img_size: int,
    batch_size: int,
    seed: int,
    val_ratio: float,
    data_root: Path,
    checkpoint: dict[str, Any],
    loss: float,
    metrics: dict[str, float | int],
    num_samples: int,
) -> dict[str, Any]:
    return {
        "dataset": "Oxford-IIIT Pet",
        "task": "binary pet foreground segmentation",
        "model": model_name,
        "split": split,
        "data_root": str(data_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "checkpoint_val_pixel_accuracy": checkpoint.get("val_pixel_accuracy"),
        "checkpoint_val_background_iou": checkpoint.get("val_background_iou"),
        "checkpoint_val_foreground_iou": checkpoint.get("val_foreground_iou"),
        "checkpoint_val_miou": checkpoint.get("val_miou"),
        "checkpoint_val_dice": checkpoint.get("val_dice"),
        "img_size": img_size,
        "batch_size": batch_size,
        "seed": seed,
        "val_ratio": val_ratio,
        "num_classes": SEG_NUM_CLASSES,
        "num_samples": num_samples,
        "mask_rule": {
            "name": "scheme_b_merge_border_into_foreground",
            "mapping": {"1": 1, "2": 0, "3": 1},
            "ignore_index": None,
        },
        "loss": float(loss),
        **metrics,
    }


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, default=_json_default)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_index",
        "image_path",
        "mask_path",
        "pixel_accuracy",
        "background_iou",
        "foreground_iou",
        "miou",
        "dice",
        "valid_pixels",
        "foreground_pred_pixels",
        "foreground_target_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "split",
        "checkpoint_epoch",
        "num_samples",
        "img_size",
        "pixel_accuracy",
        "background_iou",
        "foreground_iou",
        "miou",
        "dice",
        "loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: metrics[name] for name in fieldnames})


def main() -> None:
    args = parse_args()
    model_name = normalize_segmentation_model_name(args.model)
    if not args.binary_mask:
        raise ValueError("Two-class segmentation evaluation requires --binary-mask.")

    ensure_output_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    checkpoint_path = args.checkpoint or get_segmentation_checkpoint_path(model_name)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    img_size = int(args.img_size or checkpoint.get("img_size") or SEGMENTATION_IMAGE_SIZE)
    num_classes = int(checkpoint.get("num_classes") or SEG_NUM_CLASSES)
    if num_classes != SEG_NUM_CLASSES:
        raise ValueError(f"Expected {SEG_NUM_CLASSES} segmentation classes, checkpoint has {num_classes}.")

    loader = build_segmentation_dataloader(
        split=args.split,
        root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        binary_mask=args.binary_mask,
    )

    model = build_segmentation_model(
        model_name,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_binary_segmentation_loss()

    loss, raw_metrics, prediction_rows = run_inference(model, loader, criterion, device)
    metrics = build_metrics(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        data_root=args.data_root,
        checkpoint=checkpoint,
        loss=loss,
        metrics=raw_metrics,
        num_samples=len(loader.dataset),
    )

    metrics_path = args.metrics_out or RESULT_DIR / f"seg_metrics_{model_name}.json"
    predictions_path = args.predictions_out or RESULT_DIR / f"seg_predictions_{model_name}_{args.split}.csv"
    summary_path = args.summary_out or RESULT_DIR / f"seg_summary_{model_name}_{args.split}.csv"

    write_metrics(metrics_path, metrics)
    write_predictions(predictions_path, prediction_rows)
    write_summary(summary_path, metrics)

    print(f"model: {model_name}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"split: {args.split}")
    print(f"samples: {metrics['num_samples']}")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"pixel_accuracy: {metrics['pixel_accuracy']:.6f}")
    print(f"background_iou: {metrics['background_iou']:.6f}")
    print(f"foreground_iou: {metrics['foreground_iou']:.6f}")
    print(f"miou: {metrics['miou']:.6f}")
    print(f"dice: {metrics['dice']:.6f}")
    print(f"metrics: {metrics_path}")
    print(f"predictions: {predictions_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
