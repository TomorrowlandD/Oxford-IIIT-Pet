"""Loss and metrics for binary foreground segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BinarySegmentationMetrics:
    pixel_accuracy: float
    background_iou: float
    foreground_iou: float
    miou: float
    dice: float
    valid_pixels: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def build_binary_segmentation_loss() -> nn.CrossEntropyLoss:
    """Build the Step 4 scheme-B segmentation loss.

    Scheme B maps all labels to ordinary ``0/1`` classes, so no ignore index is
    used. Model logits should have shape ``[B, 2, H, W]`` and masks should have
    shape ``[B, H, W]`` with dtype ``torch.long``.
    """
    return nn.CrossEntropyLoss()


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    """Convert two-class logits to a ``long`` prediction mask."""
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits with shape [B, 2, H, W], got {list(logits.shape)}")
    return logits.argmax(dim=1).long()


def _as_prediction(prediction_or_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction_or_logits.ndim == target.ndim + 1:
        return logits_to_prediction(prediction_or_logits)
    if prediction_or_logits.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{list(prediction_or_logits.shape)} vs {list(target.shape)}"
        )
    return prediction_or_logits.long()


def _validate_binary_mask(mask: torch.Tensor, name: str) -> None:
    values = set(int(value) for value in torch.unique(mask).detach().cpu().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 labels, got {sorted(values)}")


@torch.no_grad()
def compute_binary_segmentation_metrics(
    prediction_or_logits: torch.Tensor,
    target: torch.Tensor,
) -> BinarySegmentationMetrics:
    """Compute Pixel Accuracy, per-class IoU, mIoU, and foreground Dice.

    This metric implementation matches Step 4 scheme B: every pixel is a normal
    background/foreground label and there is no ignored boundary value.
    """
    target = target.long()
    prediction = _as_prediction(prediction_or_logits, target)
    _validate_binary_mask(prediction, "prediction")
    _validate_binary_mask(target, "target")

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    valid_pixels = int(target.numel())
    if valid_pixels == 0:
        raise ValueError("Cannot compute segmentation metrics on an empty target.")

    correct = (prediction == target).sum().item()
    pixel_accuracy = float(correct / valid_pixels)

    ious: list[float] = []
    for class_id in (0, 1):
        pred_class = prediction == class_id
        target_class = target == class_id
        intersection = torch.logical_and(pred_class, target_class).sum().item()
        union = torch.logical_or(pred_class, target_class).sum().item()
        iou = 1.0 if union == 0 else float(intersection / union)
        ious.append(iou)

    foreground_pred = prediction == 1
    foreground_target = target == 1
    foreground_intersection = torch.logical_and(foreground_pred, foreground_target).sum().item()
    foreground_total = foreground_pred.sum().item() + foreground_target.sum().item()
    dice = 1.0 if foreground_total == 0 else float((2 * foreground_intersection) / foreground_total)

    return BinarySegmentationMetrics(
        pixel_accuracy=pixel_accuracy,
        background_iou=ious[0],
        foreground_iou=ious[1],
        miou=float(sum(ious) / len(ious)),
        dice=dice,
        valid_pixels=valid_pixels,
    )
