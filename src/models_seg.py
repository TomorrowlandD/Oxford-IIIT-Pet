"""Segmentation model builders for Oxford-IIIT Pet foreground masks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from config import CHECKPOINT_DIR, LOG_DIR, SEG_NUM_CLASSES


SegmentationModelName = Literal["deeplabv3_mobilenet"]

SUPPORTED_SEGMENTATION_MODELS: tuple[str, ...] = ("deeplabv3_mobilenet",)


def normalize_segmentation_model_name(model_name: str) -> SegmentationModelName:
    """Normalize user-facing model names to the internal command-line names."""
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "deeplab": "deeplabv3_mobilenet",
        "deeplabv3": "deeplabv3_mobilenet",
        "deeplabv3_mobilenet_v3": "deeplabv3_mobilenet",
        "deeplabv3_mobilenet_v3_large": "deeplabv3_mobilenet",
        "deeplabv3_mobilenetv3": "deeplabv3_mobilenet",
        "deeplabv3_mobilenetv3_large": "deeplabv3_mobilenet",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_SEGMENTATION_MODELS:
        supported = ", ".join(SUPPORTED_SEGMENTATION_MODELS)
        raise ValueError(f"Unsupported segmentation model '{model_name}'. Supported: {supported}")

    return normalized  # type: ignore[return-value]


def _replace_last_conv(module: nn.Sequential, num_classes: int) -> None:
    """Replace the final 1x1 classifier conv in a TorchVision segmentation head."""
    last_layer = module[-1]
    if not isinstance(last_layer, nn.Conv2d):
        raise TypeError(f"Expected final layer to be nn.Conv2d, got {type(last_layer).__name__}")
    module[-1] = nn.Conv2d(last_layer.in_channels, num_classes, kernel_size=1)


def build_deeplabv3_mobilenet(
    num_classes: int = SEG_NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build DeepLabV3-MobileNetV3-Large and replace heads for binary masks.

    TorchVision returns segmentation outputs as a dict. The main logits are in
    ``output["out"]`` with shape ``[B, num_classes, H, W]``.
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None,
    )

    _replace_last_conv(model.classifier, num_classes)
    if model.aux_classifier is not None:
        _replace_last_conv(model.aux_classifier, num_classes)

    return model


def build_segmentation_model(
    model_name: str,
    num_classes: int = SEG_NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build a supported segmentation model by name."""
    normalized_name = normalize_segmentation_model_name(model_name)
    if normalized_name == "deeplabv3_mobilenet":
        return build_deeplabv3_mobilenet(num_classes=num_classes, pretrained=pretrained)

    raise AssertionError(f"Unhandled segmentation model: {normalized_name}")


def get_segmentation_checkpoint_path(
    model_name: str,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> Path:
    """Return the canonical best-checkpoint path for a segmentation model."""
    normalized_name = normalize_segmentation_model_name(model_name)
    return checkpoint_dir / f"best_seg_{normalized_name}.pth"


def get_segmentation_log_path(
    model_name: str,
    log_dir: Path = LOG_DIR,
) -> Path:
    """Return the canonical CSV log path for a segmentation model."""
    normalized_name = normalize_segmentation_model_name(model_name)
    return log_dir / f"{normalized_name}_seg_log.csv"


@torch.no_grad()
def check_segmentation_forward(
    model_name: str = "deeplabv3_mobilenet",
    num_classes: int = SEG_NUM_CLASSES,
    img_size: int = 320,
    pretrained: bool = False,
) -> list[int]:
    """Run a lightweight forward-shape check for Step 8.1."""
    model = build_segmentation_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model.eval()
    batch = torch.randn(2, 3, img_size, img_size)
    output = model(batch)
    if not isinstance(output, dict) or "out" not in output:
        raise TypeError("Expected TorchVision segmentation output dict with key 'out'.")
    logits = output["out"]
    expected_shape = [2, num_classes, img_size, img_size]
    if list(logits.shape) != expected_shape:
        raise ValueError(f"Expected logits shape {expected_shape}, got {list(logits.shape)}")
    return list(logits.shape)
