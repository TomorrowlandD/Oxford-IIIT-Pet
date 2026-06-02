"""Classification model builders for Oxford-IIIT Pet experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from config import CHECKPOINT_DIR, LOG_DIR, NUM_CLASSES


ClassificationModelName = Literal["resnet18", "efficientnet_b0"]

SUPPORTED_CLASSIFICATION_MODELS: tuple[str, ...] = ("resnet18", "efficientnet_b0")


def normalize_classification_model_name(model_name: str) -> ClassificationModelName:
    """Normalize user-facing model names to the internal command-line names."""
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "resnet": "resnet18",
        "resnet_18": "resnet18",
        "efficientnet": "efficientnet_b0",
        "efficientnetb0": "efficientnet_b0",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_CLASSIFICATION_MODELS:
        supported = ", ".join(SUPPORTED_CLASSIFICATION_MODELS)
        raise ValueError(f"Unsupported classification model '{model_name}'. Supported: {supported}")

    return normalized  # type: ignore[return-value]


def build_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet18 classifier with a project-specific output head."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build an EfficientNet-B0 classifier with a project-specific output head."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_classification_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build a supported classification model by name."""
    normalized_name = normalize_classification_model_name(model_name)
    if normalized_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if normalized_name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)

    raise AssertionError(f"Unhandled classification model: {normalized_name}")


def get_classification_checkpoint_path(
    model_name: str,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> Path:
    """Return the canonical best-checkpoint path for a classification model."""
    normalized_name = normalize_classification_model_name(model_name)
    return checkpoint_dir / f"best_cls_{normalized_name}.pth"


def get_classification_log_path(
    model_name: str,
    log_dir: Path = LOG_DIR,
) -> Path:
    """Return the canonical CSV log path for a classification model."""
    normalized_name = normalize_classification_model_name(model_name)
    return log_dir / f"{normalized_name}_cls_log.csv"
