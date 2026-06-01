"""Shared configuration defaults for the Oxford-IIIT Pet experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

try:
    import torch
except ImportError:  # Allows lightweight tooling to import paths without torch.
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OXFORD_PET_ROOT = DATA_DIR / "oxford-iiit-pet"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"

SEED = 42
NUM_CLASSES = 37
SEG_NUM_CLASSES = 2

CLASSIFICATION_IMAGE_SIZE = 224
SEGMENTATION_IMAGE_SIZE = 320

CLASSIFICATION_BATCH_SIZE = 32
SEGMENTATION_BATCH_SIZE = 4

NUM_WORKERS = 4
PIN_MEMORY = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassificationConfig:
    model: str = "efficientnet_b0"
    img_size: int = CLASSIFICATION_IMAGE_SIZE
    batch_size: int = CLASSIFICATION_BATCH_SIZE
    epochs: int = 25
    lr: float = 1e-4
    pretrained: bool = True


@dataclass(frozen=True)
class SegmentationConfig:
    model: str = "deeplabv3_mobilenet"
    img_size: int = SEGMENTATION_IMAGE_SIZE
    batch_size: int = SEGMENTATION_BATCH_SIZE
    epochs: int = 30
    lr: float = 1e-4
    binary_mask: bool = True


def ensure_output_dirs() -> None:
    """Create output directories used by training, evaluation, and plotting."""
    for path in (CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, RESULT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Set common random seeds for reproducible data splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
