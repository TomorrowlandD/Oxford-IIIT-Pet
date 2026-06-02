"""Dataset and DataLoader helpers for Oxford-IIIT Pet experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet

from config import (
    CLASSIFICATION_BATCH_SIZE,
    CLASSIFICATION_IMAGE_SIZE,
    DATA_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    SEGMENTATION_BATCH_SIZE,
    SEGMENTATION_IMAGE_SIZE,
)
from data_transforms import build_classification_transform, build_segmentation_transform


Split = Literal["train", "val", "test"]


def make_train_val_indices(
    length: int,
    val_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[list[int], list[int]]:
    """Create the fixed train/validation split from official trainval records."""
    val_size = int(round(length * val_ratio))
    train_size = length - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        range(length), [train_size, val_size], generator=generator
    )
    return list(train_subset.indices), list(val_subset.indices)


def _select_indices(length: int, split: Split, val_ratio: float, seed: int) -> list[int]:
    if split == "test":
        return list(range(length))

    train_indices, val_indices = make_train_val_indices(length, val_ratio=val_ratio, seed=seed)
    return train_indices if split == "train" else val_indices


class OxfordPetClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Classification subset with the project-fixed train/val/test split."""

    def __init__(
        self,
        root: Path = DATA_DIR,
        split: Split = "train",
        img_size: int = CLASSIFICATION_IMAGE_SIZE,
        val_ratio: float = 0.2,
        seed: int = SEED,
        download: bool = False,
    ) -> None:
        self.split = split
        source_split = "test" if split == "test" else "trainval"
        self.dataset = OxfordIIITPet(
            root=str(root),
            split=source_split,
            target_types="category",
            download=download,
        )
        self.indices = _select_indices(len(self.dataset), split, val_ratio=val_ratio, seed=seed)
        self.transform = build_classification_transform(split, img_size=img_size)
        self.classes = list(self.dataset.classes)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        image_path = Path(self.dataset._images[index])
        label = int(self.dataset._labels[index])

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class OxfordPetSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Segmentation subset with optional raw-trimap access for inspection."""

    def __init__(
        self,
        root: Path = DATA_DIR,
        split: Split = "train",
        img_size: int = SEGMENTATION_IMAGE_SIZE,
        val_ratio: float = 0.2,
        seed: int = SEED,
        binary_mask: bool = True,
        download: bool = False,
    ) -> None:
        self.split = split
        source_split = "test" if split == "test" else "trainval"
        self.dataset = OxfordIIITPet(
            root=str(root),
            split=source_split,
            target_types="segmentation",
            download=download,
        )
        self.indices = _select_indices(len(self.dataset), split, val_ratio=val_ratio, seed=seed)
        self.transform = build_segmentation_transform(
            split,
            img_size=img_size,
            binary_mask=binary_mask,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        image_path = Path(self.dataset._images[index])
        mask_path = Path(self.dataset._segs[index])

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.copy()

        return self.transform(image, mask)


def build_classification_dataloader(
    split: Split,
    root: Path = DATA_DIR,
    img_size: int = CLASSIFICATION_IMAGE_SIZE,
    batch_size: int = CLASSIFICATION_BATCH_SIZE,
    val_ratio: float = 0.2,
    seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    download: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = OxfordPetClassificationDataset(
        root=root,
        split=split,
        img_size=img_size,
        val_ratio=val_ratio,
        seed=seed,
        download=download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_segmentation_dataloader(
    split: Split,
    root: Path = DATA_DIR,
    img_size: int = SEGMENTATION_IMAGE_SIZE,
    batch_size: int = SEGMENTATION_BATCH_SIZE,
    val_ratio: float = 0.2,
    seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    binary_mask: bool = True,
    download: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = OxfordPetSegmentationDataset(
        root=root,
        split=split,
        img_size=img_size,
        val_ratio=val_ratio,
        seed=seed,
        binary_mask=binary_mask,
        download=download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
