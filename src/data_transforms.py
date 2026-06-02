"""Image preprocessing and augmentation pipelines for Oxford-IIIT Pet.

The functions in this module intentionally keep classification and
segmentation preprocessing separate because segmentation masks require
nearest-neighbor interpolation and spatial transforms must stay synchronized
with the image.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from config import (
    CLASSIFICATION_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEGMENTATION_IMAGE_SIZE,
)


Split = Literal["train", "val", "test"]


def convert_trimap_to_binary_mask(mask: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert Oxford-IIIT Pet trimap values to a binary foreground mask.

    Scheme B is used throughout the project:
    original value 2 -> background 0; original values 1 and 3 -> foreground 1.
    """
    if isinstance(mask, torch.Tensor):
        invalid = ~((mask == 1) | (mask == 2) | (mask == 3))
        if bool(invalid.any().item()):
            values = sorted(int(value) for value in torch.unique(mask).tolist())
            raise ValueError(f"Unexpected trimap values: {values}")
        return torch.where(mask == 2, torch.zeros_like(mask), torch.ones_like(mask)).long()

    mask_array = np.asarray(mask)
    valid = np.isin(mask_array, [1, 2, 3])
    if not bool(valid.all()):
        values = sorted(int(value) for value in np.unique(mask_array).tolist())
        raise ValueError(f"Unexpected trimap values: {values}")
    return np.where(mask_array == 2, 0, 1).astype(np.int64)


def build_classification_transform(
    split: Split,
    img_size: int = CLASSIFICATION_IMAGE_SIZE,
) -> transforms.Compose:
    """Build classification transforms for train/validation/test splits."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.75, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.12,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if split in {"val", "test"}:
        resize_size = int(round(img_size * 256 / 224))
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported split: {split}")


@dataclass(frozen=True)
class SegmentationTransform:
    """Synchronized image/mask transform for foreground segmentation.

    The mask is returned as a ``long`` tensor with shape ``[H, W]``. By default,
    the Oxford-IIIT Pet trimap is converted to the project-selected binary mask:
    background ``0`` and foreground ``1``.
    """

    split: Split
    img_size: int = SEGMENTATION_IMAGE_SIZE
    binary_mask: bool = True
    hflip_prob: float = 0.5
    train_scale: tuple[float, float] = (0.85, 1.0)
    train_ratio: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        if self.split == "train":
            image, mask = self._apply_train_spatial_transforms(image, mask)
        else:
            image, mask = self._apply_eval_spatial_transforms(image, mask)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.array(mask, dtype=np.int64), dtype=torch.long)
        if self.binary_mask:
            mask_tensor = convert_trimap_to_binary_mask(mask_tensor)

        return image_tensor, mask_tensor

    def _apply_train_spatial_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image,
            scale=self.train_scale,
            ratio=self.train_ratio,
        )
        image = F.resized_crop(
            image,
            i,
            j,
            h,
            w,
            size=[self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = F.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            size=[self.img_size, self.img_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if random.random() < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        return image, mask

    def _apply_eval_spatial_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        image = F.resize(
            image,
            [self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = F.resize(
            mask,
            [self.img_size, self.img_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask


def build_segmentation_transform(
    split: Split,
    img_size: int = SEGMENTATION_IMAGE_SIZE,
    binary_mask: bool = True,
) -> SegmentationTransform:
    """Build synchronized segmentation transforms for train/validation/test splits."""
    return SegmentationTransform(split=split, img_size=img_size, binary_mask=binary_mask)
