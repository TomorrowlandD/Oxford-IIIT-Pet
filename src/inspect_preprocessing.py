"""Visual spot checks for preprocessing and augmentation outputs.

This script supports step 3.3 of the project plan. It samples several
Oxford-IIIT Pet records, applies the classification and segmentation
preprocessing pipelines, and saves a visual comparison plus a small JSON
record of tensor shapes and mask values.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from config import (
    DATA_DIR,
    FIGURE_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESULT_DIR,
    SEED,
    ensure_output_dirs,
    set_seed,
)
from data_transforms import build_classification_transform, build_segmentation_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual spot checks for classification and segmentation preprocessing."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--cls-img-size", type=int, default=224)
    parser.add_argument("--seg-img-size", type=int, default=320)
    return parser.parse_args()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    image = tensor.detach().cpu() * std.cpu() + mean.cpu()
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return image


def show_mask(axis: plt.Axes, mask: Image.Image | torch.Tensor, title: str) -> None:
    if isinstance(mask, torch.Tensor):
        mask_array = mask.detach().cpu().numpy()
    else:
        mask_array = np.array(mask)
    axis.imshow(mask_array, cmap="viridis", vmin=1, vmax=3)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    set_seed(args.seed)
    random.seed(args.seed)

    cls_dataset = OxfordIIITPet(
        root=str(args.data_root), split="trainval", target_types="category", download=False
    )
    seg_dataset = OxfordIIITPet(
        root=str(args.data_root), split="trainval", target_types="segmentation", download=False
    )

    sample_count = min(args.samples, len(cls_dataset), len(seg_dataset))
    rng = np.random.default_rng(args.seed)
    sample_indices = sorted(
        rng.choice(len(cls_dataset), size=sample_count, replace=False).astype(int).tolist()
    )

    cls_train_transform = build_classification_transform("train", img_size=args.cls_img_size)
    cls_eval_transform = build_classification_transform("val", img_size=args.cls_img_size)
    seg_train_transform = build_segmentation_transform(
        "train",
        img_size=args.seg_img_size,
        binary_mask=False,
    )
    seg_eval_transform = build_segmentation_transform(
        "val",
        img_size=args.seg_img_size,
        binary_mask=False,
    )

    rows = sample_count
    cols = 7
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.2), dpi=180)
    axes_array = np.atleast_2d(axes)
    records: list[dict[str, object]] = []

    for row, index in enumerate(sample_indices):
        image_path = Path(cls_dataset._images[index])
        mask_path = Path(seg_dataset._segs[index])
        label = int(cls_dataset._labels[index])
        class_name = cls_dataset.classes[label]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.copy()

        cls_train = cls_train_transform(image)
        cls_eval = cls_eval_transform(image)
        seg_train_image, seg_train_mask = seg_train_transform(image, mask)
        seg_eval_image, seg_eval_mask = seg_eval_transform(image, mask)

        original_mask_values = [int(value) for value in np.unique(np.array(mask))]
        train_mask_values = [int(value) for value in torch.unique(seg_train_mask).tolist()]
        eval_mask_values = [int(value) for value in torch.unique(seg_eval_mask).tolist()]

        row_axes = axes_array[row]
        row_axes[0].imshow(image)
        row_axes[0].set_title(f"Original\n{label}: {class_name}", fontsize=8)
        row_axes[0].axis("off")

        row_axes[1].imshow(denormalize_image(cls_train))
        row_axes[1].set_title("Cls train", fontsize=8)
        row_axes[1].axis("off")

        row_axes[2].imshow(denormalize_image(cls_eval))
        row_axes[2].set_title("Cls val/test", fontsize=8)
        row_axes[2].axis("off")

        show_mask(row_axes[3], mask, f"Raw trimap\n{original_mask_values}")

        row_axes[4].imshow(denormalize_image(seg_train_image))
        row_axes[4].set_title("Seg train image", fontsize=8)
        row_axes[4].axis("off")

        show_mask(row_axes[5], seg_train_mask, f"Seg train mask\n{train_mask_values}")

        show_mask(row_axes[6], seg_eval_mask, f"Seg val mask\n{eval_mask_values}")

        records.append(
            {
                "index": index,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": label,
                "class_name": class_name,
                "classification_train_shape": list(cls_train.shape),
                "classification_eval_shape": list(cls_eval.shape),
                "segmentation_train_image_shape": list(seg_train_image.shape),
                "segmentation_train_mask_shape": list(seg_train_mask.shape),
                "segmentation_eval_image_shape": list(seg_eval_image.shape),
                "segmentation_eval_mask_shape": list(seg_eval_mask.shape),
                "raw_trimap_values": original_mask_values,
                "segmentation_train_mask_values": train_mask_values,
                "segmentation_eval_mask_values": eval_mask_values,
            }
        )

    fig.suptitle("Preprocessing Spot Check", fontsize=12)
    fig.tight_layout()

    figure_path = FIGURE_DIR / "fig_preprocessing_check.png"
    result_path = RESULT_DIR / "preprocessing_check.json"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    result = {
        "seed": args.seed,
        "sample_indices": sample_indices,
        "classification_image_size": args.cls_img_size,
        "segmentation_image_size": args.seg_img_size,
        "note": "Segmentation masks are raw Oxford-IIIT Pet trimaps at this step; binary conversion is handled in step 4.",
        "records": records,
        "outputs": {
            "figure": str(figure_path),
            "result": str(result_path),
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Preprocessing spot check finished.")
    print(f"figure: {figure_path}")
    print(f"result: {result_path}")


if __name__ == "__main__":
    main()
