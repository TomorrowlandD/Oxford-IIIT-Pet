"""Batch-level checks for Step 3 preprocessing DataLoaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import (
    CLASSIFICATION_IMAGE_SIZE,
    DATA_DIR,
    NUM_CLASSES,
    RESULT_DIR,
    SEED,
    SEGMENTATION_IMAGE_SIZE,
    ensure_output_dirs,
    set_seed,
)
from data_loaders import build_classification_dataloader, build_segmentation_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate classification and segmentation batches.")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--cls-img-size", type=int, default=CLASSIFICATION_IMAGE_SIZE)
    parser.add_argument("--seg-img-size", type=int, default=SEGMENTATION_IMAGE_SIZE)
    parser.add_argument("--cls-batch-size", type=int, default=4)
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def check_classification_batch(args: argparse.Namespace, split: str) -> dict[str, object]:
    loader = build_classification_dataloader(
        split=split,
        root=args.data_root,
        img_size=args.cls_img_size,
        batch_size=args.cls_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    images, labels = next(iter(loader))
    label_min = int(labels.min().item())
    label_max = int(labels.max().item())

    expected_image_shape = [images.shape[0], 3, args.cls_img_size, args.cls_img_size]
    checks = {
        "image_shape_ok": list(images.shape) == expected_image_shape,
        "label_shape_ok": labels.ndim == 1 and labels.shape[0] == images.shape[0],
        "label_range_ok": 0 <= label_min and label_max < NUM_CLASSES,
        "image_dtype_ok": images.dtype == torch.float32,
        "label_dtype_ok": labels.dtype == torch.long,
    }

    return {
        "split": split,
        "dataset_size": len(loader.dataset),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "label_min": label_min,
        "label_max": label_max,
        "checks": checks,
        "passed": all(checks.values()),
    }


def check_segmentation_batch(args: argparse.Namespace, split: str) -> dict[str, object]:
    loader = build_segmentation_dataloader(
        split=split,
        root=args.data_root,
        img_size=args.seg_img_size,
        batch_size=args.seg_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=False,
        binary_mask=False,
    )
    images, masks = next(iter(loader))
    mask_values = sorted(int(value) for value in torch.unique(masks).tolist())

    expected_image_shape = [images.shape[0], 3, args.seg_img_size, args.seg_img_size]
    expected_mask_shape = [images.shape[0], args.seg_img_size, args.seg_img_size]
    checks = {
        "image_shape_ok": list(images.shape) == expected_image_shape,
        "mask_shape_ok": list(masks.shape) == expected_mask_shape,
        "raw_trimap_values_ok": set(mask_values).issubset({1, 2, 3}),
        "image_dtype_ok": images.dtype == torch.float32,
        "mask_dtype_ok": masks.dtype == torch.long,
    }

    return {
        "split": split,
        "dataset_size": len(loader.dataset),
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "image_dtype": str(images.dtype),
        "mask_dtype": str(masks.dtype),
        "mask_values": mask_values,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    set_seed(args.seed)

    splits = ["train", "val", "test"]
    classification = [check_classification_batch(args, split) for split in splits]
    segmentation = [check_segmentation_batch(args, split) for split in splits]
    passed = all(record["passed"] for record in classification + segmentation)

    result = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "classification_image_size": args.cls_img_size,
        "segmentation_image_size": args.seg_img_size,
        "note": "Segmentation masks remain raw trimaps in Step 3. Binary conversion starts in Step 4.",
        "classification": classification,
        "segmentation": segmentation,
        "passed": passed,
    }

    result_path = RESULT_DIR / "dataloader_check.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DataLoader check finished.")
    print(f"passed: {passed}")
    for record in classification:
        print(
            f"cls {record['split']}: size={record['dataset_size']}, "
            f"images={record['image_shape']}, labels={record['label_shape']}"
        )
    for record in segmentation:
        print(
            f"seg {record['split']}: size={record['dataset_size']}, "
            f"images={record['image_shape']}, masks={record['mask_shape']}, "
            f"values={record['mask_values']}"
        )
    print(f"result: {result_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
