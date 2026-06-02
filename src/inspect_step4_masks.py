"""Create Step 4 trimap-to-binary-mask visual checks.

This script verifies the selected scheme-B conversion for Oxford-IIIT Pet
segmentation labels:
- raw trimap value 2 maps to background 0;
- raw trimap values 1 and 3 map to foreground 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from config import DATA_DIR, FIGURE_DIR, RESULT_DIR, SEED, ensure_output_dirs
from data_transforms import convert_trimap_to_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Step 4 raw trimap and binary foreground mask checks."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--samples-per-split", type=int, default=6)
    return parser.parse_args()


def _pick_indices(dataset: OxfordIIITPet, count: int, seed: int) -> list[int]:
    sample_count = min(count, len(dataset))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(dataset), size=sample_count, replace=False).astype(int).tolist())


def _overlay_foreground(image: Image.Image, binary_mask: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask_image = Image.fromarray(binary_mask.astype(np.uint8), mode="L").resize(
        image.size,
        resample=Image.Resampling.NEAREST,
    )
    mask_array = np.asarray(mask_image, dtype=bool)

    overlay = image_array.copy()
    color = np.array([0.95, 0.12, 0.10], dtype=np.float32)
    alpha = 0.38
    overlay[mask_array] = (1.0 - alpha) * overlay[mask_array] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def _validate_values(raw_values: list[int], binary_values: list[int]) -> None:
    if not set(raw_values).issubset({1, 2, 3}):
        raise ValueError(f"Unexpected raw trimap values: {raw_values}")
    if not set(binary_values).issubset({0, 1}):
        raise ValueError(f"Unexpected binary mask values: {binary_values}")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    split_names = ("trainval", "test")
    datasets = {
        split: OxfordIIITPet(
            root=str(args.data_root),
            split=split,
            target_types="segmentation",
            download=False,
        )
        for split in split_names
    }

    selected: list[tuple[str, int]] = []
    for offset, split in enumerate(split_names):
        for index in _pick_indices(
            datasets[split],
            count=args.samples_per_split,
            seed=args.seed + offset,
        ):
            selected.append((split, index))

    rows = len(selected)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.0), dpi=180)
    axes_array = np.atleast_2d(axes)
    records: list[dict[str, object]] = []

    for row, (split, index) in enumerate(selected):
        dataset = datasets[split]
        image_path = Path(dataset._images[index])
        mask_path = Path(dataset._segs[index])

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            raw_mask = np.asarray(mask_file, dtype=np.int64)

        binary_mask = convert_trimap_to_binary_mask(raw_mask)
        raw_values = [int(value) for value in np.unique(raw_mask)]
        binary_values = [int(value) for value in np.unique(binary_mask)]
        _validate_values(raw_values, binary_values)

        foreground_pixels = int((binary_mask == 1).sum())
        background_pixels = int((binary_mask == 0).sum())
        total_pixels = int(binary_mask.size)

        row_axes = axes_array[row]
        row_axes[0].imshow(image)
        row_axes[0].set_title(f"{split} #{index}", fontsize=8)
        row_axes[0].axis("off")

        row_axes[1].imshow(raw_mask, cmap="viridis", vmin=1, vmax=3)
        row_axes[1].set_title(f"Raw trimap {raw_values}", fontsize=8)
        row_axes[1].axis("off")

        row_axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
        row_axes[2].set_title(f"Binary mask {binary_values}", fontsize=8)
        row_axes[2].axis("off")

        row_axes[3].imshow(_overlay_foreground(image, binary_mask))
        row_axes[3].set_title("Foreground overlay", fontsize=8)
        row_axes[3].axis("off")

        records.append(
            {
                "split": split,
                "index": int(index),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "raw_trimap_values": raw_values,
                "binary_mask_values": binary_values,
                "value_counts": {
                    "background_0": background_pixels,
                    "foreground_1": foreground_pixels,
                    "total": total_pixels,
                },
                "foreground_ratio": float(foreground_pixels / total_pixels),
            }
        )

    fig.suptitle("Step 4 Mask Conversion Samples", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=1.0, w_pad=0.7)

    figure_path = FIGURE_DIR / "fig_mask_samples.png"
    result_path = RESULT_DIR / "mask_conversion_check.json"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    result = {
        "step": "4.3",
        "seed": args.seed,
        "samples_per_split": args.samples_per_split,
        "sample_count": len(records),
        "selected_conversion_rule": {
            "name": "scheme_b_merge_border_into_foreground",
            "mapping": {"1": 1, "2": 0, "3": 1},
            "num_classes": 2,
            "ignore_index": None,
            "loss": "CrossEntropyLoss without ignore_index",
            "metrics": "Pixel Accuracy, background/foreground IoU, mIoU, Dice over 0/1 labels",
        },
        "checks": {
            "raw_values_ok": all(set(record["raw_trimap_values"]).issubset({1, 2, 3}) for record in records),
            "binary_values_ok": all(set(record["binary_mask_values"]).issubset({0, 1}) for record in records),
            "sample_count_ok": 6 <= len(records) <= 12,
        },
        "records": records,
        "outputs": {
            "figure": str(figure_path),
            "result": str(result_path),
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Step 4 mask conversion check finished.")
    print(f"samples: {len(records)}")
    print(f"figure: {figure_path}")
    print(f"result: {result_path}")


if __name__ == "__main__":
    main()
