"""Inspect Oxford-IIIT Pet labels, trimaps, and deterministic splits.

This script covers the project data steps for:
- segmentation trimap value inspection;
- fixed train/validation split from the official trainval split;
- class distribution, dataset samples, and dataset summary outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from config import DATA_DIR, FIGURE_DIR, NUM_CLASSES, RESULT_DIR, SEED, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Oxford-IIIT Pet category labels, trimaps, and fixed splits."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--trimap-samples", type=int, default=24)
    parser.add_argument("--dataset-samples", type=int, default=12)
    parser.add_argument("--scan-all-trimaps", action="store_true")
    return parser.parse_args()


def make_split_indices(length: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    val_size = int(round(length * val_ratio))
    train_size = length - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        range(length), [train_size, val_size], generator=generator
    )
    return list(train_subset.indices), list(val_subset.indices)


def count_labels(labels: Iterable[int]) -> Counter[int]:
    return Counter(int(label) for label in labels)


def write_class_distribution_csv(
    path: Path,
    classes: list[str],
    split_counts: dict[str, Counter[int]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["split", "class_index", "class_name", "count"]
        )
        writer.writeheader()
        for split, counts in split_counts.items():
            for class_index, class_name in enumerate(classes):
                writer.writerow(
                    {
                        "split": split,
                        "class_index": class_index,
                        "class_name": class_name,
                        "count": counts.get(class_index, 0),
                    }
                )


def plot_class_distribution(path: Path, classes: list[str], counts: Counter[int]) -> None:
    x = np.arange(len(classes))
    values = [counts.get(index, 0) for index in x]

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=180)
    ax.bar(x, values, color="#4B8BBE", edgecolor="#1F4E79", linewidth=0.4)
    ax.set_title("Oxford-IIIT Pet Train Split Class Distribution")
    ax.set_xlabel("Class index")
    ax.set_ylabel("Number of images")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in x], rotation=0, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        ax.text(index, value + 0.5, str(value), ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_samples(
    path: Path,
    dataset: OxfordIIITPet,
    classes: list[str],
    sample_count: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    sample_size = min(sample_count, len(dataset))
    sample_indices = sorted(rng.choice(len(dataset), size=sample_size, replace=False).tolist())

    cols = 4
    rows = int(np.ceil(sample_size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.7), dpi=180)
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    sample_records: list[dict[str, object]] = []

    for axis in axes_array.ravel():
        axis.axis("off")

    for axis, index in zip(axes_array.ravel(), sample_indices):
        image_path = Path(dataset._images[index])
        label = int(dataset._labels[index])
        class_name = classes[label]
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{label}: {class_name}", fontsize=8)
        axis.axis("off")
        sample_records.append(
            {
                "split": "official trainval",
                "index": int(index),
                "image_path": str(image_path),
                "label": label,
                "class_name": class_name,
            }
        )

    fig.suptitle("Oxford-IIIT Pet Dataset Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return sample_records


def load_trimap_values(mask_path: Path) -> list[int]:
    with Image.open(mask_path) as mask:
        return [int(value) for value in np.unique(np.array(mask))]


def inspect_trimaps(
    trainval_seg: OxfordIIITPet,
    test_seg: OxfordIIITPet,
    sample_count: int,
    seed: int,
    scan_all: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    split_datasets = {"trainval": trainval_seg, "test": test_seg}
    sample_records: list[dict[str, object]] = []
    split_unique_values: dict[str, list[int]] = {}

    for split, dataset in split_datasets.items():
        sample_size = min(sample_count, len(dataset))
        sample_indices = sorted(rng.choice(len(dataset), size=sample_size, replace=False).tolist())
        sample_unique = set()

        for index in sample_indices:
            values = load_trimap_values(dataset._segs[index])
            sample_unique.update(values)
            sample_records.append(
                {
                    "split": split,
                    "index": int(index),
                    "mask_path": str(dataset._segs[index]),
                    "unique_values": values,
                }
            )

        if scan_all:
            all_unique = set()
            for mask_path in dataset._segs:
                all_unique.update(load_trimap_values(mask_path))
            split_unique_values[split] = sorted(int(value) for value in all_unique)
        else:
            split_unique_values[split] = sorted(int(value) for value in sample_unique)

    return {
        "sample_count_per_split": sample_count,
        "scan_all_trimap_values": scan_all,
        "split_unique_values": split_unique_values,
        "sample_records": sample_records,
        "value_meaning_for_report": {
            "1": "pet foreground",
            "2": "background",
            "3": "border",
        },
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    trainval_cls = OxfordIIITPet(
        root=str(args.data_root), split="trainval", target_types="category", download=False
    )
    test_cls = OxfordIIITPet(
        root=str(args.data_root), split="test", target_types="category", download=False
    )
    trainval_seg = OxfordIIITPet(
        root=str(args.data_root), split="trainval", target_types="segmentation", download=False
    )
    test_seg = OxfordIIITPet(
        root=str(args.data_root), split="test", target_types="segmentation", download=False
    )

    classes = list(trainval_cls.classes)
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, got {len(classes)}")

    train_indices, val_indices = make_split_indices(
        len(trainval_cls), val_ratio=args.val_ratio, seed=args.seed
    )
    train_labels = [trainval_cls._labels[index] for index in train_indices]
    val_labels = [trainval_cls._labels[index] for index in val_indices]
    test_labels = list(test_cls._labels)

    split_counts = {
        "train": count_labels(train_labels),
        "val": count_labels(val_labels),
        "test": count_labels(test_labels),
    }
    trainval_counts = count_labels(trainval_cls._labels)

    split_indices_path = RESULT_DIR / f"split_indices_seed{args.seed}.json"
    split_indices = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source_split": "official trainval",
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_split": "official test",
    }
    split_indices_path.write_text(
        json.dumps(split_indices, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    class_distribution_path = RESULT_DIR / "class_distribution.csv"
    write_class_distribution_csv(class_distribution_path, classes, split_counts)

    figure_path = FIGURE_DIR / "fig_class_distribution.png"
    plot_class_distribution(figure_path, classes, split_counts["train"])

    dataset_samples_path = FIGURE_DIR / "fig_dataset_samples.png"
    dataset_sample_records = plot_dataset_samples(
        dataset_samples_path,
        trainval_cls,
        classes,
        sample_count=args.dataset_samples,
        seed=args.seed,
    )

    trimap_info = inspect_trimaps(
        trainval_seg=trainval_seg,
        test_seg=test_seg,
        sample_count=args.trimap_samples,
        seed=args.seed,
        scan_all=args.scan_all_trimaps,
    )

    summary = {
        "dataset": "Oxford-IIIT Pet",
        "data_root": str(args.data_root),
        "num_classes": len(classes),
        "classes": classes,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_sizes": {
            "official_trainval": len(trainval_cls),
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_cls),
        },
        "official_trainval_class_counts": {
            classes[index]: trainval_counts.get(index, 0) for index in range(len(classes))
        },
        "split_class_counts": {
            split: {classes[index]: counts.get(index, 0) for index in range(len(classes))}
            for split, counts in split_counts.items()
        },
        "trimap": trimap_info,
        "dataset_samples": dataset_sample_records,
        "outputs": {
            "split_indices": str(split_indices_path),
            "class_distribution_csv": str(class_distribution_path),
            "class_distribution_figure": str(figure_path),
            "dataset_samples_figure": str(dataset_samples_path),
        },
    }

    summary_path = RESULT_DIR / "data_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Data inspection finished.")
    print(f"train/val/test sizes: {len(train_indices)}/{len(val_indices)}/{len(test_cls)}")
    print(f"trimap unique values: {trimap_info['split_unique_values']}")
    print(f"summary: {summary_path}")
    print(f"class distribution: {class_distribution_path}")
    print(f"split indices: {split_indices_path}")
    print(f"figure: {figure_path}")
    print(f"dataset samples: {dataset_samples_path}")


if __name__ == "__main__":
    main()
