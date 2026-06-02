"""Plot Step 8 segmentation training curves from the saved CSV log."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURE_DIR, LOG_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DeepLabV3-MobileNetV3 training curves.")
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_DIR / "deeplabv3_mobilenet_seg_log.csv",
        help="Path to the segmentation CSV training log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURE_DIR / "fig_seg_training_curve.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    df = pd.read_csv(args.log)
    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_miou",
        "val_dice",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required log columns: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)

    axes[0].plot(df["epoch"], df["train_loss"], marker="o", markersize=3, label="Train loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="s", markersize=3, label="Val loss")
    axes[0].set_title("Segmentation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    axes[0].legend(frameon=False)

    axes[1].plot(df["epoch"], df["val_miou"], marker="o", markersize=3, label="Val mIoU")
    axes[1].plot(df["epoch"], df["val_dice"], marker="s", markersize=3, label="Val Dice")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.88, 0.97)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
