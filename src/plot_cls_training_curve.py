"""Plot Step 6-7 classification training curves from saved CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURE_DIR, LOG_DIR, ensure_output_dirs


MODEL_LOGS = {
    "ResNet18": LOG_DIR / "resnet18_cls_log.csv",
    "EfficientNet-B0": LOG_DIR / "efficientnet_b0_cls_log.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot classification training curves.")
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURE_DIR / "fig_cls_training_curve.png",
        help="Output image path.",
    )
    return parser.parse_args()


def read_log(path: Path) -> pd.DataFrame:
    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    }
    df = pd.read_csv(path)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    logs = {name: read_log(path) for name, path in MODEL_LOGS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=180)
    colors = {
        "ResNet18": "#2F6FAD",
        "EfficientNet-B0": "#C15A2A",
    }

    for name, df in logs.items():
        color = colors[name]
        axes[0].plot(
            df["epoch"],
            df["train_loss"],
            marker="o",
            markersize=3,
            color=color,
            linestyle="-",
            label=f"{name} train",
        )
        axes[0].plot(
            df["epoch"],
            df["val_loss"],
            marker="s",
            markersize=3,
            color=color,
            linestyle="--",
            label=f"{name} val",
        )
        axes[1].plot(
            df["epoch"],
            df["val_accuracy"],
            marker="o",
            markersize=3,
            color=color,
            label=name,
        )
        axes[2].plot(
            df["epoch"],
            df["val_macro_f1"],
            marker="o",
            markersize=3,
            color=color,
            label=name,
        )

    axes[0].set_title("Classification Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.70, 0.96)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0.70, 0.96)
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
