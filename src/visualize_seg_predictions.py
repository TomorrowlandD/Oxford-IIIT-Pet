"""Visualize DeepLabV3 segmentation predictions for report and sanity checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from config import (
    DATA_DIR,
    FIGURE_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESULT_DIR,
    SEED,
    SEGMENTATION_IMAGE_SIZE,
    SEG_NUM_CLASSES,
    ensure_output_dirs,
    set_seed,
)
from data_loaders import OxfordPetSegmentationDataset
from models_seg import build_segmentation_model, get_segmentation_checkpoint_path, normalize_segmentation_model_name
from train_seg import extract_main_logits
from segmentation_utils import logits_to_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create segmentation prediction comparison figures.")
    parser.add_argument("--model", default="deeplabv3_mobilenet", choices=["deeplabv3_mobilenet"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--img-size", type=int, default=SEGMENTATION_IMAGE_SIZE)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--figure-out", type=Path, default=FIGURE_DIR / "fig_seg_predictions.png")
    parser.add_argument("--analysis-out", type=Path, default=RESULT_DIR / "seg_failure_analysis_step10.md")
    parser.add_argument("--selection-out", type=Path, default=RESULT_DIR / "seg_visualized_samples.json")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise TypeError(f"Unsupported checkpoint format at {path}")


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["dataset_index"] = int(row["dataset_index"])
        for key in ("pixel_accuracy", "background_iou", "foreground_iou", "miou", "dice"):
            row[key] = float(row[key])
    return rows


def choose_samples(rows: list[dict[str, Any]], count_per_group: int = 3) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["miou"]))
    failures = ordered[:count_per_group]
    successes = list(reversed(ordered[-count_per_group:]))
    return successes + failures


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1)


@torch.no_grad()
def predict_one(model: torch.nn.Module, image: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    image_batch = image.unsqueeze(0).to(device)
    mask_batch = mask.unsqueeze(0).to(device)
    output = model(image_batch)
    logits = extract_main_logits(output, mask_batch)
    return logits_to_prediction(logits).squeeze(0).cpu()


def plot_samples(
    selected: list[dict[str, Any]],
    dataset: OxfordPetSegmentationDataset,
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected), 4, figsize=(11, 2.6 * len(selected)), dpi=180)

    index_to_position = {int(source_index): pos for pos, source_index in enumerate(dataset.indices)}
    for row_idx, row in enumerate(selected):
        dataset_index = int(row["dataset_index"])
        item_position = index_to_position[dataset_index]
        image, mask = dataset[item_position]
        pred = predict_one(model, image, mask, device)
        display_image = denormalize(image).permute(1, 2, 0).numpy()

        axes[row_idx, 0].imshow(display_image)
        axes[row_idx, 0].set_title(f"image #{dataset_index}", fontsize=8)
        axes[row_idx, 1].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("ground truth", fontsize=8)
        axes[row_idx, 2].imshow(pred.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("prediction", fontsize=8)
        axes[row_idx, 3].imshow(display_image)
        axes[row_idx, 3].imshow(pred.numpy(), cmap="Reds", alpha=0.38, vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f"overlay mIoU={float(row['miou']):.3f}", fontsize=8)
        for col in range(4):
            axes[row_idx, col].axis("off")

    fig.suptitle("DeepLabV3-MobileNetV3 segmentation predictions on test split", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path)
    plt.close(fig)


def write_selection(path: Path, selected: list[dict[str, Any]], figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "figure": str(figure_path),
        "selection": [
            {
                "dataset_index": row["dataset_index"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "pixel_accuracy": row["pixel_accuracy"],
                "background_iou": row["background_iou"],
                "foreground_iou": row["foreground_iou"],
                "miou": row["miou"],
                "dice": row["dice"],
            }
            for row in selected
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_analysis(path: Path, selected: list[dict[str, Any]], figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = selected[3:]
    lines = [
        "# Step 10 segmentation failure analysis",
        "",
        f"Visualization source: `{figure_path}`.",
        "",
        "The global test metrics are high, and the high-mIoU examples show that the model generally captures the pet foreground rather than collapsing to all-background or all-foreground predictions.",
        "",
        "The lowest-mIoU samples indicate failure modes that should be discussed in the report:",
    ]
    for row in failures:
        lines.append(
            f"- Dataset index {row['dataset_index']}: mIoU={float(row['miou']):.4f}, "
            f"Dice={float(row['dice']):.4f}. This sample should be treated as a hard case for boundary/foreground localization and checked visually in the generated comparison figure."
        )
    lines.extend(
        [
            "",
            "For the final report, these cases can be grouped under boundary ambiguity, complex background, unusual crop/scale, or foreground-background confusion after visual inspection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_name = normalize_segmentation_model_name(args.model)
    ensure_output_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or get_segmentation_checkpoint_path(model_name)
    predictions_csv = args.predictions_csv or RESULT_DIR / f"seg_predictions_{model_name}_{args.split}.csv"

    checkpoint = load_checkpoint(checkpoint_path, device)
    num_classes = int(checkpoint.get("num_classes") or SEG_NUM_CLASSES)
    if num_classes != SEG_NUM_CLASSES:
        raise ValueError(f"Expected {SEG_NUM_CLASSES} classes, checkpoint has {num_classes}.")

    dataset = OxfordPetSegmentationDataset(
        root=args.data_root,
        split=args.split,
        img_size=args.img_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        binary_mask=True,
    )
    model = build_segmentation_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    selected = choose_samples(load_prediction_rows(predictions_csv))
    plot_samples(selected, dataset, model, device, args.figure_out)
    write_selection(args.selection_out, selected, args.figure_out)
    write_analysis(args.analysis_out, selected, args.figure_out)

    print(f"figure: {args.figure_out}")
    print(f"selection: {args.selection_out}")
    print(f"analysis: {args.analysis_out}")


if __name__ == "__main__":
    main()
