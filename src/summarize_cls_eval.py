"""Summarize classification evaluation outputs and plot example cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from config import FIGURE_DIR, RESULT_DIR, ensure_output_dirs
from models_cls import SUPPORTED_CLASSIFICATION_MODELS, normalize_classification_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize classification test metrics.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_CLASSIFICATION_MODELS),
        help="Models to summarize. Defaults to all supported classification models.",
    )
    parser.add_argument("--main-model", default="efficientnet_b0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-correct", type=int, default=8)
    parser.add_argument("--num-errors", type=int, default=8)
    parser.add_argument("--summary-out", type=Path, default=RESULT_DIR / "cls_test_summary.csv")
    parser.add_argument("--examples-out", type=Path, default=RESULT_DIR / "cls_examples_efficientnet_b0.json")
    parser.add_argument("--correct-figure-out", type=Path, default=FIGURE_DIR / "fig_cls_correct_examples.png")
    parser.add_argument("--failure-figure-out", type=Path, default=FIGURE_DIR / "fig_failure_cases.png")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["dataset_index"] = int(row["dataset_index"])
            row["true_label"] = int(row["true_label"])
            row["pred_label"] = int(row["pred_label"])
            row["confidence"] = float(row["confidence"])
            row["correct"] = row["correct"].strip().lower() == "true"
            row["top5_labels"] = json.loads(row["top5_labels"])
            row["top5_classes"] = json.loads(row["top5_classes"])
            row["top5_confidences"] = json.loads(row["top5_confidences"])
            rows.append(row)
    return rows


def write_summary(models: list[str], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "model",
            "split",
            "num_samples",
            "checkpoint_epoch",
            "checkpoint_val_accuracy",
            "checkpoint_val_macro_f1",
            "test_loss",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "test_top5_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            metrics = read_json(RESULT_DIR / f"cls_metrics_{model}.json")
            writer.writerow(
                {
                    "model": model,
                    "split": metrics["split"],
                    "num_samples": metrics["num_samples"],
                    "checkpoint_epoch": metrics.get("checkpoint_epoch"),
                    "checkpoint_val_accuracy": f"{metrics.get('checkpoint_val_accuracy', 0):.6f}",
                    "checkpoint_val_macro_f1": f"{metrics.get('checkpoint_val_macro_f1', 0):.6f}",
                    "test_loss": f"{metrics['loss']:.6f}",
                    "test_accuracy": f"{metrics['accuracy']:.6f}",
                    "test_macro_f1": f"{metrics['macro_f1']:.6f}",
                    "test_weighted_f1": f"{metrics['weighted_f1']:.6f}",
                    "test_top5_accuracy": f"{metrics['top5_accuracy']:.6f}",
                }
            )


def write_per_class_csv(model: str) -> Path:
    metrics = read_json(RESULT_DIR / f"cls_metrics_{model}.json")
    output_path = RESULT_DIR / f"cls_per_class_metrics_{model}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["label", "class_name", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow(
                {
                    "label": row["label"],
                    "class_name": row["class_name"],
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                    "support": row["support"],
                }
            )
    return output_path


def select_examples(
    predictions: list[dict[str, Any]],
    num_correct: int,
    num_errors: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    correct = [row for row in predictions if row["correct"] and Path(row["image_path"]).exists()]
    errors = [row for row in predictions if not row["correct"] and Path(row["image_path"]).exists()]

    # Use confident correct cases and confident errors so the report examples are visually meaningful.
    correct = sorted(correct, key=lambda row: row["confidence"], reverse=True)[:num_correct]
    errors = sorted(errors, key=lambda row: row["confidence"], reverse=True)[:num_errors]
    return correct, errors


def _format_title(row: dict[str, Any], is_error: bool) -> str:
    if is_error:
        return (
            f"T: {row['true_class']}\n"
            f"P: {row['pred_class']} ({row['confidence']:.2f})"
        )
    return f"{row['true_class']}\nconf={row['confidence']:.2f}"


def plot_examples(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    is_error: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available to plot for {output_path}")

    cols = min(4, len(rows))
    rows_count = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.0, rows_count * 3.35), dpi=180)
    if not isinstance(axes, (list, tuple)):
        axes_array = [axes]
    else:
        axes_array = list(axes)
    axes_flat = list(getattr(axes, "flat", axes_array))

    for ax, row in zip(axes_flat, rows):
        with Image.open(row["image_path"]) as image_file:
            image = image_file.convert("RGB")
        ax.imshow(image)
        ax.set_title(_format_title(row, is_error=is_error), fontsize=8)
        ax.axis("off")

    for ax in axes_flat[len(rows):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def write_examples_json(
    path: Path,
    model: str,
    split: str,
    correct: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "split": split,
        "selection_rule": {
            "correct": "highest-confidence correct predictions with existing image files",
            "errors": "highest-confidence incorrect predictions with existing image files",
        },
        "correct_examples": correct,
        "error_examples": errors,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    models = [normalize_classification_model_name(model) for model in args.models]
    main_model = normalize_classification_model_name(args.main_model)

    write_summary(models, args.summary_out)
    per_class_paths = [write_per_class_csv(model) for model in models]

    predictions_path = RESULT_DIR / f"cls_predictions_{main_model}_{args.split}.csv"
    predictions = read_predictions(predictions_path)
    correct, errors = select_examples(predictions, args.num_correct, args.num_errors)

    examples_path = args.examples_out
    if examples_path == RESULT_DIR / "cls_examples_efficientnet_b0.json" and main_model != "efficientnet_b0":
        examples_path = RESULT_DIR / f"cls_examples_{main_model}.json"

    write_examples_json(examples_path, main_model, args.split, correct, errors)
    plot_examples(
        correct,
        args.correct_figure_out,
        title=f"{main_model} correct classification examples",
        is_error=False,
    )
    plot_examples(
        errors,
        args.failure_figure_out,
        title=f"{main_model} high-confidence classification errors",
        is_error=True,
    )

    print(f"summary: {args.summary_out}")
    for path in per_class_paths:
        print(f"per_class: {path}")
    print(f"examples: {examples_path}")
    print(f"correct_figure: {args.correct_figure_out}")
    print(f"failure_figure: {args.failure_figure_out}")
    print(f"correct_examples: {len(correct)}")
    print(f"error_examples: {len(errors)}")


if __name__ == "__main__":
    main()
