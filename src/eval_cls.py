"""Evaluate classification checkpoints on Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import (
    CLASSIFICATION_BATCH_SIZE,
    CLASSIFICATION_IMAGE_SIZE,
    DATA_DIR,
    FIGURE_DIR,
    NUM_CLASSES,
    RESULT_DIR,
    SEED,
    ensure_output_dirs,
    set_seed,
)
from data_loaders import Split, build_classification_dataloader
from models_cls import (
    build_classification_model,
    get_classification_checkpoint_path,
    normalize_classification_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a classification checkpoint.")
    parser.add_argument("--model", default="efficientnet_b0", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=CLASSIFICATION_BATCH_SIZE)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument("--confusion-matrix-out", type=Path, default=None)
    parser.add_argument(
        "--write-canonical-confusion-matrix",
        action="store_true",
        help="Also write outputs/figures/fig_confusion_matrix.png.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise TypeError(f"Unsupported checkpoint format at {path}")


def _dataset_metadata(loader: torch.utils.data.DataLoader, start: int, count: int) -> list[dict[str, Any]]:
    dataset = loader.dataset
    metadata: list[dict[str, Any]] = []

    indices = getattr(dataset, "indices", None)
    source_dataset = getattr(dataset, "dataset", None)
    image_paths = getattr(source_dataset, "_images", None)

    for offset in range(count):
        item = start + offset
        source_index = int(indices[item]) if indices is not None else item
        image_path = str(Path(image_paths[source_index])) if image_paths is not None else ""
        metadata.append({"dataset_index": source_index, "image_path": image_path})

    return metadata


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    seen = 0

    for images, labels in loader:
        batch_size = images.size(0)
        batch_metadata = _dataset_metadata(loader, seen, batch_size)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        confidences, preds = probabilities.max(dim=1)
        topk_count = min(5, probabilities.size(1))
        topk_confidences, topk_indices = probabilities.topk(topk_count, dim=1)

        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        confidences_cpu = confidences.cpu().numpy()
        topk_confidences_cpu = topk_confidences.cpu().numpy()
        topk_indices_cpu = topk_indices.cpu().numpy()

        for row_idx in range(batch_size):
            true_label = int(labels_cpu[row_idx])
            pred_label = int(preds_cpu[row_idx])
            top5_labels = [int(value) for value in topk_indices_cpu[row_idx].tolist()]
            top5_conf = [float(value) for value in topk_confidences_cpu[row_idx].tolist()]
            prediction_rows.append(
                {
                    **batch_metadata[row_idx],
                    "true_label": true_label,
                    "true_class": class_names[true_label],
                    "pred_label": pred_label,
                    "pred_class": class_names[pred_label],
                    "confidence": float(confidences_cpu[row_idx]),
                    "correct": bool(true_label == pred_label),
                    "top5_labels": top5_labels,
                    "top5_classes": [class_names[label] for label in top5_labels],
                    "top5_confidences": top5_conf,
                }
            )

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.extend(int(value) for value in labels_cpu.tolist())
        all_preds.extend(int(value) for value in preds_cpu.tolist())
        seen += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, prediction_rows, np.asarray(all_targets), np.asarray(all_preds)


def build_metrics(
    *,
    model_name: str,
    checkpoint_path: Path,
    split: Split,
    img_size: int,
    batch_size: int,
    seed: int,
    val_ratio: float,
    data_root: Path,
    class_names: list[str],
    loss: float,
    targets: np.ndarray,
    preds: np.ndarray,
    prediction_rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, preds, labels=labels)
    top5_accuracy = float(np.mean([row["true_label"] in row["top5_labels"] for row in prediction_rows]))

    per_class = []
    for idx, class_name in enumerate(class_names):
        per_class.append(
            {
                "label": idx,
                "class_name": class_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    return {
        "dataset": "Oxford-IIIT Pet",
        "task": "37-class pet breed classification",
        "model": model_name,
        "split": split,
        "data_root": str(data_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_accuracy": checkpoint.get("val_accuracy"),
        "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
        "img_size": img_size,
        "batch_size": batch_size,
        "seed": seed,
        "val_ratio": val_ratio,
        "num_classes": len(class_names),
        "num_samples": int(targets.size),
        "loss": float(loss),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "top5_accuracy": top5_accuracy,
        "classes": class_names,
        "per_class": per_class,
        "classification_report": classification_report(
            targets,
            preds,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": matrix.tolist(),
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "dataset_index",
            "image_path",
            "true_label",
            "true_class",
            "pred_label",
            "pred_class",
            "confidence",
            "correct",
            "top5_labels",
            "top5_classes",
            "top5_confidences",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["top5_labels"] = json.dumps(row["top5_labels"], ensure_ascii=False)
            serialized["top5_classes"] = json.dumps(row["top5_classes"], ensure_ascii=False)
            serialized["top5_confidences"] = json.dumps(row["top5_confidences"], ensure_ascii=False)
            writer.writerow(serialized)


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, default=_json_default)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 10), dpi=180)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [str(index) for index in range(len(class_names))]
    ax.set_xticks(np.arange(len(class_names)), labels=tick_labels)
    ax.set_yticks(np.arange(len(class_names)), labels=tick_labels)
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title(title)

    threshold = matrix.max() * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value == 0:
                continue
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=4.5, color=color)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_name = normalize_classification_model_name(args.model)

    ensure_output_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    checkpoint_path = args.checkpoint or get_classification_checkpoint_path(model_name)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    img_size = int(args.img_size or checkpoint.get("img_size") or CLASSIFICATION_IMAGE_SIZE)
    num_classes = int(checkpoint.get("num_classes") or NUM_CLASSES)

    loader = build_classification_dataloader(
        split=args.split,
        root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    class_names = list(getattr(loader.dataset, "classes"))
    if len(class_names) != num_classes:
        raise ValueError(
            f"Checkpoint expects {num_classes} classes, but dataset exposes {len(class_names)} classes."
        )

    model = build_classification_model(
        model_name,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, prediction_rows, targets, preds = run_inference(model, loader, device, class_names)
    metrics = build_metrics(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        data_root=args.data_root,
        class_names=class_names,
        loss=loss,
        targets=targets,
        preds=preds,
        prediction_rows=prediction_rows,
        checkpoint=checkpoint,
    )

    metrics_path = args.metrics_out or RESULT_DIR / f"cls_metrics_{model_name}.json"
    predictions_path = args.predictions_out or RESULT_DIR / f"cls_predictions_{model_name}_{args.split}.csv"
    confusion_path = args.confusion_matrix_out or FIGURE_DIR / f"fig_confusion_matrix_{model_name}.png"

    write_metrics(metrics_path, metrics)
    write_predictions(predictions_path, prediction_rows)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(
        matrix,
        class_names,
        confusion_path,
        title=f"{model_name} confusion matrix on {args.split}",
    )
    if args.write_canonical_confusion_matrix:
        plot_confusion_matrix(
            matrix,
            class_names,
            FIGURE_DIR / "fig_confusion_matrix.png",
            title=f"{model_name} confusion matrix on {args.split}",
        )

    print(f"model: {model_name}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"split: {args.split}")
    print(f"samples: {metrics['num_samples']}")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")
    print(f"top5_accuracy: {metrics['top5_accuracy']:.6f}")
    print(f"metrics: {metrics_path}")
    print(f"predictions: {predictions_path}")
    print(f"confusion_matrix: {confusion_path}")


if __name__ == "__main__":
    main()
