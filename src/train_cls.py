"""Train classification models on Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from config import (
    CLASSIFICATION_BATCH_SIZE,
    CLASSIFICATION_IMAGE_SIZE,
    DATA_DIR,
    NUM_CLASSES,
    SEED,
    ensure_output_dirs,
    set_seed,
)
from data_loaders import build_classification_dataloader
from models_cls import (
    build_classification_model,
    get_classification_checkpoint_path,
    get_classification_log_path,
    normalize_classification_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--img-size", type=int, default=CLASSIFICATION_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=CLASSIFICATION_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    val_loss = total_loss / max(total_samples, 1)
    accuracy = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return val_loss, accuracy, macro_f1


def write_log_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "lr",
                "epoch_seconds",
            ]
        )


def append_log_row(path: Path, row: dict[str, float | int]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                row["epoch"],
                f"{row['train_loss']:.6f}",
                f"{row['val_loss']:.6f}",
                f"{row['val_accuracy']:.6f}",
                f"{row['val_macro_f1']:.6f}",
                f"{row['lr']:.8f}",
                f"{row['epoch_seconds']:.2f}",
            ]
        )


def main() -> None:
    args = parse_args()
    model_name = normalize_classification_model_name(args.model)

    ensure_output_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = build_classification_dataloader(
        split="train",
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_classification_dataloader(
        split="val",
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_classification_model(
        model_name,
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    log_path = get_classification_log_path(model_name)
    checkpoint_path = get_classification_checkpoint_path(model_name)
    write_log_header(log_path)

    print(f"model: {model_name}")
    print(f"device: {device}")
    print(f"train_size: {len(train_loader.dataset)}")
    print(f"val_size: {len(val_loader.dataset)}")
    print(f"log: {log_path}")
    print(f"checkpoint: {checkpoint_path}")

    best_macro_f1 = -1.0
    best_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_accuracy, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        append_log_row(log_path, row)

        improved = (val_macro_f1 > best_macro_f1) or (
            val_macro_f1 == best_macro_f1 and val_accuracy > best_accuracy
        )
        if improved:
            best_macro_f1 = val_macro_f1
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_name,
                    "img_size": args.img_size,
                    "num_classes": NUM_CLASSES,
                    "pretrained": args.pretrained,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                },
                checkpoint_path,
            )

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"time={epoch_seconds:.1f}s "
            f"{'saved' if improved else ''}"
        )

    print(
        f"best_epoch={best_epoch}, "
        f"best_val_accuracy={best_accuracy:.6f}, "
        f"best_val_macro_f1={best_macro_f1:.6f}"
    )


if __name__ == "__main__":
    main()
