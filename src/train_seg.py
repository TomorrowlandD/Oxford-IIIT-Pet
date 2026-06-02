"""Train DeepLabV3-MobileNetV3 for Oxford-IIIT Pet foreground segmentation."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DATA_DIR,
    SEED,
    SEGMENTATION_BATCH_SIZE,
    SEGMENTATION_IMAGE_SIZE,
    SEG_NUM_CLASSES,
    ensure_output_dirs,
    set_seed,
)
from data_loaders import build_segmentation_dataloader
from models_seg import (
    build_segmentation_model,
    get_segmentation_checkpoint_path,
    get_segmentation_log_path,
    normalize_segmentation_model_name,
)
from segmentation_utils import build_binary_segmentation_loss, logits_to_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary foreground segmentation model.")
    parser.add_argument("--model", default="deeplabv3_mobilenet", choices=["deeplabv3_mobilenet"])
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--img-size", type=int, default=SEGMENTATION_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=SEGMENTATION_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument(
        "--binary-mask",
        dest="binary_mask",
        action="store_true",
        default=True,
        help="Convert trimap to project scheme-B binary masks. Enabled by default.",
    )
    parser.add_argument("--aux-loss-weight", type=float, default=0.4)
    return parser.parse_args()


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize segmentation logits to target spatial size when needed."""
    target_size = target.shape[-2:]
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def extract_main_logits(output: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Extract and spatially align the main logits from a TorchVision segmentation output."""
    if isinstance(output, dict):
        if "out" not in output:
            raise KeyError("TorchVision segmentation output is missing key 'out'.")
        logits = output["out"]
    else:
        logits = output
    return resize_logits_to_target(logits, target)


def compute_segmentation_loss(
    output: torch.Tensor | dict[str, torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    aux_loss_weight: float,
) -> torch.Tensor:
    """Compute main loss and optional TorchVision auxiliary-head loss."""
    if not isinstance(output, dict):
        logits = resize_logits_to_target(output, target)
        return criterion(logits, target)

    main_logits = resize_logits_to_target(output["out"], target)
    loss = criterion(main_logits, target)
    aux_logits = output.get("aux")
    if aux_logits is not None and aux_loss_weight > 0:
        aux_logits = resize_logits_to_target(aux_logits, target)
        loss = loss + aux_loss_weight * criterion(aux_logits, target)
    return loss


class SegmentationMetricAccumulator:
    """Global pixel accumulator for binary segmentation metrics."""

    def __init__(self) -> None:
        self.total_pixels = 0
        self.correct_pixels = 0
        self.intersections = [0, 0]
        self.unions = [0, 0]
        self.foreground_intersection = 0
        self.foreground_pred_total = 0
        self.foreground_target_total = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().long().reshape(-1).cpu()
        target = target.detach().long().reshape(-1).cpu()

        pred_values = set(int(value) for value in torch.unique(prediction).tolist())
        target_values = set(int(value) for value in torch.unique(target).tolist())
        if not pred_values.issubset({0, 1}):
            raise ValueError(f"Prediction must contain only 0/1 labels, got {sorted(pred_values)}")
        if not target_values.issubset({0, 1}):
            raise ValueError(f"Target must contain only 0/1 labels, got {sorted(target_values)}")

        self.total_pixels += int(target.numel())
        self.correct_pixels += int((prediction == target).sum().item())

        for class_id in (0, 1):
            pred_class = prediction == class_id
            target_class = target == class_id
            self.intersections[class_id] += int(torch.logical_and(pred_class, target_class).sum().item())
            self.unions[class_id] += int(torch.logical_or(pred_class, target_class).sum().item())

        foreground_pred = prediction == 1
        foreground_target = target == 1
        self.foreground_intersection += int(
            torch.logical_and(foreground_pred, foreground_target).sum().item()
        )
        self.foreground_pred_total += int(foreground_pred.sum().item())
        self.foreground_target_total += int(foreground_target.sum().item())

    def compute(self) -> dict[str, float | int]:
        if self.total_pixels == 0:
            raise ValueError("Cannot compute metrics without any pixels.")

        pixel_accuracy = self.correct_pixels / self.total_pixels
        background_iou = (
            1.0 if self.unions[0] == 0 else self.intersections[0] / self.unions[0]
        )
        foreground_iou = (
            1.0 if self.unions[1] == 0 else self.intersections[1] / self.unions[1]
        )
        foreground_total = self.foreground_pred_total + self.foreground_target_total
        dice = (
            1.0
            if foreground_total == 0
            else (2 * self.foreground_intersection) / foreground_total
        )

        return {
            "pixel_accuracy": float(pixel_accuracy),
            "background_iou": float(background_iou),
            "foreground_iou": float(foreground_iou),
            "miou": float((background_iou + foreground_iou) / 2),
            "dice": float(dice),
            "valid_pixels": self.total_pixels,
        }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    aux_loss_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(images)
                loss = compute_segmentation_loss(output, masks, criterion, aux_loss_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = compute_segmentation_loss(output, masks, criterion, aux_loss_weight)
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
    aux_loss_weight: float,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    accumulator = SegmentationMetricAccumulator()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        output = model(images)
        loss = compute_segmentation_loss(output, masks, criterion, aux_loss_weight)
        logits = extract_main_logits(output, masks)
        prediction = logits_to_prediction(logits)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        accumulator.update(prediction, masks)

    val_loss = total_loss / max(total_samples, 1)
    return val_loss, accumulator.compute()


def write_log_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_pixel_accuracy",
                "val_background_iou",
                "val_foreground_iou",
                "val_miou",
                "val_dice",
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
                f"{row['val_pixel_accuracy']:.6f}",
                f"{row['val_background_iou']:.6f}",
                f"{row['val_foreground_iou']:.6f}",
                f"{row['val_miou']:.6f}",
                f"{row['val_dice']:.6f}",
                f"{row['lr']:.8f}",
                f"{row['epoch_seconds']:.2f}",
            ]
        )


def main() -> None:
    args = parse_args()
    model_name = normalize_segmentation_model_name(args.model)

    if not args.binary_mask:
        raise ValueError("Two-class segmentation training requires --binary-mask.")

    ensure_output_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = build_segmentation_dataloader(
        split="train",
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        binary_mask=args.binary_mask,
    )
    val_loader = build_segmentation_dataloader(
        split="val",
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        binary_mask=args.binary_mask,
    )

    model = build_segmentation_model(
        model_name,
        num_classes=SEG_NUM_CLASSES,
        pretrained=args.pretrained,
    ).to(device)
    criterion = build_binary_segmentation_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    log_path = get_segmentation_log_path(model_name)
    checkpoint_path = get_segmentation_checkpoint_path(model_name)
    write_log_header(log_path)

    print(f"model: {model_name}")
    print(f"device: {device}")
    print(f"train_size: {len(train_loader.dataset)}")
    print(f"val_size: {len(val_loader.dataset)}")
    print(f"image_size: {args.img_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"binary_mask: {args.binary_mask}")
    print(f"log: {log_path}")
    print(f"checkpoint: {checkpoint_path}")

    best_miou = -1.0
    best_dice = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.aux_loss_weight,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.aux_loss_weight,
        )
        epoch_seconds = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_background_iou": val_metrics["background_iou"],
            "val_foreground_iou": val_metrics["foreground_iou"],
            "val_miou": val_metrics["miou"],
            "val_dice": val_metrics["dice"],
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        append_log_row(log_path, row)

        val_miou = float(val_metrics["miou"])
        val_dice = float(val_metrics["dice"])
        improved = (val_miou > best_miou) or (val_miou == best_miou and val_dice > best_dice)
        if improved:
            best_miou = val_miou
            best_dice = val_dice
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_name,
                    "img_size": args.img_size,
                    "num_classes": SEG_NUM_CLASSES,
                    "pretrained": args.pretrained,
                    "binary_mask": args.binary_mask,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                    "val_background_iou": val_metrics["background_iou"],
                    "val_foreground_iou": val_metrics["foreground_iou"],
                    "val_miou": val_metrics["miou"],
                    "val_dice": val_metrics["dice"],
                },
                checkpoint_path,
            )

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_pa={float(val_metrics['pixel_accuracy']):.4f} "
            f"val_miou={val_miou:.4f} "
            f"val_dice={val_dice:.4f} "
            f"time={epoch_seconds:.1f}s "
            f"{'saved' if improved else ''}"
        )

    print(
        f"best_epoch={best_epoch}, "
        f"best_val_miou={best_miou:.6f}, "
        f"best_val_dice={best_dice:.6f}"
    )


if __name__ == "__main__":
    main()
