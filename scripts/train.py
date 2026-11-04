"""Reproducible training, validation, logging, checkpointing and resume."""

from dataclasses import dataclass, asdict
from pathlib import Path
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .model import build_b0, set_trainable_stage


@dataclass(frozen=True)
class TrainConfig:
    """Store the hyperparameters used by one EfficientNet-B0 experiment."""
    name: str
    stage: str
    backbone_lr: float
    head_lr: float
    label_smoothing: float = 0.05
    weight_decay: float = 1e-4
    image_size: int = 224
    seed: int = 42


def set_seed(seed=42):
    """Set Python, NumPy and PyTorch random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(payload, path):
    """Atomically write a checkpoint to avoid partial files after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def paths_for_run(name, checkpoint_dir, report_dir):
    """Get isolated best, last and training-history paths for an experiment."""
    return {
        "best": Path(checkpoint_dir) / f"{name}_best.pt",
        "last": Path(checkpoint_dir) / f"{name}_last.pt",
        "history": Path(report_dir) / f"{name}_history.csv",}


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None,
              freeze_features=False):
    """Return mean loss and top-1 accuracy for a training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    if training and freeze_features:
        # Frozen BatchNorm buffers should not change in a classifier-only run.
        model.features.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    amp = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader, desc="Train" if training else "Valid",
                                    leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            count = len(targets)
            total += count
            total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            total_loss += loss.detach().item() * count
    return total_loss / total, total_correct / total


def run_experiment(config, train_loader, val_loader, class_names, split_id,
                   checkpoint_dir, report_dir, device, target_epochs,
                   patience=3, pretrained_weight_path=None):
    """Train or resume one EfficientNet-B0 experiment.

    Reuse the last checkpoint when available, validate its configuration
    and data split, and continue until target_epochs or early stopping.
    Save the best checkpoint, latest training state, and epoch history.

    Returns:
        A summary containing the experiment name, best validation
        accuracy and epoch, completed epochs, and output file paths.
    """
    checkpoint_dir = Path(checkpoint_dir)
    report_dir = Path(report_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = paths_for_run(config.name, checkpoint_dir, report_dir)
    serialized_config = asdict(config)
    resume = paths["last"].is_file()
    if resume:
        checkpoint = torch.load(paths["last"], map_location="cpu", weights_only=False)
        if (checkpoint["config"] != serialized_config or
                checkpoint["split_id"] != split_id or
                checkpoint["class_names"] != list(class_names)):
            raise ValueError(f"Existing checkpoint {config.name} has different data/config")
        model = build_b0(len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        set_seed(config.seed)
        model = build_b0(len(class_names), pretrained=True,
                         weight_path=pretrained_weight_path)

    set_trainable_stage(model, config.stage)
    model = model.to(device)
    classifier_params = list(model.classifier.parameters())
    groups = [{"params": classifier_params, "lr": config.head_lr}]
    if config.stage == "full":
        groups.insert(0, {"params": model.features.parameters(),
                          "lr": config.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    start_epoch, best_accuracy, best_epoch, stale, history, stopped_early = (
        0, -1.0, 0, 0, [], False)
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        stopped_early = bool(checkpoint.get("stopped_early", False))
        print(f"{config.name}: checkpoint found, completed epoch {start_epoch}")

    if not stopped_early and start_epoch < target_epochs:
        for epoch in range(start_epoch, target_epochs):
            started = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, device, criterion, optimizer, scaler,
                freeze_features=config.stage == "head",)
            val_loss, val_acc = run_epoch(
                model, val_loader, device, criterion)
            scheduler.step(val_acc)
            improved = val_acc > best_accuracy
            if improved:
                best_accuracy, best_epoch, stale = val_acc, epoch + 1, 0
            else:
                stale += 1
            duration = (time.time() - started) / 60
            history.append({
                "epoch": epoch + 1, "train_loss": train_loss,
                "train_accuracy": train_acc, "val_loss": val_loss,
                "val_accuracy": val_acc, "minutes": round(duration, 2),})
            print(
                f"{config.name} | Epoch {epoch+1}/{target_epochs} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {duration:.1f} min")
            if improved:
                best_state = {
                    "architecture": "efficientnet_b0",
                    "model_state_dict": model.state_dict(),
                    "image_size": config.image_size,
                    "class_names": list(class_names),
                    "split_id": split_id,
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "config": serialized_config,}
                save_checkpoint(best_state, paths["best"])
                print(f"  Best saved: {paths['best'].name} ({best_accuracy:.4%})")
            stopped_early = patience is not None and stale >= patience
            last_state = {
                "architecture": "efficientnet_b0",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": serialized_config,
                "class_names": list(class_names),
                "split_id": split_id,
                "epoch": epoch + 1,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "stale_epochs": stale,
                "stopped_early": stopped_early,
                "history": history,}
            save_checkpoint(last_state, paths["last"])
            pd.DataFrame(history).to_csv(paths["history"], index=False)
            if stopped_early:
                print(f"Early stopping: no improvement for {patience} epochs")
                break
    else:
        print(f"{config.name}: no epochs needed; best={best_accuracy:.4%}")

    if not paths["best"].is_file():
        raise FileNotFoundError(f"No best checkpoint saved for {config.name}")
    return {
        "name": config.name,
        "best_accuracy": float(best_accuracy),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "best_path": paths["best"],
        "history_path": paths["history"],}
