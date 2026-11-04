"""Validation inference, classification metrics and learning curves."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm

from .model import build_b0


def load_model(checkpoint_path, device, expected_classes=None,
               expected_split_id=None):
    """Load a trained B0 checkpoint and switch the model to evaluation mode.

    If supplied, verify the class order and validation split ID before
    loading the saved weights.

    Returns:
        The evaluation-ready model and checkpoint metadata.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("architecture", "efficientnet_b0") != "efficientnet_b0":
        raise ValueError("Checkpoint is not EfficientNet-B0")
    names = state.get("class_names")
    if not names:
        raise ValueError("Checkpoint missing class_names")
    if expected_classes is not None and list(names) != list(expected_classes):
        raise ValueError("Checkpoint class order differs from metadata")
    if (expected_split_id is not None and state.get("split_id") is not None and
            state["split_id"] != expected_split_id):
        raise ValueError("Checkpoint was trained with a different validation split")
    model = build_b0(len(names), pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    return model, state


def evaluate_model(model, loader, device, class_names, report_dir=None):
    """Predict validation labels and return accuracy, report and confusion matrix."""
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Validation inference"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
            predicted.extend(outputs.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    accuracy = float(np.mean(actual == predicted))
    report = pd.DataFrame(classification_report(
        actual, predicted, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0)).T
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_dir / "classification_report.csv")
        fig, ax = plt.subplots(figsize=(12, 10))
        ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
            ax=ax, xticks_rotation=90, colorbar=False)
        ax.set_title("Best B0 — Validation Confusion Matrix")
        fig.tight_layout()
        fig.savefig(report_dir / "confusion_matrix.png", dpi=130, bbox_inches="tight")
        plt.show()
    return accuracy, report, matrix


def plot_history(history_df, title="EfficientNet-B0", save_path=None):
    """Plot training/validation loss and accuracy from actual logged epoch metrics."""
    if history_df.empty:
        raise ValueError("Training history is empty")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, key, ylabel in zip(
        axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        axis.plot(history_df["epoch"], history_df[f"train_{key}"],
                  marker="o", label="Training")
        axis.plot(history_df["epoch"], history_df[f"val_{key}"],
                  marker="o", label="Validation")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=f"{title} — {ylabel}")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
