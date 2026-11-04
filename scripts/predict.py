"""Validation inference, classification metrics and learning curves."""

from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm


def predict_labels(model, loader, device):
    """Predict integer class labels for test images in loader order."""
    labels = []
    model.eval()
    with torch.inference_mode():
        for images in tqdm(loader, desc="Test inference"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            labels.extend(logits.argmax(dim=1).cpu().tolist())
    return labels


def build_submission(test_df, sample_submission, predictions, output_path,
                     num_classes=16):
    """Create and save a Kaggle submission with IDs in template order.

    Check the prediction count, unique IDs, missing predictions, and
    valid class range before writing the CSV file.

    Returns:
        Submission DataFrame with id and label columns.
    """
    if len(predictions) != len(test_df):
        raise ValueError("Prediction count differs from test row count")
    pred_df = pd.DataFrame({"id": test_df["id"], "label": predictions})
    if pred_df["id"].duplicated().any():
        raise ValueError("Test metadata contains duplicate IDs")
    submission = sample_submission[["id"]].merge(
        pred_df, how="left", on="id", validate="one_to_one")
    if len(submission) != len(sample_submission) or submission["label"].isna().any():
        raise ValueError("Submission contains missing predictions")
    submission["label"] = submission["label"].astype(int)
    if not submission["label"].between(0, num_classes - 1).all():
        raise ValueError("Predicted labels are out of bounds")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
