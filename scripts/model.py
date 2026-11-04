"""EfficientNet-B0 architecture and ImageNet weight loading."""

from pathlib import Path
import torch
from torch import nn
from torchvision import models

IMAGENET_FILENAME = "efficientnet_b0_rwightman-7f5810bc.pth"


def locate_pretrained_weights(input_root=Path("/kaggle/input"), custom_path=None):
    """Look for a local EfficientNet-B0 ImageNet weight file.

    Search the custom path, PyTorch cache, and Kaggle Input in that order.
    Return None when no local file is found.
    """
    if custom_path:
        path = Path(custom_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained weights missing: {path}")
        return path
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / IMAGENET_FILENAME
    if cache_path.is_file():
        return cache_path
    input_root = Path(input_root)
    if input_root.exists():
        matches = sorted(input_root.rglob(IMAGENET_FILENAME))
        if matches:
            return matches[0]
    return None


def build_b0(num_classes=16, pretrained=False, weight_path=None):
    """Build an EfficientNet-B0 model for document classification.

    Optionally initialize the backbone with ImageNet weights, then replace
    the classifier with dropout and a linear layer for num_classes.
    The classifier layout is kept compatible with the saved B0 checkpoints.
    """
    model = models.efficientnet_b0(weights=None)
    if pretrained:
        path = locate_pretrained_weights(custom_path=weight_path)
        if path:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded ImageNet B0 weights from {path}")
        else:
            try:
                state = models.EfficientNet_B0_Weights.DEFAULT.get_state_dict(
                    progress=True, check_hash=True)
                model.load_state_dict(state)
                print("Downloaded ImageNet B0 weights using torchvision")
            except Exception as exc:
                raise RuntimeError(
                    "ImageNet B0 weights unavailable. Enable Kaggle Internet or "
                    f"attach {IMAGENET_FILENAME} through Kaggle Add Input."
                ) from exc
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(p=0.30),
                                        nn.Linear(input_features, num_classes))
    return model


def set_trainable_stage(model, stage):
    """Train classifier only (baseline) or fine-tune all layers."""
    if stage not in ("head", "full"):
        raise ValueError("stage must be 'head' or 'full'")
    for param in model.features.parameters():
        param.requires_grad = stage == "full"
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model
