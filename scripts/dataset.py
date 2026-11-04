"""Metadata loading, TIFF preprocessing and PyTorch data loaders."""

from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


BAD_TIFF = "train/scientific_publication/2500126531_2500126536.tif"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_data_dir(explicit_path=None):
    """Locate the folder containing all three competition CSV files.

    Check an explicit path when provided; otherwise search Kaggle Input
    and common local data folders.

    Returns:
        Path to the folder containing the competition CSV files.
    """
    if explicit_path is not None:
        directory = Path(explicit_path)
        if all((directory / name).is_file() for name in
               ("train.csv", "test.csv", "sample_submission.csv")):
            return directory
        raise FileNotFoundError(f"Competition CSV files not found at {directory}")

    search_roots = [Path("/kaggle/input"), Path.cwd(), Path.cwd() / "data",
                    Path.cwd().parent / "data"]
    for root in search_roots:
        if not root.exists():
            continue
        if all((root / name).is_file() for name in
               ("train.csv", "test.csv", "sample_submission.csv")):
            return root
        if root == Path("/kaggle/input"):
            for match in sorted(root.rglob("train.csv")):
                folder = match.parent
                if (folder / "test.csv").is_file() and (folder / "sample_submission.csv").is_file():
                    return folder
    raise FileNotFoundError("Competition data not found. Add it through Kaggle > Add Input.")


def load_metadata(data_dir):
    """Read the competition metadata and prepare class information.

    Validate required columns and IDs, and exclude the known unreadable
    training TIFF.

    Returns:
        Train metadata, test metadata, submission template, ordered
        class names, and the number of excluded training rows.
    """
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    for col in ("id", "filename", "class_name", "label"):
        if col not in train_df:
            raise ValueError(f"Missing training column: {col}")
    for col in ("id", "filename"):
        if col not in test_df:
            raise ValueError(f"Missing test column: {col}")
    if not {"id", "label"}.issubset(sample.columns):
        raise ValueError("sample_submission.csv needs id and label columns")

    excluded = (train_df["filename"] == BAD_TIFF).sum()
    train_df = train_df.loc[train_df["filename"] != BAD_TIFF].reset_index(drop=True)
    label_map = (train_df[["label", "class_name"]].drop_duplicates()
                 .sort_values("label").reset_index(drop=True))
    class_names = label_map["class_name"].tolist()
    if label_map["label"].tolist() != list(range(len(class_names))):
        raise ValueError("Labels must be consecutive integers starting at zero")
    if train_df["id"].duplicated().any() or test_df["id"].duplicated().any():
        raise ValueError("Duplicate image IDs found")
    if not sample["id"].is_unique or set(sample["id"]) != set(test_df["id"]):
        raise ValueError("Submission IDs do not match test IDs")
    return train_df, test_df, sample, class_names, int(excluded)


def make_train_val_split(train_df, valid_size=0.20, seed=42):
    """Create a fixed stratified holdout used by all experiments."""
    train_part, val_part = train_test_split(
        train_df, test_size=valid_size, random_state=seed,
        stratify=train_df["label"],)
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def split_fingerprint(val_df):
    """Create a short hash from validation IDs to identify the data split."""
    ids = "\n".join(val_df["id"].astype(str).tolist())
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]


def find_image_root(data_dir, filename):
    """Find the root directory used by image paths in the CSV.

    Supports both direct paths such as train/... and nested Kaggle
    layouts such as train/train/....
    """
    data_dir = Path(data_dir)
    for root in (data_dir, data_dir / "train", data_dir / "test",
                 data_dir / "images"):
        if (root / filename).is_file():
            return root
    raise FileNotFoundError(f"Cannot locate image: {filename} below {data_dir}")


class ResizeWithPadding:
    """Resize document images while keeping their original aspect ratio."""

    def __init__(self, size=224):
        """Store the target width and height for the square output image."""
        self.size = int(size)

    def __call__(self, image):
        """Resize an image to fit the square and fill unused space with white."""
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = ImageOps.contain(image, (self.size, self.size), method=resampling)
        canvas = Image.new("RGB", (self.size, self.size), color=(255, 255, 255))
        canvas.paste(resized, ((self.size - resized.width) // 2,
                               (self.size - resized.height) // 2))
        return canvas


def build_transform(image_size=224, training=False):
    """Build the preprocessing pipeline for training or evaluation.

    Training uses light random augmentation. Validation and test images
    only receive resize, tensor conversion, and ImageNet normalization.
    """
    ops = [ResizeWithPadding(image_size)]
    if training:
        ops.extend([
            transforms.RandomApply([
                transforms.RandomRotation(5, fill=(255, 255, 255))], p=0.50),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0, translate=(0.02, 0.02),
                    scale=(0.97, 1.03), fill=(255, 255, 255))], p=0.30),
            transforms.ColorJitter(brightness=0.12, contrast=0.15),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))], p=0.10),])
    ops.extend([transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])
    return transforms.Compose(ops)


class DocumentDataset(Dataset):
    """Labeled and test-document dataset using TIFF paths from the CSV file."""

    def __init__(self, frame, image_root, transform, labeled=True):
        """Store the metadata, image directory, transform, and label mode."""
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        """Return the number of documents in the dataset."""
        return len(self.frame)

    def __getitem__(self, index):
        """Load and transform one TIFF image by its metadata index.

        Returns:
            An image tensor for test data, or an (image, label) pair
            when labeled=True.
        """
        row = self.frame.iloc[index]
        filename = self.image_root / row["filename"]
        try:
            with Image.open(filename) as img:
                img.seek(0)
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unable to read TIFF: {filename}") from exc
        tensor = self.transform(image)
        if self.labeled:
            return tensor, int(row["label"])
        return tensor


def make_loader(frame, root, image_size, batch_size=16, training=False,
                labeled=True, num_workers=0, pin_memory=False):
    """Create a data loader; num_workers=0 avoids notebook worker shutdown errors."""
    dataset = DocumentDataset(frame, root, build_transform(image_size, training), labeled)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)
