import hashlib
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def calculate_sha256(file_path, chunk_size=1024 * 1024):
    """Calculate a SHA-256 file hash."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def collect_image_geometry(dataframe):
    """Collect width, height, and aspect ratio for images in a DataFrame."""
    records = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), leave=False):
        with Image.open(row["path"]) as img:
            width, height = img.size

        records.append(
            {
                "class": row["class"],
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
            }
        )

    return pd.DataFrame(records)


def estimate_intensity_stats(dataframe, sample_size=None, seed=42):
    """Collect per-image grayscale mean and standard deviation."""
    if sample_size is not None:
        dataframe = dataframe.sample(
            n=min(sample_size, len(dataframe)),
            random_state=seed,
        )

    records = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), leave=False):
        with Image.open(row["path"]) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

        records.append(
            {
                "class": row["class"],
                "mean_intensity": float(arr.mean()),
                "std_intensity": float(arr.std()),
            }
        )

    return pd.DataFrame(records)
