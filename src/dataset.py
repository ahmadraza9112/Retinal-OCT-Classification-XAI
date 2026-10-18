from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files(folder):
    """Return valid image files while excluding hidden/macOS metadata files."""
    folder = Path(folder)
    return [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    ]


def extract_patient_id(image_path):
    """Extract the randomized patient ID from a Kermany OCT filename."""
    image_path = Path(image_path)
    parts = image_path.stem.split("-")
    return parts[1] if len(parts) >= 3 else image_path.stem


class OCTDataset(Dataset):
    """DataFrame-backed grayscale OCT dataset."""

    def __init__(self, dataframe, transform=None, class_to_idx=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        label = self.class_to_idx[row["class"]]

        with Image.open(row["path"]) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)

        return image, label
