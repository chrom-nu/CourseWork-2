import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.transforms import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transforms,
    get_eval_transforms,
    preprocess_image,
)


class PlantDataset(Dataset):
    """
    Dataset for labeled splits (train, val, test).
    Labels: healthy = 0, sick = 1.
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 224, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform or get_transforms(split, img_size=img_size)
        self.samples = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(self.split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts or not ext:
                    fpath = os.path.join(cls_dir, fname)
                    self.samples.append((fpath, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class UnlabeledPlantDataset(Dataset):
    """
    Dataset for unlabeled images.
    Returns (image_tensor, image_path).
    """
    def __init__(self, root_dir: str, img_size: int = 224, transform=None):
        self.root_dir = root_dir
        self.unlabeled_dir = os.path.join(root_dir, "unlabeled")
        self.transform = transform or get_transforms("unlabeled", img_size=img_size)
        self.file_paths = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for root, _, files in os.walk(self.unlabeled_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts or not ext:
                    self.file_paths.append(os.path.join(root, fname))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def get_dataloader(
    root_dir: str = os.path.join("ml", "data"),
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
    shuffle: bool = None
):
    """
    Factory helper to create DataLoaders with appropriate worker and pin_memory settings.
    """
    if split == "unlabeled":
        ds = UnlabeledPlantDataset(root_dir=root_dir, img_size=img_size)
        is_shuffle = False if shuffle is None else shuffle
    else:
        ds = PlantDataset(root_dir=root_dir, split=split, img_size=img_size)
        is_shuffle = (split == "train") if shuffle is None else shuffle

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Invert ImageNet normalization for visualization.
    """
    t = tensor.clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = t * std + mean
    return torch.clamp(t, 0.0, 1.0)


def save_augmentation_grid(output_path: str, data_dir: str = os.path.join("ml", "data"), img_size: int = 224):
    """
    Loads 16 train images after augmentations, denormalizes and saves a 4x4 visual grid.
    """
    loader = get_dataloader(root_dir=data_dir, split="train", batch_size=16, shuffle=True, num_workers=0, img_size=img_size)
    images, labels = next(iter(loader))

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        label_str = IDX_TO_CLASS[labels[idx].item()]
        ax.imshow(img)
        ax.set_title(f"{label_str} ({labels[idx].item()})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved augmentation grid to: {output_path}")
