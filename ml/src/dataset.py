import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

CLASS_TO_IDX = {"healthy": 0, "sick": 1}
IDX_TO_CLASS = {0: "healthy", 1: "sick"}

# ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(split: str, img_size: int = 224):
    """
    Returns torchvision transforms for a given split.
    - train: resize, random resized crop/crop, horizontal/vertical flip, color jitter, normalization.
    - val / test / unlabeled: resize to 1.15x, center crop to img_size, normalization.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        resize_dim = int(img_size * 1.14)
        return transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

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

if __name__ == "__main__":
    print("Testing PlantDataset and DataLoader...")
    data_dir = os.path.join("ml", "data")
    
    for split in ["train", "val", "test"]:
        loader = get_dataloader(root_dir=data_dir, split=split, batch_size=16, num_workers=2, img_size=224)
        batch_imgs, batch_lbls = next(iter(loader))
        print(f"[{split.upper()}] Total: {len(loader.dataset)}, Batch Images: {batch_imgs.shape}, Batch Labels: {batch_lbls.shape}")

    unlabeled_loader = get_dataloader(root_dir=data_dir, split="unlabeled", batch_size=16, num_workers=2, img_size=224)
    u_imgs, u_paths = next(iter(unlabeled_loader))
    print(f"[UNLABELED] Total: {len(unlabeled_loader.dataset)}, Batch Images: {u_imgs.shape}, Paths count: {len(u_paths)}")

    # Save visual augmentation grid
    grid_path = os.path.join("ml", "reports", "augmentation_grid.png")
    save_augmentation_grid(grid_path, data_dir=data_dir, img_size=224)
    print("Dataset & DataLoader test passed successfully!")
