import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

CLASS_TO_IDX = {"healthy": 0, "sick": 1}
IDX_TO_CLASS = {0: "healthy", 1: "sick"}

# ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, img_size: int = 224):
    """
    Returns torchvision transforms for a given split.
    - train: resize, random resized crop, horizontal/vertical flip, color jitter, normalization.
    - val / test / unlabeled / inference: resize to 1.14x, center crop to img_size, normalization.
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


def get_eval_transforms(img_size: int = 224):
    """Convenience alias for validation/test/inference transform."""
    return get_transforms("val", img_size=img_size)


def preprocess_image(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    """
    Preprocess a PIL image using exact eval transforms.
    Returns:
        Tensor of shape [1, 3, img_size, img_size]
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = get_eval_transforms(img_size=img_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def get_model(arch: str = "resnet34", num_classes: int = 2, pretrained: bool = False):
    """
    Initializes a ResNet model with customized fc layer.
    """
    if arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
