from PIL import Image
import torch
from shared.transforms import (
    get_transforms as shared_get_transforms,
    get_eval_transforms,
    preprocess_image,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from ml.src.dataset import get_transforms as ml_get_transforms


def test_preprocessing_consistency():
    img_size = 224
    shared_tf = get_eval_transforms(img_size)
    ml_tf = ml_get_transforms("val", img_size)

    # Check identical transform steps and parameters
    assert len(shared_tf.transforms) == len(ml_tf.transforms)
    for t1, t2 in zip(shared_tf.transforms, ml_tf.transforms):
        assert type(t1) is type(t2)
        if hasattr(t1, "size"):
            assert t1.size == t2.size


def test_preprocess_image_tensor():
    pil_img = Image.new("RGB", (300, 400), color=(120, 200, 80))
    tensor = preprocess_image(pil_img, img_size=224)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32

    # Verify channels normalized
    assert tensor.min() < 0.0 or tensor.max() > 1.0  # normalized with ImageNet stats
