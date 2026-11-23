import io
import base64
import asyncio
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_jet_colormap(gray_heatmap: np.ndarray) -> np.ndarray:
    """
    Applies standard Jet colormap without requiring matplotlib or OpenCV.
    Args:
        gray_heatmap: 2D float array with values in [0, 1]
    Returns:
        3D uint8 array of shape [H, W, 3] with RGB values in [0, 255]
    """
    x = gray_heatmap * 4.0
    r = np.clip(np.minimum(x - 1.5, 4.5 - x), 0.0, 1.0)
    g = np.clip(np.minimum(x - 0.5, 3.5 - x), 0.0, 1.0)
    b = np.clip(np.minimum(x + 0.5, 2.5 - x), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


class GradCAMService:
    def __init__(self, target_layer_name: str = "layer4"):
        self.target_layer_name = target_layer_name

    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """Finds the target convolutional layer in the model."""
        if hasattr(model, self.target_layer_name):
            return getattr(model, self.target_layer_name)
        raise ValueError(f"Target layer '{self.target_layer_name}' not found in model.")

    def _compute_gradcam_sync(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class_idx: int,
        alpha: float = 0.5,
    ) -> Tuple[str, np.ndarray]:
        """
        Computes Grad-CAM heatmap and overlays it onto the original image.
        Returns:
            base64_png_str (str): base64 encoded PNG
            heatmap_norm (np.ndarray): 2D float array [0, 1] of the heatmap
        """
        device = next(model.parameters()).device
        tensor = input_tensor.clone().to(device)
        tensor.requires_grad = True

        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = self._get_target_layer(model)
        h_forward = target_layer.register_forward_hook(forward_hook)
        h_backward = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Enable gradient calculation for Grad-CAM
            with torch.enable_grad():
                model.zero_grad()
                logits = model(tensor)
                score = logits[0, target_class_idx]
                score.backward(retain_graph=False)

            if not activations or not gradients:
                raise RuntimeError("Failed to capture activations or gradients during Grad-CAM.")

            act = activations[0]  # [1, C, H, W]
            grad = gradients[0]   # [1, C, H, W]

            # 1. Global Average Pooling over gradients -> alpha weights [1, C, 1, 1]
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)

            # 2. Weighted combination of activation maps
            cam = torch.sum(weights * act, dim=1, keepdim=True)  # [1, 1, H, W]

            # 3. Apply ReLU
            cam = F.relu(cam)

            # 4. Normalize to [0, 1]
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 1e-7:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            cam_np = cam.squeeze().detach().cpu().numpy()  # [H, W]

            # 5. Resize heatmap to original image dimensions (W, H)
            orig_w, orig_h = original_image.size
            cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8))
            cam_resized = cam_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            heatmap_resized = np.array(cam_resized).astype(np.float32) / 255.0

            # 6. Apply Jet colormap
            colored_heatmap_uint8 = apply_jet_colormap(heatmap_resized)

            # 7. Blend original image with heatmap
            orig_np = np.array(original_image.convert("RGB"))
            blended = (alpha * orig_np + (1.0 - alpha) * colored_heatmap_uint8).astype(np.uint8)
            blended_img = Image.fromarray(blended)

            # 8. Encode to PNG base64
            buffer = io.BytesIO()
            blended_img.save(buffer, format="PNG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return base64_str, heatmap_resized

        finally:
            h_forward.remove()
            h_backward.remove()

    async def generate_heatmap(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class_idx: int,
        alpha: float = 0.5,
    ) -> Tuple[str, np.ndarray]:
        """
        Asynchronously computes Grad-CAM heatmap without blocking the event loop.
        """
        return await asyncio.to_thread(
            self._compute_gradcam_sync,
            model,
            input_tensor,
            original_image,
            target_class_idx,
            alpha,
        )
