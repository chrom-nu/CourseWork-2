import os
import asyncio
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models

from backend.app.config import settings
from shared.transforms import get_model, IDX_TO_CLASS, CLASS_TO_IDX


class ModelService:
    def __init__(
        self,
        checkpoint_path: str = settings.MODEL_CHECKPOINT_PATH,
        arch: str = settings.MODEL_ARCH,
        img_size: int = settings.IMG_SIZE,
        device_str: str = settings.DEVICE,
        temperature: float = settings.TEMPERATURE,
    ):
        self.checkpoint_path = checkpoint_path
        self.arch = arch
        self.img_size = img_size
        self.temperature = max(1e-4, temperature)
        self.device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        self.model: Optional[nn.Module] = None
        self.model_version: str = "unloaded"
        self.is_loaded: bool = False

    def load_model(self, allow_stub: bool = False):
        """
        Loads the PyTorch model from checkpoint into memory.
        If allow_stub is True and checkpoint is missing, initializes an unweighted stub (useful for tests/CI).
        """
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            # Extract config/arch
            config = checkpoint.get("config", {})
            self.arch = checkpoint.get("arch", config.get("arch", self.arch))
            self.img_size = config.get("img_size", self.img_size)

            # Initialize architecture
            self.model = get_model(arch=self.arch, num_classes=settings.NUM_CLASSES, pretrained=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Version name based on folder or path
            self.model_version = os.path.basename(os.path.dirname(self.checkpoint_path)) or "resnet34_baseline"
            self.is_loaded = True
            print(f"Model successfully loaded: version={self.model_version}, device={self.device}")
        elif allow_stub:
            print(f"Checkpoint not found at {self.checkpoint_path}. Initializing dummy stub model for CI/testing.")
            self.model = get_model(arch=self.arch, num_classes=settings.NUM_CLASSES, pretrained=False)
            self.model.to(self.device)
            self.model.eval()
            self.model_version = f"{self.arch}_stub"
            self.is_loaded = True
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

    def _predict_sync(self, image_tensor: torch.Tensor) -> Tuple[str, int, Dict[str, float], torch.Tensor]:
        """
        Synchronous forward pass with temperature scaling.
        Args:
            image_tensor: shape [1, 3, H, W]
        Returns:
            predicted_label (str), class_idx (int), probabilities_dict (Dict[str, float]), logits (Tensor)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot execute inference.")

        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=1).squeeze(0)

        prob_healthy = float(probs[CLASS_TO_IDX["healthy"]].item())
        prob_sick = float(probs[CLASS_TO_IDX["sick"]].item())
        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = IDX_TO_CLASS[predicted_idx]

        probabilities = {
            "healthy": round(prob_healthy, 4),
            "sick": round(prob_sick, 4),
        }

        return predicted_label, predicted_idx, probabilities, logits

    async def predict(self, image_tensor: torch.Tensor) -> Tuple[str, int, Dict[str, float], torch.Tensor]:
        """
        Asynchronous wrapper ensuring inference does not block the event loop.
        """
        return await asyncio.to_thread(self._predict_sync, image_tensor)
