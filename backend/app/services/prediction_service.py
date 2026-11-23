import uuid
from typing import Optional
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.image_service import ImageService
from backend.app.services.model_service import ModelService
from backend.app.services.gradcam_service import GradCAMService
from backend.app.repositories.prediction_repo import PredictionRepository
from backend.app.schemas.prediction import PredictionResponse
from shared.transforms import preprocess_image


class PredictionService:
    def __init__(
        self,
        model_service: ModelService,
        image_service: ImageService,
        gradcam_service: GradCAMService,
    ):
        self.model_service = model_service
        self.image_service = image_service
        self.gradcam_service = gradcam_service

    async def predict_and_store(
        self,
        file: UploadFile,
        session: AsyncSession,
        include_heatmap: bool = True,
    ) -> PredictionResponse:
        # 1. Validate and save image to disk / volume
        pil_img, saved_path, _ = await self.image_service.validate_and_save(file)

        # 2. Preprocess image using exact evaluation transforms
        img_tensor = preprocess_image(pil_img, img_size=self.model_service.img_size)

        # 3. Non-blocking model inference
        predicted_label, predicted_idx, probabilities, _ = await self.model_service.predict(img_tensor)

        # 4. Non-blocking Grad-CAM heatmap generation (if requested)
        heatmap_base64: Optional[str] = None
        if include_heatmap and self.model_service.model is not None:
            heatmap_base64, _ = await self.gradcam_service.generate_heatmap(
                model=self.model_service.model,
                input_tensor=img_tensor,
                original_image=pil_img,
                target_class_idx=predicted_idx,
            )

        # 5. Persist prediction in database
        prediction_id = str(uuid.uuid4())
        repo = PredictionRepository(session)
        created_prediction = await repo.create(
            prediction_id=prediction_id,
            image_path=saved_path,
            label=predicted_label,
            prob_healthy=probabilities["healthy"],
            prob_sick=probabilities["sick"],
            model_version=self.model_service.model_version,
        )

        return PredictionResponse(
            prediction_id=created_prediction.id,
            label=created_prediction.label,
            probabilities=probabilities,
            heatmap=heatmap_base64,
            model_version=created_prediction.model_version,
            created_at=created_prediction.created_at,
        )
