from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.prediction_service import PredictionService
from backend.app.services.image_service import ImageService
from backend.app.services.gradcam_service import GradCAMService

router = APIRouter(tags=["Inference"])

image_service = ImageService()
gradcam_service = GradCAMService()


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_image(
    file: UploadFile = File(..., description="JPEG or PNG image file (max 10MB)"),
    include_heatmap: bool = Form(True, description="Whether to include Grad-CAM heatmap"),
    db: AsyncSession = Depends(get_db),
):
    from backend.app.main import model_service

    prediction_service = PredictionService(
        model_service=model_service,
        image_service=image_service,
        gradcam_service=gradcam_service,
    )

    return await prediction_service.predict_and_store(
        file=file,
        session=db,
        include_heatmap=include_heatmap,
    )
