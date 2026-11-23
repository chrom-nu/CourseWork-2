from fastapi import APIRouter
from backend.app.schemas.model_info import ModelInfoResponse

router = APIRouter(tags=["Model"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    from backend.app.main import model_service

    return ModelInfoResponse(
        model_version=model_service.model_version,
        architecture=model_service.arch,
        input_size=[model_service.img_size, model_service.img_size],
        classes=["healthy", "sick"],
        temperature=model_service.temperature,
        device=str(model_service.device),
    )
