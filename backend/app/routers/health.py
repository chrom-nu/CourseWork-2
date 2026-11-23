from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.app.database import get_db
from backend.app.schemas.health import HealthResponse
from backend.app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def check_health(db: AsyncSession = Depends(get_db)):
    from backend.app.main import model_service

    model_loaded = model_service.is_loaded if model_service else False
    db_connected = False

    try:
        await db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    is_healthy = model_loaded and db_connected
    overall_status = "ok" if is_healthy else "degraded"
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall_status,
            "model_loaded": model_loaded,
            "database_connected": db_connected,
            "version": settings.VERSION,
        }
    )
