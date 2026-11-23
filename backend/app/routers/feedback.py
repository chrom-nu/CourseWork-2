import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.schemas.feedback import FeedbackCreate, FeedbackResponse
from backend.app.repositories.prediction_repo import PredictionRepository
from backend.app.repositories.feedback_repo import FeedbackRepository

router = APIRouter(tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def submit_feedback(
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    prediction_repo = PredictionRepository(db)
    feedback_repo = FeedbackRepository(db)

    # 1. Verify prediction exists
    prediction = await prediction_repo.get_by_id(payload.prediction_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PREDICTION_NOT_FOUND",
                "message": f"Prediction with ID '{payload.prediction_id}' does not exist.",
            }
        )

    # 2. Record feedback
    feedback_id = str(uuid.uuid4())
    feedback = await feedback_repo.create(
        feedback_id=feedback_id,
        prediction_id=payload.prediction_id,
        correct_label=payload.correct_label,
    )

    return FeedbackResponse(
        status="ok",
        feedback_id=feedback.id,
        prediction_id=feedback.prediction_id,
        correct_label=feedback.correct_label,
        created_at=feedback.created_at,
    )
