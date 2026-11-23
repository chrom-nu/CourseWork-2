from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    prediction_id: str = Field(..., description="UUID of the existing prediction")
    correct_label: Literal["healthy", "sick"] = Field(..., description="User verified label: healthy or sick")


class FeedbackResponse(BaseModel):
    status: str = "ok"
    feedback_id: str
    prediction_id: str
    correct_label: str
    created_at: datetime
