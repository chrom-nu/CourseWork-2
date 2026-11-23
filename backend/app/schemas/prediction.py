from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class Probabilities(BaseModel):
    healthy: float = Field(..., ge=0.0, le=1.0)
    sick: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    prediction_id: str
    label: str
    probabilities: Dict[str, float]
    heatmap: Optional[str] = None
    model_version: str
    created_at: datetime
