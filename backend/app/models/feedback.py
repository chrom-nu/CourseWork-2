import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from backend.app.database import Base


def get_utc_now():
    return datetime.now(timezone.utc)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    correct_label = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False, index=True)

    prediction = relationship("Prediction", back_populates="feedbacks")

    def __repr__(self):
        return f"<Feedback(id={self.id}, prediction_id={self.prediction_id}, correct_label={self.correct_label})>"
