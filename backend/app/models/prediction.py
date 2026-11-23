import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.orm import relationship

from backend.app.database import Base


def get_utc_now():
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False, index=True)
    image_path = Column(String(512), nullable=False)
    label = Column(String(32), nullable=False, index=True)
    prob_healthy = Column(Float, nullable=False)
    prob_sick = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False)

    feedbacks = relationship("Feedback", back_populates="prediction", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Prediction(id={self.id}, label={self.label}, prob_sick={self.prob_sick:.4f})>"
