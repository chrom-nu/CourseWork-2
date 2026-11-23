from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.models.prediction import Prediction


class PredictionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        prediction_id: str,
        image_path: str,
        label: str,
        prob_healthy: float,
        prob_sick: float,
        model_version: str
    ) -> Prediction:
        prediction = Prediction(
            id=prediction_id,
            image_path=image_path,
            label=label,
            prob_healthy=prob_healthy,
            prob_sick=prob_sick,
            model_version=model_version,
        )
        self.session.add(prediction)
        await self.session.commit()
        await self.session.refresh(prediction)
        return prediction

    async def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
        stmt = select(Prediction).where(Prediction.id == prediction_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_recent(self, limit: int = 50) -> List[Prediction]:
        stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
