from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.models.feedback import Feedback


class FeedbackRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        feedback_id: str,
        prediction_id: str,
        correct_label: str
    ) -> Feedback:
        feedback = Feedback(
            id=feedback_id,
            prediction_id=prediction_id,
            correct_label=correct_label,
        )
        self.session.add(feedback)
        await self.session.commit()
        await self.session.refresh(feedback)
        return feedback

    async def get_by_id(self, feedback_id: str) -> Optional[Feedback]:
        stmt = select(Feedback).where(Feedback.id == feedback_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_prediction_id(self, prediction_id: str) -> List[Feedback]:
        stmt = select(Feedback).where(Feedback.prediction_id == prediction_id).order_by(Feedback.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
