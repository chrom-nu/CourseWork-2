"""create predictions and feedback tables

Revision ID: 001_initial_tables
Revises: 
Create Date: 2026-11-23 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_tables'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Predictions Table
    op.create_table(
        'predictions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('image_path', sa.String(length=512), nullable=False),
        sa.Column('label', sa.String(length=32), nullable=False),
        sa.Column('prob_healthy', sa.Float(), nullable=False),
        sa.Column('prob_sick', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_predictions_id', 'predictions', ['id'], unique=False)
    op.create_index('ix_predictions_created_at', 'predictions', ['created_at'], unique=False)
    op.create_index('ix_predictions_label', 'predictions', ['label'], unique=False)

    # 2. Feedback Table
    op.create_table(
        'feedback',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('prediction_id', sa.String(length=36), nullable=False),
        sa.Column('correct_label', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_feedback_id', 'feedback', ['id'], unique=False)
    op.create_index('ix_feedback_prediction_id', 'feedback', ['prediction_id'], unique=False)
    op.create_index('ix_feedback_created_at', 'feedback', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_feedback_created_at', table_name='feedback')
    op.drop_index('ix_feedback_prediction_id', table_name='feedback')
    op.drop_index('ix_feedback_id', table_name='feedback')
    op.drop_table('feedback')

    op.drop_index('ix_predictions_label', table_name='predictions')
    op.drop_index('ix_predictions_created_at', table_name='predictions')
    op.drop_index('ix_predictions_id', table_name='predictions')
    op.drop_table('predictions')
