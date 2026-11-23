import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Plant Disease Detection API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""

    # Model settings
    MODEL_CHECKPOINT_PATH: str = "ml/runs/resnet34_weighted_seed42/best_model.pt"
    MODEL_ARCH: str = "resnet34"
    IMG_SIZE: int = 224
    NUM_CLASSES: int = 2
    DEVICE: str = "cpu"
    TEMPERATURE: float = 1.0

    # Image upload settings
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png"]
    UPLOAD_DIR: str = "backend/uploads"

    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/plant_disease_db"
    SYNC_DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/plant_disease_db"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
