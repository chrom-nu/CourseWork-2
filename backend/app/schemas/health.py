from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    version: str
