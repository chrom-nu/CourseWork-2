from typing import List
from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    input_size: List[int]
    classes: List[str]
    temperature: float
    device: str
