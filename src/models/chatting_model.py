from pydantic import BaseModel
from typing import Optional

class ChattingModel(BaseModel):
    api_key: Optional[str] = ""
    data_path: str
    model_id: str
    temperature: Optional[float] = 0.3
    question: str
