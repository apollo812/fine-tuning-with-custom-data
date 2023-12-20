from pydantic import BaseModel
from typing import Optional

class ChattingModel(BaseModel):
    api_key: Optional[str] = ""
    data_path: Optional[str] = ""
    model: Optional[str] = ""
    temperature: Optional[float] = 0.3
    question: Optional[str] = ""
