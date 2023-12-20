from pydantic import BaseModel
from typing import Optional

class FineTuneModel(BaseModel):
    api_key: Optional[str] = ""
    data_path: Optional[str] = ""
    model: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.3
    max_retries: Optional[int] = 5
