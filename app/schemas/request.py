from pydantic import BaseModel

class TextData(BaseModel):
    origin: str
    summary: str