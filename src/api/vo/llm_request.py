from pydantic import BaseModel

class LLMRequest(BaseModel):
    prompt: str
    top_k: str | None = None
    temperature: str | None = None
    top_p: str | None = None

class PretrainRequest(BaseModel):
    text: str