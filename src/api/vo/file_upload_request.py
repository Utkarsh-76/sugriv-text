from pydantic import BaseModel

# process a file if it exists 
class FileRequest(BaseModel):
    name: str
    prompt: str | None = None