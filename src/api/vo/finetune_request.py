from typing import List
from pydantic import BaseModel

# process a file if it exists 
class FinetuneRequest(BaseModel):
    texts: List[str]