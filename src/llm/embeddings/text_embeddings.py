import os
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables from .env file
load_dotenv()

MAXIMUM_SEQUENCE_LENGTH = int(os.getenv("MAXIMUM_SEQUENCE_LENGTH"))
D_MODEL = int(os.getenv("D_MODEL"))

class TextEmbeddings():
    def __init__(self) -> None:

        # we will use the open ai embeddingand make the dimentions 1024
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=D_MODEL,
        )


    def get_text_embedding(self, text):
        embeddings = self.embed_model.get_text_embedding(text)
        return embeddings
