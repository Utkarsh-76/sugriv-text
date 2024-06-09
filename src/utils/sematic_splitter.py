from llama_index.core import SimpleDirectoryReader

from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

class SemanticSplitter():

    def __init__(self):
        # we will use the open ai embeddingand make the dimentions 1024
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=768,
        )

        # create a semantic splitter
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )

        # also baseline splitter
        self.base_splitter = SentenceSplitter(chunk_size=512)

    def get_nodes(self,input_files):
        # load documents
        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        nodes = self.splitter.get_nodes_from_documents(documents)
        return nodes