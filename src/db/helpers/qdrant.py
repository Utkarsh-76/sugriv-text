import os
import numpy as np
import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from  src.utils.logger import logging as logger
from dotenv import load_dotenv
# from logger import logging as logger

load_dotenv()

QDRANT_HOST = str(os.getenv("QDARANT_HOST"))
QDRANT_PORT = str(os.getenv("QDARANT_PORT"))
VECTOR_SIZE = int(os.getenv('D_MODEL'))

class QdrantDB():

    def __init__(self) -> None:
        ''' connects to the qdrant vector database'''
        try:
            logger.info(f'connecting to qdrant at {QDRANT_HOST} and port {QDRANT_PORT}')
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except ConnectionError as e:
            logger.error(f'error connecting to quadrant db at {QDRANT_HOST} and {QDRANT_PORT}')
            logger.error(e)

    def create_collection(self,name):
        ''' create a collection in the vector database '''
        try:
            logger.info(f'creating a collection with name {name}')
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            return {"status": "ok"}
        except RuntimeError as e:
            logger.error(e)
            logger.error(f'can not create collection with {name}')

    def get(self,name,vector,top_k=5):
        ''' get a result from the vector databse using the given query'''
        try:
            logger.info(f'getting from collection with name {name} using query')
            search_result = self.client.search(
                collection_name=name, query_vector=vector, limit=top_k
            )
            logger.info("successfully got results from collection")
            return {"status":"ok","result":search_result[0].json()}
        except RuntimeError as e:
            logger.error(f'error executing query on collection {name}')
            logger.error(e)

    def add(self,id,payload,vector,name):
        ''' adds if the id does not exists or else uptates the existing 
            recordd for the given collection'''
        try:
            logger.info(f'upserting {str(payload)} into vector database')
            self.client.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=id,
                        payload=payload,
                        vector=vector,
                    )
                ],
            )
            logger.info("upsert successfull")
        except RuntimeError as e:
            logger.error(f'error upserting {str(payload)} into vector database')
            logger.error(e)

    def check_collection_exists(self,collection_name):
        try:       
            logger.info("check if client exists")
            if collection_name == "prompts":
                response = requests.get("http://172.18.144.1:6333/collections/prompts/exists")
                if response.status_code == 404:
                    return False
                return True
            
            if collection_name == "documents":
                response = requests.get("http://172.18.144.1:6333/collections/documents/exists")
                if response.status_code == 404:
                    return False
                return True
        except RuntimeError as e:
            logger.error(f'error checking if collection exists')
            logger.error(e)
