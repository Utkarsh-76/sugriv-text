import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import json
import time
import threading
import confluent_kafka
import requests
from dotenv import load_dotenv
from src.utils.logger import logging as logger
from src.llm.model.sugriv import sugriv


# Load environment variables from .env file
load_dotenv()

KAFKA_TOPICS= str(os.getenv("KAFKA_TOPICS"))
KAFKA_SERVER = str(os.getenv("KAFKA_SERVER"))
KAFKA_CONSUMER_GROUP_NAME = str(os.getenv("KAFKA_CONSUMER_GROUP_NAME"))
KAFKA_CONSUMER_OFFSET = str(os.getenv("KAFKA_CONSUMER_OFFSET"))
PRETRAIN_URL = "http://127.0.0.1:8000/pretrain"

consumer_conf = {'bootstrap.servers': KAFKA_SERVER,'group.id': KAFKA_CONSUMER_GROUP_NAME,'auto.offset.reset': KAFKA_CONSUMER_OFFSET}

class KafkaConsumer():
    def __init__(self,conf) -> None:
        self.c = confluent_kafka.Consumer(conf)
        self.c.subscribe([KAFKA_TOPICS])
        self.sugriv = sugriv.get_model()

    def worker(self,text:str):
        logger.info("Worker thread is starting.")
        time.sleep(5)  # Simulate a task taking 5 seconds
        if text is not None:
            requests.post(PRETRAIN_URL,data=json.dumps({'text': json.loads(text)["data"]}))
        logger.info("Worker thread has finished.")

    def consume(self):
        ''' consume the messgaes from a kafka consumer'''
            # Create a new thread that runs the worker function
        while True:
            #poll the consumer every 10 seconds
            msg = self.c.poll(10.0)
            if msg != None:
                text = msg.value().decode('utf-8')
                logger.info(text)
                # spin a new  thread to pretrain so as not to block
                worker_thread = threading.Thread(target=self.worker, args=(text,))
                worker_thread.start()

            logger.info('No message recieved')
            time.sleep(10) # Sleep for 10 seconds

if __name__ == "__main__":
   consumer = KafkaConsumer(consumer_conf)
   logger.info('starting consumer')
   consumer.consume()

