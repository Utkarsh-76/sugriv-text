import os
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv
from src.utils.logger import logging as logger

# Load environment variables from .env file
load_dotenv()

KAFKA_TOPICS= str(os.getenv("KAFKA_TOPICS"))

class KafKaTopics():
    def __init__(self,config) -> None:
        admin_client = AdminClient(config)
        topic_list = []
        topic_list.append(NewTopic(KAFKA_TOPICS, 1, 1))
        admin_client.create_topics(topic_list)