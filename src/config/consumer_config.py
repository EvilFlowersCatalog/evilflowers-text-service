import os
from dotenv import load_dotenv

load_dotenv()

consumer_conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'security.protocol': os.getenv('KAFKA_SECURITY_PROTOCOL'),
    'sasl.mechanism': os.getenv('KAFKA_SASL_MECHANISM'),
    'sasl.username': os.getenv('KAFKA_SASL_USERNAME'),
    'sasl.password': os.getenv('KAFKA_SASL_PASSWORD'),
    'group.id': os.getenv('KAFKA_CONSUMER_GROUP'),
    'auto.offset.reset': os.getenv('KAFKA_AUTO_OFFSET_RESET'),
}
consumer_subscriptions = ["text-service-topic"]