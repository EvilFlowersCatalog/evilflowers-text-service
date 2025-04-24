from consumer import DefaultConsumer
from config.consumer_config import consumer_conf, consumer_subscriptions
from text_handler.TextService import TextHandler
import json

class MyConsumer(DefaultConsumer):
    def msg_process(self, msg):
        
        json_string = msg.value().decode('utf-8')
        json_object = json.loads(json_string)
        
        text_handler = TextHandler(json_object["document_path"])
        text = text_handler.extract_text()
        
        print(text)

if __name__ == "__main__":
    consumer = MyConsumer(consumer_conf, consumer_subscriptions)
    consumer.start_consume()
    print("Consumer started")
