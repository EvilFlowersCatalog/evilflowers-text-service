from confluent_kafka import Producer as KafkaProducer
import json
from config.Config import Config

cycle = True

class Producer:
    def __init__(self, topic, additional_config=None):
        # Merge default config with any additional config
        producer_config = conf.copy()
        if additional_config:
            producer_config.update(additional_config)
            
        self.producer = KafkaProducer(producer_config)
        self.topic = topic

    def produce_message(self, message_data, key=None):
        try:
            # Convert dict/object to JSON string if not already a string
            message = (
                json.dumps(message_data) 
                if not isinstance(message_data, str) 
                else message_data
            )
            self.producer.produce(
                topic=self.topic,
                value=message.encode('utf-8'),
                key=key.encode('utf-8') if key else None,
                callback=self.delivery_report
            )
            self.producer.flush()
                
        except Exception as e:
            print(f"Error producing message: {e}")
            raise

    def delivery_report(self, err, msg):
        if err:
            print(f'Message delivery failed: {err}')
        else:
            global cycle
            cycle = False
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def close(self):
        """Clean up resources"""
        self.producer.flush()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

__all__ = ['Producer']
