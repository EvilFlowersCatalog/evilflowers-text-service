from abc import abstractmethod
from confluent_kafka import Consumer
import sys
from confluent_kafka.error import KafkaError, KafkaException

class DefaultConsumer():
    def __init__(self, conf: dict, subscriptions: list):
        self.conf = conf
        self.consumer = self.build_consumer()
        self.running = True
        self.subscriptions = subscriptions

    def build_consumer(self):
        return Consumer(self.conf)

    def start_consume(self):
        self.basic_consume_loop(self.subscriptions)

    def subscribe(self, subscriptions: list):
        self.consumer.subscribe(subscriptions)


    def basic_consume_loop(self, topics: list):
        try:
            self.consumer.subscribe(topics)
            print("Subscribed to topics: ", topics)
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None: continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                        (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    self.msg_process(msg)
        finally:
            self.consumer.close()

    def shutdown(self):
        self.running = False

    @abstractmethod
    def msg_process(self, msg):
        pass