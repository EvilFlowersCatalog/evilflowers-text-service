import os
from dotenv import load_dotenv

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = {}
            load_dotenv()
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        # Assuming the use of a library like os.environ to access environment variables
        # This is a placeholder for actual environment variable loading logic
        self._config = {
            'TABLE_EXTRACTOR': os.getenv('TABLE_EXTRACTOR'),
            'TABLE_EXTRACTOR_MODE': os.getenv('TABLE_EXTRACTOR_MODE')
        }

    def get_config(self):
        return self._config
