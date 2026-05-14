from config.Config import Config

from celery import Celery

app = Celery('text_service', broker='redis://redis:6379/0', backend=Config.REDIS_URL  # Already there, but verify
)

result = app.send_task(
    'evilflowers_text_worker.process_pdf',
    args=['cc12e850.pdf', 'test-124'],
    queue='evilflowers_text_worker'
)

print(f"Task ID: {result.id}")
# print("Waiting for result...")
print(result.get(timeout=600))  # Wait up to 5 minutes