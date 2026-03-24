from config.Config import Config

from celery import Celery

app = Celery('text_service', broker='redis://redis:6379/0', backend=Config.REDIS_URL  # Already there, but verify
)

result = app.send_task(
    'evilflowers_text_worker.process_pdf',
    args=['0b52959f-1c2a-4e1f-b44f-4e2c9bd1403e/42ca0c8c-243b-4a49-8465-73ada285a802.pdf', 'test-123'],
    queue='evilflowers_text_worker'
)

print(f"Task ID: {result.id}")
# print("Waiting for result...")
print(result.get(timeout=300))  # Wait up to 5 minutes