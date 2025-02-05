PYTHON := python

install:
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install -r requirements.txt
run:
	$(PYTHON) src/main.py

docker-rebuild:
	docker compose down && \
	docker builder prune -f && \
	docker compose build --no-cache && \
	docker compose up -d
