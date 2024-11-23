PYTHON := python

install:
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install -r requirements.txt
run:
	$(PYTHON) src/main.py