.PHONY: install train test api monitor

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

train:
	python pipelines/training_flow.py

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload --port 8000

monitor:
	python src/monitor.py