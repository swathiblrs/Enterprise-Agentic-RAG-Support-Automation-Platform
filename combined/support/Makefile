.PHONY: install test evaluate run-api run-ui docker-build docker-up docker-down

install:
	pip install -r requirements.txt

test:
	python -m pytest

evaluate:
	python -m src.evaluator

run-api:
	uvicorn src.api:app --reload

run-ui:
	streamlit run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
