.PHONY: install install-frontend dev run-frontend test docker-up docker-down

install:
	pip install -r requirements-dev.txt

install-frontend:
	pip install -r requirements-frontend.txt

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	streamlit run frontend/streamlit_app.py

test:
	pytest -v

docker-up:
	docker compose up --build

docker-down:
	docker compose down
