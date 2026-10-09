.PHONY: install data pipeline train test api dashboard docker-build docker-up clean

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/generate_sample_data.py --customers 3000 --products 800 --interactions 60000

pipeline:
	python scripts/run_pipeline.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

clean:
	rm -rf __pycache__ .pytest_cache outputs/metrics/* outputs/figures/* artifacts/models/*
