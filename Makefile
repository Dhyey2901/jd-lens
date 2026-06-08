.PHONY: install api app dev test lint eval

install:
	pip install -r requirements.txt

api:
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app.py

# Starts API in background then launches the Streamlit UI
dev:
	@echo "Starting JD Lens API on http://localhost:8000 ..."
	uvicorn api:app --port 8000 --reload &
	@echo "Starting JD Lens UI on http://localhost:8501 ..."
	streamlit run app.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check . --select=E,F,W --ignore=E501

# Full eval against ground-truth pairs (downloads embedding model on first run)
eval:
	python eval/run_eval.py

# Fast eval — signals only, no model download
eval-fast:
	python eval/run_eval.py --fast
