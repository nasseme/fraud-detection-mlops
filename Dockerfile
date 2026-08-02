FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY mlflow.db ./mlflow.db
COPY params.yaml ./params.yaml

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]