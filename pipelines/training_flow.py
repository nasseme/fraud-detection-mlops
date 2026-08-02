import subprocess
import sys
from prefect import flow, task
from prefect.logging import get_run_logger


@task(name="ingest", retries=2, retry_delay_seconds=10)
def ingest_task():
    logger = get_run_logger()
    logger.info("Starting data ingestion...")
    result = subprocess.run(
        [sys.executable, "src/ingest.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ingest failed:\n{result.stderr}")
    logger.info(result.stdout)


@task(name="validate", retries=1)
def validate_task():
    logger = get_run_logger()
    logger.info("Validating data...")
    result = subprocess.run(
        [sys.executable, "src/validate.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Validation failed:\n{result.stderr}")
    logger.info(result.stdout)


@task(name="preprocess", retries=1)
def preprocess_task():
    logger = get_run_logger()
    logger.info("Preprocessing data...")
    result = subprocess.run(
        [sys.executable, "src/preprocess.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Preprocess failed:\n{result.stderr}")
    logger.info(result.stdout)


@task(name="train", retries=1)
def train_task():
    logger = get_run_logger()
    logger.info("Training model...")
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    logger.info(result.stdout)


@task(name="evaluate")
def evaluate_task():
    logger = get_run_logger()
    logger.info("Evaluating model...")
    result = subprocess.run(
        [sys.executable, "src/evaluate.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed:\n{result.stderr}")
    logger.info(result.stdout)


@flow(name="fraud-detection-training", log_prints=True)
def training_flow():
    ingest_task()
    validate_task()
    preprocess_task()
    train_task()
    evaluate_task()


if __name__ == "__main__":
    training_flow()