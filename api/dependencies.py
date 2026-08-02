import mlflow.xgboost
import joblib
import json
import mlflow
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

_model = None
_preprocessor = None
_threshold = None


def load_artifacts():
    global _model, _preprocessor, _threshold

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    model_name = os.getenv("MODEL_NAME", "fraud-detector")

    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{model_name}/latest"
        _model = mlflow.xgboost.load_model(model_uri)
        logger.info(f"Model loaded from registry: {model_uri}")
    except Exception:
        runs = mlflow.search_runs(
            experiment_names=["fraud-detection"],
            order_by=["start_time DESC"],
            max_results=1
        )
        run_id = runs.iloc[0]["run_id"]
        _model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        logger.info(f"Model loaded from run: {run_id}")

    _preprocessor = joblib.load("data/processed/preprocessor.joblib")
    logger.info("Preprocessor loaded")

    with open("data/processed/threshold.json") as f:
        _threshold = json.load(f)["threshold"]
    logger.info(f"Threshold loaded: {_threshold}")


def get_model():
    return _model


def get_preprocessor():
    return _preprocessor


def get_threshold():
    return _threshold