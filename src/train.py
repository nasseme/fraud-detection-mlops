import numpy as np
import yaml
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from loguru import logger
from pathlib import Path


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def train(params: dict) -> None:
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"Positive class ratio after SMOTE: {y_train.mean():.4%}")

    mlflow.set_tracking_uri(params.get("mlflow", {}).get("tracking_uri", "mlruns"))
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgboost_baseline"):
        model_params = {
            "n_estimators": params["model"]["n_estimators"],
            "max_depth": params["model"]["max_depth"],
            "learning_rate": params["model"]["learning_rate"],
            "scale_pos_weight": params["model"]["scale_pos_weight"],
            "eval_metric": params["model"]["eval_metric"],
            "random_state": params["model"]["random_state"],
            "use_label_encoder": False,
        }

        mlflow.log_params(model_params)

        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)

        logger.info("Model trained successfully")

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud-detector"
        )

        logger.info("Model logged to MLflow")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")


if __name__ == "__main__":
    params = load_params()
    train(params)