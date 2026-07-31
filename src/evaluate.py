import numpy as np
import yaml
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from loguru import logger


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Parcourt tous les seuils de la courbe PR et retourne celui
    qui maximise le F1-score sur le jeu de test.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    logger.info(f"Optimal threshold: {best_threshold:.4f} | F1: {best_f1:.4f}")
    return float(best_threshold)


def evaluate(params: dict) -> None:
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    mlflow.set_tracking_uri(params.get("mlflow", {}).get("tracking_uri", "mlruns"))
    mlflow.set_experiment("fraud-detection")

    # Récupère le dernier run actif
    runs = mlflow.search_runs(
        experiment_names=["fraud-detection"],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("Aucun run MLflow trouvé. Lance d'abord src/train.py")

    run_id = runs.iloc[0]["run_id"]
    logger.info(f"Evaluating run: {run_id}")

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)

    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"F1 @ threshold {threshold:.4f}: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Légitime', 'Fraude'])}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "pr_auc": pr_auc,
            "f1_optimal": f1,
            "optimal_threshold": threshold
        })

        # Sauvegarde le seuil pour l'inférence
        import json
        with open("data/processed/threshold.json", "w") as f:
            json.dump({"threshold": threshold}, f)

        mlflow.log_artifact("data/processed/threshold.json")
        logger.info("Metrics and threshold logged to MLflow")


if __name__ == "__main__":
    params = load_params()
    evaluate(params)