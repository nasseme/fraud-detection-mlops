import numpy as np
import pytest
import mlflow
import mlflow.xgboost
import json
from pathlib import Path


def load_model_and_threshold():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=["fraud-detection"],
        order_by=["start_time DESC"],
        max_results=1
    )
    assert not runs.empty, "Aucun run MLflow trouvé"
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    with open("data/processed/threshold.json") as f:
        threshold = json.load(f)["threshold"]

    return model, threshold


def test_model_output_shape():
    """Le modèle doit retourner une probabilité par transaction."""
    model, _ = load_model_and_threshold()
    X_test = np.load("data/processed/X_test.npy")
    proba = model.predict_proba(X_test[:10])
    assert proba.shape == (10, 2), f"Shape inattendue : {proba.shape}"


def test_probabilities_in_range():
    """Les probabilités doivent être entre 0 et 1."""
    model, _ = load_model_and_threshold()
    X_test = np.load("data/processed/X_test.npy")
    proba = model.predict_proba(X_test[:100])[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_threshold_in_range():
    """Le seuil optimal doit être entre 0 et 1."""
    _, threshold = load_model_and_threshold()
    assert 0.0 < threshold < 1.0, f"Seuil hors range : {threshold}"


def test_predictions_binary():
    """Les prédictions finales doivent être 0 ou 1."""
    model, threshold = load_model_and_threshold()
    X_test = np.load("data/processed/X_test.npy")
    proba = model.predict_proba(X_test[:100])[:, 1]
    preds = (proba >= threshold).astype(int)
    assert set(np.unique(preds)).issubset({0, 1})


def test_fraud_detection_rate():
    """Le modèle doit détecter au moins 50% des fraudes sur le test set."""
    model, threshold = load_model_and_threshold()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    fraud_idx = y_test == 1
    recall = preds[fraud_idx].mean()
    assert recall >= 0.50, f"Recall trop faible : {recall:.4f}"