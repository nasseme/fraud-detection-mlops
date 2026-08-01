import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_processed_files_exist():
    """Vérifie que tous les fichiers de preprocessing existent."""
    files = [
        "data/processed/X_train.npy",
        "data/processed/y_train.npy",
        "data/processed/X_test.npy",
        "data/processed/y_test.npy",
        "data/processed/X_prod.npy",
        "data/processed/y_prod.npy",
        "data/processed/preprocessor.joblib",
    ]
    for f in files:
        assert Path(f).exists(), f"Fichier manquant : {f}"


def test_shapes_consistency():
    """X et y doivent avoir le même nombre de lignes."""
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_smote_balance():
    """Après SMOTE le train set doit être équilibré à 50/50."""
    y_train = np.load("data/processed/y_train.npy")
    fraud_rate = y_train.mean()
    assert 0.49 <= fraud_rate <= 0.51, f"SMOTE mal appliqué : fraud_rate={fraud_rate:.4f}"


def test_no_nan_in_arrays():
    """Aucune valeur NaN dans les arrays preprocessés."""
    for fname in ["X_train.npy", "X_test.npy", "X_prod.npy"]:
        arr = np.load(f"data/processed/{fname}")
        assert not np.isnan(arr).any(), f"NaN détecté dans {fname}"


def test_feature_count():
    """Le nombre de features doit être 30 (V1-V28 + Amount + Time)."""
    X_train = np.load("data/processed/X_train.npy")
    assert X_train.shape[1] == 30, f"Attendu 30 features, obtenu {X_train.shape[1]}"


def test_labels_binary():
    """Les labels doivent être uniquement 0 et 1."""
    for fname in ["y_train.npy", "y_test.npy", "y_prod.npy"]:
        y = np.load(f"data/processed/{fname}")
        unique = np.unique(y)
        assert set(unique).issubset({0, 1}), f"Labels non binaires dans {fname}: {unique}"