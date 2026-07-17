import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def build_preprocessor() -> ColumnTransformer:
    """
    Pipeline sklearn pour le preprocessing.
    
    On applique un log-transform sur Amount car sa distribution est très skewed
    (quelques transactions à montants très élevés tirent la moyenne vers le haut).
    Sans ça, XGBoost accorderait trop d'importance aux valeurs extrêmes.
    
    On applique un StandardScaler sur Time pour le normaliser.
    
    Les features V1-V28 sont déjà issues d'une PCA donc déjà centrées-réduites
    par la banque — on ne les retouche pas.
    """
    amount_pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
    ])

    time_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("amount", amount_pipeline, ["Amount"]),
            ("time", time_pipeline, ["Time"]),
        ],
        remainder="passthrough"
    )

    return preprocessor


def apply_smote(X_train: np.ndarray, y_train: pd.Series, random_state: int = 42):
    """
    SMOTE génère des exemples synthétiques de la classe minoritaire (fraudes)
    par interpolation entre exemples existants.
    
    CRITIQUE : on applique SMOTE uniquement sur le train set, JAMAIS sur le test
    ou le prod_sim. Appliquer SMOTE sur le test ferait fuiter de l'information
    synthétique dans l'évaluation — erreur classique et fatale pour la rigueur.
    
    Après SMOTE le train set est parfaitement équilibré 50/50.
    """
    logger.info(f"Avant SMOTE — fraudes : {y_train.sum()} / {len(y_train)}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Après SMOTE — fraudes : {y_resampled.sum()} / {len(y_resampled)}")
    return X_resampled, y_resampled


def preprocess(params: dict) -> None:
    """
    Pipeline complet :
    1. Charge les splits parquet produits par ingest.py
    2. Sépare features / cible
    3. Fit le preprocessor sur le train uniquement (jamais sur test/prod)
       → évite le data leakage : le scaler ne doit pas "voir" le test
    4. Transforme les 3 splits avec le même preprocessor fitté
    5. Applique SMOTE sur le train uniquement
    6. Sauvegarde les arrays numpy + le preprocessor fitté (pour l'inférence)
    """
    train = pd.read_parquet(params["data"]["train_path"])
    test = pd.read_parquet(params["data"]["test_path"])
    prod_sim = pd.read_parquet(params["data"]["prod_sim_path"])

    feature_cols = [c for c in train.columns if c != "Class"]
    
    X_train = train[feature_cols]
    y_train = train["Class"]
    X_test = test[feature_cols]
    y_test = test["Class"]
    X_prod = prod_sim[feature_cols]
    y_prod = prod_sim["Class"]

    preprocessor = build_preprocessor()

    # Fit UNIQUEMENT sur le train
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_prod_processed = preprocessor.transform(X_prod)

    logger.info(f"X_train après preprocessing : {X_train_processed.shape}")

    # SMOTE sur train uniquement
    X_train_resampled, y_train_resampled = apply_smote(
        X_train_processed, y_train, params["model"]["random_state"]
    )

    # Sauvegarde
    output_dir = Path("data/processed")
    np.save(output_dir / "X_train.npy", X_train_resampled)
    np.save(output_dir / "y_train.npy", y_train_resampled)
    np.save(output_dir / "X_test.npy", X_test_processed)
    np.save(output_dir / "y_test.npy", y_test.values)
    np.save(output_dir / "X_prod.npy", X_prod_processed)
    np.save(output_dir / "y_prod.npy", y_prod.values)

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    logger.info("Preprocessor et arrays sauvegardés dans data/processed/")


if __name__ == "__main__":
    params = load_params()
    preprocess(params)