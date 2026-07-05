import pandas as pd
import yaml
from pathlib import Path
from loguru import logger


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Fraud rate: {df['Class'].mean():.4%}")
    return df


def split_data(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporel basé sur la feature Time.
    Les données sont triées par Time croissant.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)

    train_end = int(n * params["data"]["time_split_train"])
    test_end = train_end + int(n * params["data"]["time_split_test"])

    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:test_end].copy()
    prod_sim = df.iloc[test_end:].copy()

    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows | Prod sim: {len(prod_sim)} rows")
    logger.info(f"Train fraud rate: {train['Class'].mean():.4%}")
    logger.info(f"Test fraud rate: {test['Class'].mean():.4%}")
    logger.info(f"Prod sim fraud rate: {prod_sim['Class'].mean():.4%}")

    return train, test, prod_sim


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prod_sim: pd.DataFrame,
    params: dict
) -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train.to_parquet(params["data"]["train_path"], index=False)
    test.to_parquet(params["data"]["test_path"], index=False)
    prod_sim.to_parquet(params["data"]["prod_sim_path"], index=False)

    logger.info("Splits saved to data/processed/")


if __name__ == "__main__":
    params = load_params()
    df = load_raw_data(params["data"]["raw_path"])
    train, test, prod_sim = split_data(df, params)
    save_splits(train, test, prod_sim, params)