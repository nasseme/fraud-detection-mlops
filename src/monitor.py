import pandas as pd
import numpy as np
import json
import os
import requests
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

load_dotenv()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet("data/processed/train.parquet")
    prod_sim = pd.read_parquet("data/processed/production_sim.parquet")
    feature_cols = [c for c in train.columns if c != "Class"]
    return train[feature_cols], prod_sim[feature_cols]


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    definition = DataDefinition()
    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    Path("monitoring/reports").mkdir(parents=True, exist_ok=True)

    snapshot = result.json()
    with open("monitoring/reports/drift_report.json", "w") as f:
        f.write(snapshot)

    logger.info("Drift report saved to monitoring/reports/drift_report.json")

    return json.loads(snapshot)


def parse_drift_results(result: dict) -> dict:
    try:
        drift_results = result["metrics"][0]["result"]
        n_drifted = drift_results.get("number_of_drifted_columns", 0)
        n_total = drift_results.get("number_of_columns", len(drift_results))
        dataset_drift = drift_results.get("dataset_drift", n_drifted > n_total * 0.5)
    except Exception:
        n_drifted = 0
        n_total = 0
        dataset_drift = False

    summary = {
        "dataset_drift_detected": dataset_drift,
        "drifted_features": n_drifted,
        "total_features": n_total,
        "drift_ratio": n_drifted / n_total if n_total > 0 else 0
    }

    logger.info(f"Drift summary: {summary}")
    return summary


def send_slack_alert(summary: dict) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL non défini — alerte Slack ignorée")
        return

    message = {
        "text": (
            f":warning: *Drift détecté en production*\n"
            f"• Features driftées : {summary['drifted_features']}/{summary['total_features']}\n"
            f"• Ratio de drift : {summary['drift_ratio']:.2%}\n"
            f"• Action recommandée : retraining du modèle"
        )
    }

    response = requests.post(webhook_url, json=message)
    if response.status_code == 200:
        logger.info("Slack alert sent successfully")
    else:
        logger.error(f"Slack alert failed: {response.status_code}")


def monitor() -> None:
    reference, current = load_data()
    result = run_evidently_report(reference, current)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    summary = parse_drift_results(result)

    if summary["dataset_drift_detected"]:
        logger.warning("Dataset drift detected — sending alert")
        send_slack_alert(summary)
    else:
        logger.info("No significant drift detected")

    with open("monitoring/reports/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    monitor()