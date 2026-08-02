from fastapi import FastAPI, Depends
from loguru import logger
import numpy as np
import pandas as pd

from api.schemas import TransactionInput, PredictionOutput
from api.dependencies import load_artifacts, get_model, get_preprocessor, get_threshold

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire — MLOps Portfolio",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    load_artifacts()
    logger.info("API started — artifacts loaded")


@app.get("/health")
def health():
    return {"status": "ok", "model": "fraud-detector"}


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    model = get_model()
    preprocessor = get_preprocessor()
    threshold = get_threshold()

    feature_order = (
        ["Time"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount"]
    )

    input_df = pd.DataFrame([transaction.model_dump()])[feature_order]
    input_processed = preprocessor.transform(input_df)

    fraud_proba = float(model.predict_proba(input_processed)[0, 1])
    is_fraud = fraud_proba >= threshold

    if fraud_proba < 0.3:
        risk_level = "LOW"
    elif fraud_proba < threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    logger.info(f"Prediction: fraud={is_fraud} | proba={fraud_proba:.4f} | risk={risk_level}")

    return PredictionOutput(
        is_fraud=is_fraud,
        fraud_probability=fraud_proba,
        threshold_used=threshold,
        risk_level=risk_level
    )