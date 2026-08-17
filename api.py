"""HTTP API for the current Traverse prediction."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from realtime_inference import predict_now


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
model = Path(os.environ.get("TRAVERSE_MODEL", "artifacts/traverse_model.joblib"))


@app.get("/predict")
def predict():
    try:
        return predict_now(model)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
