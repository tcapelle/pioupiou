"""HTTP API for the current Traverse prediction and historical dashboard."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from pioupiou.dashboard import build_dashboard_data
from pioupiou.inference.deployment import ensure_deployment_model
from realtime_inference import predict_now


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
model = Path(os.environ.get("TRAVERSE_MODEL", "artifacts/traverse_model.joblib"))
model_manifest = Path(__file__).parent / "deployment_model.json"
dashboard_html = Path(__file__).parent / "pioupiou" / "dashboard.html"


@app.get("/predict")
def predict():
    try:
        ensure_deployment_model(model, model_manifest)
        return predict_now(model)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return dashboard_html.read_text()


@app.get("/dashboard/data")
def dashboard_data():
    return build_dashboard_data()
