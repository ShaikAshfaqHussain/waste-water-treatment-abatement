"""FastAPI Application for Wastewater Treatment Pollution Abatement Platform.

Provides high-performance REST API endpoints and serves the modern Tailwind & Chart.js dashboard.
"""

import json
import os
import sys
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ml_model import WastewaterMLPipeline
from rule_engine import WastewaterAbatementEngine, REGULATORY_THRESHOLDS

app = FastAPI(
    title="EcoVision AI: Wastewater Treatment Abatement API",
    description="Multi-target ML predictions (DO, BOD, COD, pH, Turbidity) and regulatory abatement rule engine.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global lazy loaders for Pipeline & Engine
ml_pipeline: Optional[WastewaterMLPipeline] = None
rule_engine: Optional[WastewaterAbatementEngine] = None
cached_cleaned_df: Optional[pd.DataFrame] = None


def get_pipeline():
    global ml_pipeline
    if ml_pipeline is None:
        ml_pipeline = WastewaterMLPipeline(model_dir=MODELS_DIR)
        ml_pipeline.load()
    return ml_pipeline


def get_rule_engine():
    global rule_engine
    if rule_engine is None:
        rule_engine = WastewaterAbatementEngine()
    return rule_engine


def get_cleaned_data() -> pd.DataFrame:
    global cached_cleaned_df
    if cached_cleaned_df is None:
        csv_path = os.path.join(DATA_DIR, "cleaned_wastewater_data.csv")
        if not os.path.exists(csv_path):
            from run_pipeline import run_full_pipeline
            run_full_pipeline(output_dir=BASE_DIR)
        cached_cleaned_df = pd.read_csv(csv_path)
    return cached_cleaned_df


class SimulationInputSchema(BaseModel):
    Q_in: float = Field(22000.0, description="Influent Flow Rate (m3/day)")
    Temp_water: float = Field(20.0, description="Wastewater Temperature (deg C)")
    pH_in: float = Field(7.3, description="Influent pH")
    BOD_in: float = Field(220.0, description="Influent 5-day Biochemical Demand (mg/L)")
    COD_in: float = Field(460.0, description="Influent Chemical Oxygen Demand (mg/L)")
    TSS_in: float = Field(240.0, description="Influent Total Suspended Solids (mg/L)")
    TN_in: float = Field(38.0, description="Influent Total Nitrogen (mg/L)")
    TP_in: float = Field(7.0, description="Influent Total Phosphorus (mg/L)")
    DO_aeration: float = Field(2.2, description="Aeration Basin Dissolved Oxygen (mg/L)")
    Blower_speed_pct: float = Field(70.0, description="Blower Frequency Speed (%)")
    RAS_flow: float = Field(14000.0, description="Return Activated Sludge Flow (m3/day)")
    WAS_flow: float = Field(240.0, description="Waste Activated Sludge Flow (m3/day)")
    MLSS: float = Field(3200.0, description="Aeration Mixed Liquor Suspended Solids (mg/L)")
    PAC_dosage: float = Field(12.0, description="Poly-Aluminum Chloride Coagulant Dose (mg/L)")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard(request: Request):
    """Serves the interactive single-page dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "EcoVision Wastewater Treatment AI",
        "version": "1.0.0",
        "models_loaded": os.path.exists(os.path.join(MODELS_DIR, "rf_multioutput_model.joblib")),
    }


@app.get("/api/summary")
def get_system_summary():
    """Returns cleaning report, dataset summary, and model performance metrics."""
    report_path = os.path.join(DATA_DIR, "cleaning_report.json")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")

    cleaning_report = {}
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            cleaning_report = json.load(f)

    model_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)

    return {
        "cleaning_report": cleaning_report,
        "model_metrics": model_metrics,
        "regulatory_thresholds": REGULATORY_THRESHOLDS,
    }


@app.get("/api/timeseries")
def get_timeseries_data(
    metric: str = Query("BOD", description="Target metric: BOD, COD, DO, pH, Turbidity"),
    limit: int = Query(200, description="Number of recent hourly data points"),
):
    """Returns time-series telemetry data for interactive visualization."""
    df = get_cleaned_data()
    sub_df = df.tail(limit).copy()

    metric_mapping = {
        "BOD": ("BOD_eff", "BOD_in", "mg/L"),
        "COD": ("COD_eff", "COD_in", "mg/L"),
        "DO": ("DO_eff", "DO_aeration", "mg/L"),
        "pH": ("pH_eff", "pH_in", "pH"),
        "Turbidity": ("Turbidity_eff", None, "NTU"),
    }

    if metric not in metric_mapping:
        raise HTTPException(status_code=400, detail=f"Unsupported metric: {metric}")

    eff_col, inf_col, unit = metric_mapping[metric]

    timestamps = [str(ts)[5:16] for ts in sub_df["timestamp"]]
    eff_vals = [float(x) for x in sub_df[eff_col]]
    inf_vals = [float(x) for x in sub_df[inf_col]] if inf_col and inf_col in sub_df.columns else None

    return {
        "metric": metric,
        "unit": unit,
        "timestamps": timestamps,
        "effluent_values": eff_vals,
        "influent_values": inf_vals,
        "regulatory_limits": REGULATORY_THRESHOLDS.get(eff_col, {}),
    }


@app.post("/api/predict")
def predict_and_evaluate(input_data: SimulationInputSchema):
    """Predicts effluent metrics (DO, BOD, COD, pH, Turbidity) and evaluates abatement actions."""
    pipeline = get_pipeline()
    engine = get_rule_engine()

    raw_dict = input_data.model_dump()
    predictions = pipeline.predict_from_raw_inputs(raw_dict)
    compliance = engine.evaluate_compliance(predictions, raw_dict)

    return {
        "inputs": raw_dict,
        "predictions": predictions,
        "compliance": compliance,
    }


@app.get("/api/download/cleaned-csv")
def download_cleaned_csv():
    """Serves the cleaned and feature-engineered CSV for download."""
    csv_path = os.path.join(DATA_DIR, "cleaned_wastewater_data.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Cleaned CSV not found. Run pipeline first.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="cleaned_wastewater_telemetry.csv",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="127.0.0.1", port=8000, reload=True)
