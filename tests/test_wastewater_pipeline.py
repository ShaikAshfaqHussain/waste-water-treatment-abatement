"""Automated Pytest Suite for Wastewater Treatment Pollution Abatement System."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Configure path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data_generator import generate_wastewater_dataset
from data_cleaning import WastewaterDataCleaner
from ml_model import WastewaterMLPipeline, TARGET_COLUMNS
from rule_engine import WastewaterAbatementEngine, REGULATORY_THRESHOLDS


def test_data_generator_output():
    """Validates data generator integrity, dimensions, and realistic anomaly injection."""
    df = generate_wastewater_dataset(n_samples=200, random_seed=42, inject_anomalies=True)
    assert len(df) == 200
    assert "timestamp" in df.columns
    for target in TARGET_COLUMNS:
        assert target in df.columns
    # Check that anomalies were injected (NaNs exist)
    assert df.isnull().sum().sum() > 0


def test_data_cleaning_and_feature_engineering():
    """Validates that cleaner resolves missing values, fixes negative drifts, and engineers features."""
    raw_df = generate_wastewater_dataset(n_samples=300, random_seed=42, inject_anomalies=True)
    cleaner = WastewaterDataCleaner(tank_volume_m3=9000.0, clarifier_area_m2=1200.0)
    cleaned_df, report = cleaner.clean_and_engineer(raw_df, is_training=True)

    # 1. No remaining missing values
    assert cleaned_df.isnull().sum().sum() == 0

    # 2. Key engineered features exist
    expected_feats = [
        "BOD_COD_ratio_in",
        "COD_TN_ratio_in",
        "OLR_kg_m3_day",
        "FM_ratio",
        "HRT_hours",
        "Clarifier_SOR_m_h",
        "DO_deficit",
    ]
    for feat in expected_feats:
        assert feat in cleaned_df.columns

    # 3. Biodegradability ratio is within physical range (0 to 1)
    assert (cleaned_df["BOD_COD_ratio_in"] > 0).all()
    assert (cleaned_df["BOD_COD_ratio_in"] <= 1.0).all()

    # 4. Audit quality score is high
    assert report["data_quality_score"] > 95.0


def test_ml_pipeline_train_and_inference(tmp_path):
    """Validates multi-output training, metric calculation, persistence, and inference."""
    raw_df = generate_wastewater_dataset(n_samples=250, random_seed=42, inject_anomalies=False)
    cleaner = WastewaterDataCleaner()
    clean_df, _ = cleaner.clean_and_engineer(raw_df, is_training=True)

    pipeline = WastewaterMLPipeline(model_dir=str(tmp_path))
    metrics, holdout_df = pipeline.train_and_evaluate(clean_df, test_size=0.2, random_state=42)

    assert "overall_mean_R2" in metrics
    assert len(metrics["per_target_metrics"]) == 5
    for tgt in TARGET_COLUMNS:
        assert tgt in metrics["per_target_metrics"]
        assert "MAE" in metrics["per_target_metrics"][tgt]
        assert "RMSE" in metrics["per_target_metrics"][tgt]

    # Test persistence & reload
    pipeline.save()
    reloaded_pipeline = WastewaterMLPipeline(model_dir=str(tmp_path))
    reloaded_pipeline.load()

    # Test live prediction
    preds = reloaded_pipeline.predict_from_raw_inputs({
        "Q_in": 22000.0,
        "Temp_water": 20.0,
        "pH_in": 7.3,
        "BOD_in": 220.0,
        "COD_in": 460.0,
        "TSS_in": 240.0,
        "TN_in": 38.0,
        "TP_in": 7.0,
        "DO_aeration": 2.2,
        "Blower_speed_pct": 70.0,
        "RAS_flow": 14000.0,
        "WAS_flow": 240.0,
        "MLSS": 3200.0,
        "PAC_dosage": 12.0,
    })

    for tgt in TARGET_COLUMNS:
        assert tgt in preds
        assert isinstance(preds[tgt], float)


def test_rule_engine_compliant_scenario():
    """Validates that optimal effluent values yield OPTIMAL status and zero violations."""
    engine = WastewaterAbatementEngine()
    compliant_preds = {
        "DO_eff": 6.2,        # Standard >= 5.0
        "BOD_eff": 12.5,      # Cap <= 30.0
        "COD_eff": 58.0,      # Cap <= 125.0
        "pH_eff": 7.35,       # Band 6.5 - 8.5
        "Turbidity_eff": 2.8, # Cap <= 10.0
    }
    result = engine.evaluate_compliance(compliant_preds)
    assert result["overall_status"] == "OPTIMAL_COMPLIANT"
    assert result["critical_violations"] == 0
    assert result["warning_conditions"] == 0
    assert result["plant_health_score"] == 100
    assert len(result["abatement_actions"]) == 0


def test_rule_engine_critical_violations():
    """Validates that severe discharge exceedances trigger CRITICAL alerts and actionable abatement plans."""
    engine = WastewaterAbatementEngine()
    violation_preds = {
        "DO_eff": 1.8,         # Critical Hypoxia (< 3.0)
        "BOD_eff": 42.0,       # Severe BOD breach (> 30.0)
        "COD_eff": 145.0,      # Severe COD breach (> 125.0)
        "pH_eff": 5.4,         # Acidic shock (< 6.0)
        "Turbidity_eff": 16.5, # Clarifier washout (> 10.0)
    }
    result = engine.evaluate_compliance(violation_preds)
    assert result["overall_status"] == "CRITICAL_NON_COMPLIANT"
    assert result["critical_violations"] >= 4
    assert result["plant_health_score"] < 50
    assert len(result["abatement_actions"]) >= 4

    # Verify action cards contain root cause and chemical action steps
    for action in result["abatement_actions"]:
        assert "root_cause" in action
        assert "action_steps" in action
        assert len(action["action_steps"]) > 0
        assert "expected_impact" in action


def test_fastapi_endpoints():
    """Validates FastAPI core endpoints and request/response contracts."""
    from app_fastapi import (
        health_check,
        get_system_summary,
        get_timeseries_data,
        predict_and_evaluate,
        SimulationInputSchema,
    )

    # 1. Health endpoint
    health_res = health_check()
    assert health_res["status"] == "healthy"
    assert health_res["models_loaded"] is True

    # 2. Summary endpoint
    summary_res = get_system_summary()
    assert "cleaning_report" in summary_res
    assert "model_metrics" in summary_res
    assert "regulatory_thresholds" in summary_res

    # 3. Time-series endpoint
    ts_res = get_timeseries_data(metric="BOD", limit=25)
    assert ts_res["metric"] == "BOD"
    assert len(ts_res["timestamps"]) > 0
    assert len(ts_res["effluent_values"]) > 0

    # 4. Predict & Evaluate endpoint
    payload = SimulationInputSchema(
        Q_in=22000.0,
        Temp_water=20.0,
        pH_in=7.3,
        BOD_in=220.0,
        COD_in=460.0,
        TSS_in=240.0,
        TN_in=38.0,
        TP_in=7.0,
        DO_aeration=2.2,
        Blower_speed_pct=70.0,
        RAS_flow=14000.0,
        WAS_flow=240.0,
        MLSS=3200.0,
        PAC_dosage=12.0,
    )
    pred_res = predict_and_evaluate(payload)
    assert "predictions" in pred_res
    assert "compliance" in pred_res
    for tgt in TARGET_COLUMNS:
        assert tgt in pred_res["predictions"]
        assert tgt in pred_res["compliance"]["parameter_audits"]
