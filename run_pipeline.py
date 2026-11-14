"""End-to-End Wastewater Treatment Pollution Abatement Pipeline.

Executes:
1. Synthetic / Realistic WWTP Dataset Generation (1,500 hourly telemetry records)
2. Production Data Cleaning, Sensor Anomaly Correction, and Feature Engineering
3. Multi-Output Machine Learning Model Training (DO, BOD, COD, pH, Turbidity)
4. Comprehensive Model Evaluation & Metric Persistence
5. Environmental Compliance Rule Engine Verification
"""

import json
import os
import sys

# Ensure local src directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_generator import generate_wastewater_dataset
from data_cleaning import WastewaterDataCleaner
from ml_model import WastewaterMLPipeline
from rule_engine import WastewaterAbatementEngine


def run_full_pipeline(
    n_samples: int = 1500,
    output_dir: str = CURRENT_DIR,
):
    print("=" * 75)
    print("ECOVISION AI: WASTEWATER TREATMENT POLLUTION ABATEMENT PIPELINE")
    print("=" * 75)

    data_dir = os.path.join(output_dir, "data")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Generate Realistic WWTP Telemetry Dataset
    # -------------------------------------------------------------
    print(f"\n[Step 1/5] Generating {n_samples} realistic WWTP telemetry records...")
    raw_df = generate_wastewater_dataset(
        n_samples=n_samples,
        random_seed=42,
        inject_anomalies=True,
        anomaly_rate=0.045,
    )
    raw_path = os.path.join(data_dir, "raw_wastewater_data.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f" -> Saved raw dataset to: {raw_path}")
    print(f" -> Shape: {raw_df.shape} | Null cells: {raw_df.isnull().sum().sum()}")

    # -------------------------------------------------------------
    # Step 2: Clean Telemetry & Engineer Biochemical Features
    # -------------------------------------------------------------
    print("\n[Step 2/5] Sanitizing sensor errors and engineering biochemical features...")
    cleaner = WastewaterDataCleaner(tank_volume_m3=9000.0, clarifier_area_m2=1200.0)
    cleaned_df, cleaning_report = cleaner.clean_and_engineer(raw_df, is_training=True)

    cleaned_path = os.path.join(data_dir, "cleaned_wastewater_data.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    report_path = os.path.join(data_dir, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(cleaning_report, f, indent=2)

    print(f" -> Saved cleaned dataset to: {cleaned_path}")
    print(f" -> Saved cleaning audit report to: {report_path}")
    print(f" -> Quality Score: {cleaning_report['data_quality_score']}%")
    print(f" -> Features Engineered: {len(cleaning_report['engineered_features'])} features")
    print(f"    {', '.join(cleaning_report['engineered_features'][:5])}...")

    # -------------------------------------------------------------
    # Step 3: Multi-Output Machine Learning Model Training
    # -------------------------------------------------------------
    print("\n[Step 3/5] Training Multi-Output Random Forest Regressor on 5 effluent targets...")
    pipeline = WastewaterMLPipeline(model_dir=models_dir)
    metrics, holdout_df = pipeline.train_and_evaluate(cleaned_df, test_size=0.2, random_state=42)
    pipeline.save()
    print(" -> Model artifacts serialized to models/ directory.")

    # -------------------------------------------------------------
    # Step 4: Model Evaluation Summary
    # -------------------------------------------------------------
    print("\n[Step 4/5] Multi-Output Regression Performance Report:")
    print("-" * 75)
    print(f"{'Target Variable':<18} | {'R² Score':<9} | {'MAE':<8} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 75)
    for target, tgt_metrics in metrics["per_target_metrics"].items():
        print(
            f"{target:<18} | {tgt_metrics['R2_score']:<9.4f} | {tgt_metrics['MAE']:<8.3f} | {tgt_metrics['RMSE']:<8.3f} | {tgt_metrics['MAPE_pct']:<8.2f}%"
        )
    print("-" * 75)
    print(f"Overall Mean R² Score: {metrics['overall_mean_R2']:.4f}")
    print(f"Training observations: {metrics['train_samples']} | Holdout test samples: {metrics['test_samples']}")

    # -------------------------------------------------------------
    # Step 5: Anomaly Detection Rule Engine Demonstration
    # -------------------------------------------------------------
    print("\n[Step 5/5] Testing Rule Engine Abatement Protocol on Holdout Outlier...")
    engine = WastewaterAbatementEngine()
    # Sample extreme holdout scenario
    sample_inputs = {
        "Q_in": 34000.0,       # High hydraulic surge
        "Temp_water": 24.5,
        "pH_in": 6.1,          # Acidic shock
        "BOD_in": 380.0,       # Heavy organic load
        "COD_in": 850.0,
        "TSS_in": 440.0,
        "TN_in": 58.0,
        "TP_in": 12.0,
        "DO_aeration": 1.1,    # Depleted aeration
        "Blower_speed_pct": 55.0,
        "RAS_flow": 12000.0,
        "WAS_flow": 280.0,
        "MLSS": 2600.0,
        "PAC_dosage": 6.0,     # Underdosed coagulant
    }
    preds = pipeline.predict_from_raw_inputs(sample_inputs)
    compliance = engine.evaluate_compliance(preds, sample_inputs)

    print("\nSimulator Predictions:")
    for tgt in ["DO_eff", "BOD_eff", "COD_eff", "pH_eff", "Turbidity_eff"]:
        status = compliance["parameter_audits"][tgt]["status"]
        val = preds[tgt]
        unit = compliance["parameter_audits"][tgt]["unit"]
        print(f"  * {tgt:<15}: {val:.2f} {unit:<6} [{status}]")

    print(f"\nOverall Plant Compliance: [{compliance['overall_status']}]")
    print(f"Plant Health Score: {compliance['plant_health_score']}/100")
    print(f"Abatement Interventions Generated: {len(compliance['abatement_actions'])}")
    for act in compliance["abatement_actions"][:2]:
        print(f"  - [{act['level']}] {act['parameter']}: {act['action_steps'][0]}")

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETE & READY FOR WEB DASHBOARD")
    print("=" * 75)


if __name__ == "__main__":
    run_full_pipeline()
