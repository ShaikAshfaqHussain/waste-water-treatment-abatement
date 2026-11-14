"""Multi-Output Machine Learning Model for Wastewater Effluent Prediction.

Uses Scikit-learn Random Forest Regressor to predict:
- Dissolved Oxygen (DO_eff)
- Biochemical Oxygen Demand (BOD_eff)
- Chemical Oxygen Demand (COD_eff)
- Effluent pH (pH_eff)
- Effluent Turbidity (Turbidity_eff)
"""

import json
import os
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMNS = ["DO_eff", "BOD_eff", "COD_eff", "pH_eff", "Turbidity_eff"]

FEATURE_COLUMNS = [
    "Q_in",
    "Temp_water",
    "pH_in",
    "BOD_in",
    "COD_in",
    "TSS_in",
    "TN_in",
    "TP_in",
    "DO_aeration",
    "Blower_speed_pct",
    "RAS_flow",
    "WAS_flow",
    "MLSS",
    "PAC_dosage",
    "HRT_hours",
    "BOD_COD_ratio_in",
    "COD_TN_ratio_in",
    "OLR_kg_m3_day",
    "FM_ratio",
    "Clarifier_SOR_m_h",
    "DO_deficit",
    "Arrhenius_factor",
]


class WastewaterMLPipeline:
    """Manages training, evaluation, persistence, and live inference for multi-target effluent prediction."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model: RandomForestRegressor = None
        self.scaler: StandardScaler = StandardScaler()
        self.metrics: Dict[str, Any] = {}
        self.feature_importances: Dict[str, float] = {}

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Trains multi-output Random Forest and evaluates per-target metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned wastewater dataset with engineered features and target columns.
        test_size : float
            Fraction of data reserved for evaluation.
        random_state : int
            Seed for deterministic splitting and training.

        Returns
        -------
        Tuple[Dict[str, Any], pd.DataFrame]
            Evaluation metrics dict and test set holdout predictions DataFrame.
        """
        # Ensure all required features and targets exist
        available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
        X = df[available_features].values
        y = df[TARGET_COLUMNS].values

        # Split into training and holdout test set (chronological or shuffled)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Scale input features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Multi-Output Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=160,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train_scaled, y_train)

        # Generate holdout predictions
        y_pred = self.model.predict(X_test_scaled)

        # Compute comprehensive metrics per effluent target
        per_target_metrics: Dict[str, Dict[str, float]] = {}
        r2_scores: List[float] = []

        for idx, target in enumerate(TARGET_COLUMNS):
            y_true_col = y_test[:, idx]
            y_pred_col = y_pred[:, idx]

            r2 = float(r2_score(y_true_col, y_pred_col))
            mae = float(mean_absolute_error(y_true_col, y_pred_col))
            rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
            # Mean absolute percentage error avoiding divide by zero
            mape = float(np.mean(np.abs((y_true_col - y_pred_col) / (np.abs(y_true_col) + 1e-4))) * 100.0)

            per_target_metrics[target] = {
                "R2_score": round(r2, 4),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "MAPE_pct": round(mape, 2),
                "mean_actual": round(float(np.mean(y_true_col)), 2),
                "mean_predicted": round(float(np.mean(y_pred_col)), 2),
            }
            r2_scores.append(r2)

        # Extract and rank global feature importances
        raw_importances = self.model.feature_importances_
        sorted_indices = np.argsort(raw_importances)[::-1]
        self.feature_importances = {
            available_features[i]: round(float(raw_importances[i]), 4)
            for i in sorted_indices
        }

        # Build overall model summary
        self.metrics = {
            "model_type": "Multi-Output Random Forest Regressor",
            "n_estimators": 160,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "overall_mean_R2": round(float(np.mean(r2_scores)), 4),
            "targets": TARGET_COLUMNS,
            "features": available_features,
            "per_target_metrics": per_target_metrics,
            "feature_importance_ranking": self.feature_importances,
        }

        # Construct holdout comparison DataFrame
        holdout_df = pd.DataFrame()
        for idx, target in enumerate(TARGET_COLUMNS):
            holdout_df[f"{target}_actual"] = np.round(y_test[:, idx], 2)
            holdout_df[f"{target}_predicted"] = np.round(y_pred[:, idx], 2)
            holdout_df[f"{target}_residual"] = np.round(y_test[:, idx] - y_pred[:, idx], 2)

        return self.metrics, holdout_df

    def save(self):
        """Serializes model, scaler, and evaluation artifacts to disk."""
        joblib.dump(self.model, os.path.join(self.model_dir, "rf_multioutput_model.joblib"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        with open(os.path.join(self.model_dir, "model_metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        with open(os.path.join(self.model_dir, "feature_importance.json"), "w") as f:
            json.dump(self.feature_importances, f, indent=2)
        with open(os.path.join(self.model_dir, "feature_names.json"), "w") as f:
            json.dump(FEATURE_COLUMNS, f, indent=2)

    def load(self):
        """Loads trained weights, scaler, and metadata from disk."""
        model_path = os.path.join(self.model_dir, "rf_multioutput_model.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        metrics_path = os.path.join(self.model_dir, "model_metrics.json")
        importance_path = os.path.join(self.model_dir, "feature_importance.json")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model artifacts not found in {self.model_dir}. Please train model first.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

        if os.path.exists(importance_path):
            with open(importance_path, "r") as f:
                self.feature_importances = json.load(f)

    def predict_from_raw_inputs(self, raw_inputs: Dict[str, float]) -> Dict[str, float]:
        """Engineers required features from raw simulator inputs and predicts all 5 effluent metrics.

        Parameters
        ----------
        raw_inputs : Dict[str, float]
            Dictionary containing plant operational parameters:
            - Q_in (m3/day)
            - Temp_water (deg C)
            - pH_in
            - BOD_in (mg/L)
            - COD_in (mg/L)
            - TSS_in (mg/L)
            - TN_in (mg/L)
            - TP_in (mg/L)
            - DO_aeration (mg/L)
            - Blower_speed_pct (%)
            - RAS_flow (m3/day)
            - WAS_flow (m3/day)
            - MLSS (mg/L)
            - PAC_dosage (mg/L)
            - HRT_hours (optional, recomputed if absent)

        Returns
        -------
        Dict[str, float]
            Predictions for DO_eff, BOD_eff, COD_eff, pH_eff, Turbidity_eff.
        """
        if self.model is None:
            self.load()

        tank_volume_m3 = 9000.0
        clarifier_area_m2 = 1200.0

        q_in = float(raw_inputs.get("Q_in", 22000.0))
        temp = float(raw_inputs.get("Temp_water", 20.0))
        ph_in = float(raw_inputs.get("pH_in", 7.2))
        bod_in = float(raw_inputs.get("BOD_in", 220.0))
        cod_in = float(raw_inputs.get("COD_in", 480.0))
        tss_in = float(raw_inputs.get("TSS_in", 240.0))
        tn_in = float(raw_inputs.get("TN_in", 40.0))
        tp_in = float(raw_inputs.get("TP_in", 7.5))
        do_aeration = float(raw_inputs.get("DO_aeration", 2.2))
        blower_speed = float(raw_inputs.get("Blower_speed_pct", 70.0))
        ras_flow = float(raw_inputs.get("RAS_flow", 14000.0))
        was_flow = float(raw_inputs.get("WAS_flow", 250.0))
        mlss = float(raw_inputs.get("MLSS", 3200.0))
        pac_dosage = float(raw_inputs.get("PAC_dosage", 12.0))

        # Dynamic Biochemical Feature Engineering
        hrt_hours = (tank_volume_m3 / (q_in + 1e-5)) * 24.0
        bod_cod_ratio = round(bod_in / (cod_in + 1e-5), 3)
        cod_tn_ratio = round(cod_in / (tn_in + 1e-5), 2)
        olr = round((q_in * cod_in) / (tank_volume_m3 * 1000.0), 3)
        mlvss = mlss * 0.75
        fm_ratio = round((q_in * bod_in) / (tank_volume_m3 * mlvss + 1e-5), 3)
        clarifier_sor = round(q_in / (clarifier_area_m2 * 24.0), 3)

        temp_c = temp
        do_sat = 14.652 - 0.41022 * temp_c + 0.007991 * (temp_c ** 2)
        do_deficit = round(max(0.0, do_sat - do_aeration), 2)
        arrhenius = round(1.047 ** (temp - 20.0), 3)

        feature_vector = np.array([
            q_in,
            temp,
            ph_in,
            bod_in,
            cod_in,
            tss_in,
            tn_in,
            tp_in,
            do_aeration,
            blower_speed,
            ras_flow,
            was_flow,
            mlss,
            pac_dosage,
            hrt_hours,
            bod_cod_ratio,
            cod_tn_ratio,
            olr,
            fm_ratio,
            clarifier_sor,
            do_deficit,
            arrhenius,
        ]).reshape(1, -1)

        # Scale features using fitted scaler
        feature_vector_scaled = self.scaler.transform(feature_vector)
        pred = self.model.predict(feature_vector_scaled)[0]

        return {
            "DO_eff": round(float(pred[0]), 2),
            "BOD_eff": round(float(pred[1]), 2),
            "COD_eff": round(float(pred[2]), 2),
            "pH_eff": round(float(pred[3]), 2),
            "Turbidity_eff": round(float(pred[4]), 2),
            # Helper removal estimations
            "BOD_removal_pct": round(float(max(0.0, min(99.5, ((bod_in - pred[1]) / (bod_in + 1e-5)) * 100.0))), 1),
            "COD_removal_pct": round(float(max(0.0, min(99.0, ((cod_in - pred[2]) / (cod_in + 1e-5)) * 100.0))), 1),
            "OLR_kg_m3_day": olr,
            "FM_ratio": fm_ratio,
            "HRT_hours": round(hrt_hours, 1),
            "Clarifier_SOR_m_h": clarifier_sor,
        }


if __name__ == "__main__":
    from data_generator import generate_wastewater_dataset
    from data_cleaning import WastewaterDataCleaner

    raw_df = generate_wastewater_dataset(n_samples=1000, inject_anomalies=True)
    cleaner = WastewaterDataCleaner()
    clean_df, _ = cleaner.clean_and_engineer(raw_df, is_training=True)

    pipeline = WastewaterMLPipeline(model_dir="../models")
    metrics, holdout = pipeline.train_and_evaluate(clean_df)

    print("Overall Mean R2:", metrics["overall_mean_R2"])
    print("\nPer-Target Metrics:")
    for tgt, met in metrics["per_target_metrics"].items():
        print(f"  {tgt}: R2={met['R2_score']}, MAE={met['MAE']}, RMSE={met['RMSE']}, MAPE={met['MAPE_pct']}%")

    print("\nTop 5 Important Features:")
    for feat, imp in list(metrics["feature_importance_ranking"].items())[:5]:
        print(f"  {feat}: {imp}")
