"""Wastewater Treatment Data Cleaning, Validation, and Feature Engineering Pipeline.

Implements robust physical bounds enforcement, temporal sensor imputation,
statistical anomaly filtering, and biochemical feature engineering.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


PHYSICAL_BOUNDS = {
    "Q_in": (1000.0, 60000.0),
    "Temp_water": (2.0, 45.0),
    "pH_in": (1.0, 14.0),
    "BOD_in": (5.0, 1200.0),
    "COD_in": (10.0, 2500.0),
    "TSS_in": (5.0, 1500.0),
    "TN_in": (1.0, 200.0),
    "TP_in": (0.1, 50.0),
    "DO_aeration": (0.0, 15.0),
    "Blower_speed_pct": (0.0, 100.0),
    "RAS_flow": (0.0, 45000.0),
    "WAS_flow": (0.0, 3000.0),
    "MLSS": (500.0, 9000.0),
    "PAC_dosage": (0.0, 100.0),
    "HRT_hours": (1.0, 48.0),
    "DO_eff": (0.0, 16.0),
    "BOD_eff": (0.5, 300.0),
    "COD_eff": (2.0, 600.0),
    "pH_eff": (1.0, 14.0),
    "Turbidity_eff": (0.0, 250.0),
}


class WastewaterDataCleaner:
    """Production-grade cleaner and feature engineer for wastewater treatment telemetry."""

    def __init__(self, tank_volume_m3: float = 9000.0, clarifier_area_m2: float = 1200.0):
        self.tank_volume_m3 = tank_volume_m3
        self.clarifier_area_m2 = clarifier_area_m2
        self.cleaning_report: Dict[str, Any] = {}

    def clean_and_engineer(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cleans telemetry data, removes anomalies, and constructs domain features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw wastewater telemetry dataset.
        is_training : bool
            Whether dataset includes effluent target columns.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Cleaned dataframe with engineered features and comprehensive cleaning audit report.
        """
        raw_rows, raw_cols = df.shape
        cleaned = df.copy()

        # Ensure no duplicate columns
        cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]

        # Parse and sort by timestamp if present
        if "timestamp" in cleaned.columns:
            cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])
            cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

        # 1. Audit Initial Missingness
        initial_nulls = cleaned.isnull().sum().to_dict()
        total_initial_nulls = sum(initial_nulls.values())

        # 2. Enforce Physical Sensor Boundaries & Flag Negative Readings
        negative_corrections: Dict[str, int] = {}
        out_of_bounds_flags: Dict[str, int] = {}

        for col, (min_val, max_val) in PHYSICAL_BOUNDS.items():
            if col in cleaned.columns:
                # Count and invalidate impossible negative sensor values
                neg_mask = cleaned[col] < 0.0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    negative_corrections[col] = neg_count
                    cleaned.loc[neg_mask, col] = np.nan

                # Count and invalidate severe non-physical values (e.g. pH > 14 or DO > 20)
                bounds_mask = (cleaned[col] < min_val) | (cleaned[col] > max_val)
                bounds_count = int(bounds_mask.sum())
                if bounds_count > 0:
                    out_of_bounds_flags[col] = bounds_count
                    cleaned.loc[bounds_mask, col] = np.nan

        # 3. Handle Missing Sensor Values using Time-Aware & Rolling Median Imputation
        # For numeric telemetry, interpolate temporally then backfill with rolling median
        imputed_counts: Dict[str, int] = {}
        numeric_cols = [c for c in cleaned.select_dtypes(include=[np.number]).columns]

        for col in numeric_cols:
            nans_before = int(cleaned[col].isnull().sum())
            if nans_before > 0:
                # Local interpolation (order 1) for short sensor dropouts
                cleaned[col] = cleaned[col].interpolate(method="linear", limit=4)
                # Rolling window median for sustained dropouts
                rolling_med = cleaned[col].rolling(window=7, min_periods=1, center=True).median()
                cleaned[col] = cleaned[col].fillna(rolling_med)
                # Global median fallback for any boundary edge cases
                col_median = float(cleaned[col].median() if not pd.isna(cleaned[col].median()) else 0.0)
                cleaned[col] = cleaned[col].fillna(col_median)
                imputed_counts[col] = nans_before

        # 4. Statistical Anomaly Filtering (IQR-based Robust Winsorization)
        # Filters artificial sensor electrical spikes while preserving actual storm loads
        clipped_counts: Dict[str, int] = {}
        for col in numeric_cols:
            q25 = float(cleaned[col].quantile(0.25))
            q75 = float(cleaned[col].quantile(0.75))
            iqr = q75 - q25
            if iqr > 1e-4:
                lower_limit = max(PHYSICAL_BOUNDS.get(col, (0, 1e6))[0], q25 - 3.5 * iqr)
                upper_limit = min(PHYSICAL_BOUNDS.get(col, (0, 1e6))[1], q75 + 3.5 * iqr)
                outliers_mask = (cleaned[col] < lower_limit) | (cleaned[col] > upper_limit)
                n_outliers = int(outliers_mask.sum())
                if n_outliers > 0:
                    cleaned[col] = cleaned[col].clip(lower=lower_limit, upper=upper_limit)
                    clipped_counts[col] = n_outliers

        # 5. Biochemical & Environmental Feature Engineering
        # (A) BOD/COD Biodegradability Ratio (Index of treatability)
        # Ensures COD is strictly >= BOD physically
        cleaned["COD_in"] = np.maximum(cleaned["COD_in"], cleaned["BOD_in"] * 1.1)
        cleaned["BOD_COD_ratio_in"] = np.round(cleaned["BOD_in"] / (cleaned["COD_in"] + 1e-5), 3)

        # (B) COD / TN Ratio (Biological Denitrification Carbon Adequacy)
        if "TN_in" in cleaned.columns:
            cleaned["COD_TN_ratio_in"] = np.round(cleaned["COD_in"] / (cleaned["TN_in"] + 1e-5), 2)

        # (C) Organic Loading Rate (OLR, kg COD / m3 * day)
        cleaned["OLR_kg_m3_day"] = np.round(
            (cleaned["Q_in"] * cleaned["COD_in"]) / (self.tank_volume_m3 * 1000.0), 3
        )

        # (D) Food-to-Microorganism (F/M) Ratio (kg BOD / kg MLVSS * day)
        # Standard biological assumption: MLVSS is ~75% of MLSS
        mlvss = cleaned["MLSS"] * 0.75
        cleaned["FM_ratio"] = np.round(
            (cleaned["Q_in"] * cleaned["BOD_in"]) / (self.tank_volume_m3 * mlvss + 1e-5), 3
        )

        # (E) Recalculate Hydraulic Retention Time (HRT hours) if not provided or to ensure fidelity
        cleaned["HRT_hours"] = np.round((self.tank_volume_m3 / (cleaned["Q_in"] + 1e-5)) * 24.0, 2)

        # (F) Clarifier Surface Overflow Rate (SOR, m/h)
        cleaned["Clarifier_SOR_m_h"] = np.round(
            cleaned["Q_in"] / (self.clarifier_area_m2 * 24.0), 3
        )

        # (G) Theoretical Water Oxygen Saturation & Aeration Deficit (mg/L)
        temp_c = cleaned["Temp_water"]
        do_saturation = 14.652 - 0.41022 * temp_c + 0.007991 * (temp_c ** 2)
        cleaned["DO_saturation_theoretical"] = np.round(do_saturation, 2)
        cleaned["DO_deficit"] = np.round(
            np.maximum(0.0, cleaned["DO_saturation_theoretical"] - cleaned["DO_aeration"]), 2
        )

        # (H) Temperature Kinetics Factor (Arrhenius correction theta^(T - 20))
        cleaned["Arrhenius_factor"] = np.round(1.047 ** (cleaned["Temp_water"] - 20.0), 3)

        # (I) Removal Efficiencies (if targets exist in dataset)
        if is_training and "BOD_eff" in cleaned.columns and "COD_eff" in cleaned.columns:
            cleaned["BOD_removal_eff_pct"] = np.round(
                ((cleaned["BOD_in"] - cleaned["BOD_eff"]) / (cleaned["BOD_in"] + 1e-5)) * 100.0, 2
            ).clip(0.0, 100.0)
            cleaned["COD_removal_eff_pct"] = np.round(
                ((cleaned["COD_in"] - cleaned["COD_eff"]) / (cleaned["COD_in"] + 1e-5)) * 100.0, 2
            ).clip(0.0, 100.0)

        # 6. Construct Comprehensive Audit Report
        final_nulls = int(cleaned.isnull().sum().sum())
        self.cleaning_report = {
            "initial_records": raw_rows,
            "final_records": len(cleaned),
            "initial_missing_cells": total_initial_nulls,
            "final_missing_cells": final_nulls,
            "negative_sensor_corrections": negative_corrections,
            "out_of_bounds_sensor_flags": out_of_bounds_flags,
            "imputed_cells_by_feature": imputed_counts,
            "clipped_outliers_by_feature": clipped_counts,
            "engineered_features": [
                "BOD_COD_ratio_in",
                "COD_TN_ratio_in",
                "OLR_kg_m3_day",
                "FM_ratio",
                "HRT_hours",
                "Clarifier_SOR_m_h",
                "DO_saturation_theoretical",
                "DO_deficit",
                "Arrhenius_factor",
            ],
            "data_quality_score": round((1.0 - (total_initial_nulls / (raw_rows * raw_cols + 1e-5))) * 100, 2),
        }

        return cleaned, self.cleaning_report


if __name__ == "__main__":
    from data_generator import generate_wastewater_dataset

    raw_df = generate_wastewater_dataset(n_samples=500, inject_anomalies=True)
    cleaner = WastewaterDataCleaner()
    clean_df, report = cleaner.clean_and_engineer(raw_df, is_training=True)

    print("Cleaning Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nEngineered Cleaned Shape: {clean_df.shape}")
    print("New Engineered Columns:", [c for c in clean_df.columns if c not in raw_df.columns])
