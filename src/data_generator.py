"""Wastewater Treatment Plant (WWTP) Realistic Dataset Generator.

Simulates authentic influent telemetry, operational control parameters,
and biological/chemical kinetics based on the Activated Sludge Model (ASM)
principles and the UCI Water Treatment Plant benchmark structure.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_wastewater_dataset(
    n_samples: int = 1500,
    random_seed: int = 42,
    inject_anomalies: bool = True,
    anomaly_rate: float = 0.04,
) -> pd.DataFrame:
    """Generates a realistic multi-parameter wastewater treatment dataset.

    Parameters
    ----------
    n_samples : int
        Number of hourly operational records (default 1500, ~62 days).
    random_seed : int
        Seed for numerical reproducibility.
    inject_anomalies : bool
        Whether to inject sensor faults, missing values, and negative readings.
    anomaly_rate : float
        Fraction of samples corrupted with realistic sensor anomalies.

    Returns
    -------
    pd.DataFrame
        Complete WWTP time-series dataset with influent, operational, and effluent parameters.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Temporal Framework (Hourly telemetry starting 60 days prior)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    hours = np.array([ts.hour for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    day_index = np.arange(n_samples) / 24.0

    # 2. Environmental & Ambient Influences
    # Diurnal temperature cycle + seasonal drift (14 - 26 deg C)
    seasonal_temp = 18.0 + 5.0 * np.sin(2 * np.pi * day_index / 365.0)
    diurnal_temp = 2.5 * np.sin(2 * np.pi * (hours - 14) / 24.0)
    water_temp = seasonal_temp + diurnal_temp + rng.normal(0, 0.4, n_samples)
    water_temp = np.clip(water_temp, 11.0, 32.0)

    # 3. Influent Hydraulic Flow Rate (Q_in, m3/day)
    # Diurnal twin peaks: morning peak (8-10 AM), evening peak (19-21 PM), night lull (2-4 AM)
    diurnal_flow = 3500.0 * np.sin(2 * np.pi * (hours - 8) / 24.0) + \
                   1800.0 * np.sin(4 * np.pi * (hours - 9) / 24.0)
    weekend_factor = np.where(day_of_week >= 5, 0.88, 1.0)
    base_flow = 22000.0 * weekend_factor + diurnal_flow + rng.normal(0, 750, n_samples)

    # Simulate wet-weather/storm infiltration surge events (~3% of days)
    storm_mask = rng.random(n_samples) < 0.025
    flow_surge = np.where(storm_mask, rng.uniform(6000, 14000, n_samples), 0.0)
    q_in = np.clip(base_flow + flow_surge, 11000.0, 38000.0)

    # 4. Influent Contaminant Concentrations
    # Dilution effect occurs during storm events
    dilution_ratio = 22000.0 / q_in

    # Biochemical Oxygen Demand (BOD_in, mg/L)
    bod_in_base = 230.0 + 40.0 * np.sin(2 * np.pi * (hours - 10) / 24.0)
    bod_in = (bod_in_base * (0.8 + 0.2 * dilution_ratio) + rng.normal(0, 18, n_samples))
    bod_in = np.clip(bod_in, 110.0, 460.0)

    # Chemical Oxygen Demand (COD_in, mg/L)
    # COD is stoichiometrically higher than BOD (biodegradability ratio 0.42 - 0.58)
    biodegradability_base = rng.uniform(0.44, 0.55, n_samples)
    cod_in = bod_in / biodegradability_base + rng.normal(0, 25, n_samples)
    cod_in = np.clip(cod_in, bod_in * 1.35, 950.0)

    # Total Suspended Solids (TSS_in, mg/L)
    tss_in = 0.55 * cod_in + rng.normal(0, 20, n_samples)
    tss_in = np.clip(tss_in, 100.0, 520.0)

    # Total Nitrogen (TN_in, mg/L) and Total Phosphorus (TP_in, mg/L)
    tn_in = 0.16 * bod_in + rng.normal(0, 3.5, n_samples)
    tn_in = np.clip(tn_in, 20.0, 75.0)

    tp_in = 0.032 * bod_in + rng.normal(0, 0.7, n_samples)
    tp_in = np.clip(tp_in, 3.5, 16.0)

    # Influent pH (typically 6.8 - 7.9 with periodic industrial acid/base pulses)
    ph_in_base = 7.35 + 0.15 * np.sin(2 * np.pi * hours / 24.0)
    ph_in = ph_in_base + rng.normal(0, 0.22, n_samples)
    # Rare industrial acid discharge (pH drop) or caustic discharge
    ind_spikes = rng.choice([0, -1.2, 1.1], size=n_samples, p=[0.97, 0.015, 0.015])
    ph_in = np.clip(ph_in + ind_spikes, 5.2, 9.4)

    # 5. Plant Operational Control Parameters
    # Aeration Tank Volume = 9000 m3
    aeration_volume_m3 = 9000.0
    hrt_hours = (aeration_volume_m3 / q_in) * 24.0

    # Aeration Basin DO setpoint (mg/L): target 2.2 mg/L, feedback control with blower
    # High influent COD load drops aeration DO unless blower compensates
    blower_speed = 65.0 + 0.035 * (cod_in - 450.0) + rng.normal(0, 4.0, n_samples)
    blower_speed = np.clip(blower_speed, 45.0, 100.0)

    do_aeration = 1.2 + (blower_speed / 100.0) * 2.1 - 0.0012 * (cod_in - 400.0)
    do_aeration = np.clip(do_aeration + rng.normal(0, 0.12, n_samples), 0.7, 3.9)

    # Return Activated Sludge (RAS) Recycle Ratio (0.45 - 0.95)
    ras_ratio = 0.65 - 0.05 * (q_in / 22000.0) + rng.normal(0, 0.04, n_samples)
    ras_ratio = np.clip(ras_ratio, 0.40, 1.05)
    ras_flow = q_in * ras_ratio

    # Mixed Liquor Suspended Solids (MLSS, mg/L in aeration basin)
    mlss = 2800.0 + 750.0 * ras_ratio + rng.normal(0, 120, n_samples)
    mlss = np.clip(mlss, 2100.0, 4800.0)

    # Waste Activated Sludge (WAS, m3/day) controlling SRT
    was_flow = 220.0 + 0.003 * q_in + rng.normal(0, 25, n_samples)

    # Poly-Aluminum Chloride Coagulant Dosage (PAC, mg/L)
    pac_dosage = 12.0 + 0.02 * (tss_in - 240.0) + rng.normal(0, 2.0, n_samples)
    pac_dosage = np.clip(pac_dosage, 3.0, 28.0)

    # 6. Effluent Target Kinetics (Activated Sludge Biological & Settling Models)
    # Arrhenius temperature correction factor: theta^(T - 20)
    theta_biochem = 1.047 ** (water_temp - 20.0)

    # Food to Microorganism (F/M) ratio (kg BOD / kg MLVSS * day)
    # MLVSS is ~0.75 of MLSS
    fm_ratio = (q_in * bod_in) / (aeration_volume_m3 * (mlss * 0.75) + 1e-5)

    # (A) Effluent BOD (BOD_eff, mg/L)
    # Substrate removal: high HRT, optimal DO (>2.0), healthy MLSS, optimal F/M (0.2-0.4) yield low BOD
    do_limitation = np.clip(do_aeration / (do_aeration + 0.3), 0.5, 1.0)
    kinetics_removal = 0.925 * (1.0 - np.exp(-0.28 * hrt_hours * theta_biochem * do_limitation))
    # Penalize extreme F/M shock overload (> 0.45)
    overload_penalty = np.where(fm_ratio > 0.42, (fm_ratio - 0.42) * 55.0, 0.0)
    bod_eff = bod_in * (1.0 - kinetics_removal) + overload_penalty + rng.normal(0, 1.6, n_samples)
    bod_eff = np.clip(bod_eff, 4.0, 65.0)

    # (B) Effluent COD (COD_eff, mg/L)
    # Non-biodegradable soluble inert COD (S_I ~ 8% of influent COD) cannot be biologically oxidized
    inert_cod = 0.075 * cod_in
    biodegradable_residual_cod = bod_eff * rng.uniform(1.65, 1.95, n_samples)
    cod_eff = inert_cod + biodegradable_residual_cod + rng.normal(0, 3.5, n_samples)
    cod_eff = np.clip(cod_eff, bod_eff * 1.5, 155.0)

    # (C) Effluent Dissolved Oxygen (DO_eff, mg/L)
    # Post-aeration weir cascades re-aerate water toward oxygen saturation,
    # but residual respiration demand lowers DO if BOD_eff is high or water is hot
    do_sat = 14.652 - 0.41022 * water_temp + 0.007991 * (water_temp ** 2)
    reoxygenation = 0.65 * do_aeration + 0.35 * (do_sat * 0.75)
    do_eff = reoxygenation - 0.045 * bod_eff + rng.normal(0, 0.25, n_samples)
    do_eff = np.clip(do_eff, 1.8, 8.8)

    # (D) Effluent pH (pH_eff)
    # Biological nitrification consumes alkalinity (~7.14 mg CaCO3 per mg NH4 oxidized),
    # dropping pH slightly. Coagulant PAC also reduces pH slightly.
    alkalinity_drop = 0.008 * (tn_in - 15.0) + 0.005 * pac_dosage
    ph_eff = ph_in - alkalinity_drop + rng.normal(0, 0.12, n_samples)
    ph_eff = np.clip(ph_eff, 6.1, 8.8)

    # (E) Effluent Turbidity (Turbidity_eff, NTU)
    # Governed by secondary clarifier surface overflow rate (SOR = Q_in / Clarifier_Area)
    # Clarifier Area = 1200 m2
    clarifier_area = 1200.0
    sor = q_in / (clarifier_area * 24.0)  # m/hour
    pac_benefit = 1.0 - (pac_dosage / 40.0)
    turbidity_base = 3.2 * (sor / 0.75) * pac_benefit
    # High hydraulic surge or low HRT increases floc shearing / pin-point floc carryover
    clarifier_stress = np.where(sor > 0.95, (sor - 0.95) * 16.0, 0.0)
    turbidity_eff = turbidity_base + clarifier_stress + rng.normal(0, 0.6, n_samples)
    turbidity_eff = np.clip(turbidity_eff, 1.2, 28.0)

    # Assemble base Clean Ground-Truth DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "Q_in": np.round(q_in, 1),
        "Temp_water": np.round(water_temp, 2),
        "pH_in": np.round(ph_in, 2),
        "BOD_in": np.round(bod_in, 1),
        "COD_in": np.round(cod_in, 1),
        "TSS_in": np.round(tss_in, 1),
        "TN_in": np.round(tn_in, 2),
        "TP_in": np.round(tp_in, 2),
        "DO_aeration": np.round(do_aeration, 2),
        "Blower_speed_pct": np.round(blower_speed, 1),
        "RAS_flow": np.round(ras_flow, 1),
        "WAS_flow": np.round(was_flow, 1),
        "MLSS": np.round(mlss, 1),
        "PAC_dosage": np.round(pac_dosage, 2),
        "HRT_hours": np.round(hrt_hours, 2),
        # Effluent Targets to Predict
        "DO_eff": np.round(do_eff, 2),
        "BOD_eff": np.round(bod_eff, 1),
        "COD_eff": np.round(cod_eff, 1),
        "pH_eff": np.round(ph_eff, 2),
        "Turbidity_eff": np.round(turbidity_eff, 2),
    })

    if not inject_anomalies:
        return df

    # 7. Inject Authentic Industrial Sensor Faults & Missing Values
    corrupted_df = df.copy()
    n_corrupt = int(n_samples * anomaly_rate)
    corrupt_indices = rng.choice(n_samples, size=n_corrupt, replace=False)

    for idx in corrupt_indices:
        anomaly_mode = rng.choice(["missing", "negative_sensor", "stuck_reading", "extreme_spike"])

        if anomaly_mode == "missing":
            # Missing sensor packet / telemetry dropout
            target_cols = rng.choice(
                ["DO_aeration", "BOD_in", "pH_in", "Turbidity_eff", "COD_in", "DO_eff"],
                size=2,
                replace=False,
            )
            for t_col in target_cols:
                corrupted_df.at[idx, t_col] = np.nan

        elif anomaly_mode == "negative_sensor":
            # Optical zero-drift yielding impossible negative values
            col = rng.choice(["Turbidity_eff", "DO_eff", "DO_aeration", "PAC_dosage"])
            corrupted_df.loc[idx, col] = -1.0 * abs(rng.uniform(0.5, 4.5))

        elif anomaly_mode == "stuck_reading":
            # Sensor ADC saturation / biofouling lock (e.g. pH=0.0 or DO=99.9)
            col = rng.choice(["pH_in", "pH_eff", "DO_eff"])
            if "pH" in col:
                corrupted_df.loc[idx, col] = rng.choice([0.0, 14.0])
            else:
                corrupted_df.loc[idx, col] = 99.9

        elif anomaly_mode == "extreme_spike":
            # Extreme electrical interference transient
            col = rng.choice(["COD_in", "BOD_in", "Turbidity_eff", "TSS_in"])
            corrupted_df.loc[idx, col] = corrupted_df.loc[idx, col] * rng.uniform(4.0, 7.0)

    # Also sprinkle small random dropouts (0.8% missing across columns)
    for col in ["Q_in", "Temp_water", "TN_in", "MLSS", "BOD_eff", "COD_eff"]:
        mask_drop = rng.random(n_samples) < 0.008
        corrupted_df.loc[mask_drop, col] = np.nan

    return corrupted_df


if __name__ == "__main__":
    df = generate_wastewater_dataset(n_samples=1500, inject_anomalies=True)
    print(f"Generated {len(df)} records with shape {df.shape}")
    print("Null count:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nSummary Head:\n", df.head(3))
