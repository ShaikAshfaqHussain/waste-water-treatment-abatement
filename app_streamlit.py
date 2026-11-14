"""EcoVision AI - Wastewater Treatment Pollution Abatement Web Application (Streamlit)

Interactive dashboard featuring:
1. Data Summary & Sensor Cleaning Report
2. Interactive Time-Series & Correlation Heatmaps (Plotly)
3. Multi-Output ML Model Performance & Feature Importances
4. 'What-If' Simulation & Evidence-Based Abatement Advisory Engine
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_generator import generate_wastewater_dataset
from data_cleaning import WastewaterDataCleaner
from ml_model import WastewaterMLPipeline, TARGET_COLUMNS
from rule_engine import WastewaterAbatementEngine, REGULATORY_THRESHOLDS

# Page Configuration
st.set_page_config(
    page_title="EcoVision AI | Wastewater Abatement System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Environmental Engineering Aesthetics
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .alert-critical {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .alert-warning {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .alert-optimal {
        background-color: #f0fdf4;
        border-left: 5px solid #10b981;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    clean_csv = os.path.join(DATA_DIR, "cleaned_wastewater_data.csv")
    raw_csv = os.path.join(DATA_DIR, "raw_wastewater_data.csv")
    rep_json = os.path.join(DATA_DIR, "cleaning_report.json")

    if not os.path.exists(clean_csv):
        from run_pipeline import run_full_pipeline
        run_full_pipeline(output_dir=BASE_DIR)

    cleaned_df = pd.read_csv(clean_csv)
    if "timestamp" in cleaned_df.columns:
        cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"])

    raw_df = pd.read_csv(raw_csv) if os.path.exists(raw_csv) else None
    if raw_df is not None and "timestamp" in raw_df.columns:
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    report = {}
    if os.path.exists(rep_json):
        with open(rep_json, "r") as f:
            report = json.load(f)

    return cleaned_df, raw_df, report


@st.cache_resource
def load_pipeline_and_engine():
    pipeline = WastewaterMLPipeline(model_dir=MODELS_DIR)
    pipeline.load()
    engine = WastewaterAbatementEngine()
    return pipeline, engine


# Load Data and Models
try:
    cleaned_df, raw_df, cleaning_report = load_datasets()
    ml_pipeline, rule_engine = load_pipeline_and_engine()
except Exception as e:
    st.error(f"Initialization Error: {e}. Running end-to-end pipeline...")
    from run_pipeline import run_full_pipeline
    run_full_pipeline(output_dir=BASE_DIR)
    cleaned_df, raw_df, cleaning_report = load_datasets()
    ml_pipeline, rule_engine = load_pipeline_and_engine()

# Application Header
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0; font-size: 2.2rem; display: flex; align-items: center; gap: 12px;">
            💧 EcoVision AI: Wastewater Treatment Abatement Platform
        </h1>
        <p style="margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.9;">
            Intelligent Multi-Target Effluent Prediction (DO, BOD, COD, pH, Turbidity) & Regulatory Abatement Rule Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Tabs
tabs = st.tabs([
    "📊 Data Summary & Cleaning Audit",
    "📈 Time-Series & Correlations",
    "🧠 Multi-Output ML Metrics",
    "🔬 What-If Simulator & Abatement Advisor",
])

# =========================================================================
# TAB 1: DATA SUMMARY & CLEANING REPORT
# =========================================================================
with tabs[0]:
    st.subheader("Data Cleaning & Sensor Telemetry Audit Report")

    # Top KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_records = len(cleaned_df)
    missing_cells = cleaning_report.get("initial_missing_cells", 0)
    quality_score = cleaning_report.get("data_quality_score", 99.5)
    eng_features_cnt = len(cleaning_report.get("engineered_features", []))
    neg_corrections = sum(cleaning_report.get("negative_sensor_corrections", {}).values())

    kpi1.metric("Total Records", f"{total_records:,} hrs")
    kpi2.metric("Telemetry Gaps Imputed", f"{missing_cells} cells")
    kpi3.metric("Negative Drifts Fixed", f"{neg_corrections} events")
    kpi4.metric("Features Engineered", f"{eng_features_cnt} metrics")
    kpi5.metric("Data Quality Score", f"{quality_score:.2f}%")

    st.markdown("---")

    col_audit_left, col_audit_right = st.columns([1, 1])

    with col_audit_left:
        st.markdown("##### 🛠️ Telemetry Data Preprocessing Pipeline")
        st.markdown(
            """
            * **Physical Bounds Validation**: Checked all 20 sensors against physical limits (e.g., $pH \\in [1, 14]$, non-negative concentrations).
            * **Negative Zero-Drift Correction**: Optical sensors (DO, Turbidity) exhibiting baseline negative values were identified and isolated.
            * **Temporal Linear & Rolling Median Imputation**: Forward-interpolated short communication timeouts while using a 7-period rolling median for sustained dropouts.
            * **IQR Robust Winsorization**: Clipped non-physical high-frequency ADC spikes without distorting authentic storm surge hydraulic peaks.
            * **Biochemical Feature Engineering**: Constructed BOD/COD biodegradability ratios, Organic Loading Rates ($OLR$), Food-to-Microorganism ($F/M$) kinetics, and Clarifier Surface Overflow Rates ($SOR$).
            """
        )

        st.markdown("##### 🔍 Sensor Anomaly Flag Breakdown")
        audit_tab_data = []
        imputed_dict = cleaning_report.get("imputed_cells_by_feature", {})
        neg_dict = cleaning_report.get("negative_sensor_corrections", {})
        oob_dict = cleaning_report.get("out_of_bounds_sensor_flags", {})

        all_flagged_keys = sorted(set(list(imputed_dict.keys()) + list(neg_dict.keys()) + list(oob_dict.keys())))
        for k in all_flagged_keys:
            audit_tab_data.append({
                "Sensor Metric": k,
                "Imputed Missing": imputed_dict.get(k, 0),
                "Negative Drifts": neg_dict.get(k, 0),
                "Out of Bounds": oob_dict.get(k, 0),
            })

        if audit_tab_data:
            st.dataframe(pd.DataFrame(audit_tab_data), use_container_width=True, hide_index=True)
        else:
            st.info("No sensor anomalies detected.")

    with col_audit_right:
        st.markdown("##### 🔬 Missing & Drifted Sensor Distribution")
        if audit_tab_data:
            fig_audit = px.bar(
                pd.DataFrame(audit_tab_data),
                x="Sensor Metric",
                y=["Imputed Missing", "Negative Drifts", "Out of Bounds"],
                title="Sensor Corrections & Imputations by Telemetry Channel",
                barmode="stack",
                color_discrete_sequence=["#3b82f6", "#ef4444", "#f59e0b"],
            )
            fig_audit.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=340)
            st.plotly_chart(fig_audit, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📋 Cleaned Telemetry Dataset Preview")
    st.dataframe(cleaned_df.head(8), use_container_width=True)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "⬇️ Download Cleaned WWTP Dataset (CSV)",
            data=cleaned_df.to_csv(index=False),
            file_name="cleaned_wastewater_telemetry.csv",
            mime="text/csv",
        )
    with c_dl2:
        if raw_df is not None:
            st.download_button(
                "⬇️ Download Raw Corrupted Telemetry (CSV)",
                data=raw_df.to_csv(index=False),
                file_name="raw_wastewater_telemetry.csv",
                mime="text/csv",
            )


# =========================================================================
# TAB 2: TIME-SERIES & CORRELATIONS
# =========================================================================
with tabs[1]:
    st.subheader("Interactive Telemetry Time-Series & Correlation Explorer")

    ts_col1, ts_col2 = st.columns([1, 3])
    with ts_col1:
        st.markdown("##### ⚙️ Chart Controls")
        metric_choice = st.selectbox(
            "Select Effluent Metric:",
            ["BOD_eff", "COD_eff", "DO_eff", "pH_eff", "Turbidity_eff"],
            index=0,
        )
        overlay_influent = st.checkbox("Overlay Corresponding Influent", value=True)
        time_window = st.slider("Display Window (Hours):", min_value=72, max_value=len(cleaned_df), value=336, step=24)

    with ts_col2:
        sub_df = cleaned_df.tail(time_window).copy()
        fig_ts = go.Figure()

        fig_ts.add_trace(go.Scatter(
            x=sub_df["timestamp"],
            y=sub_df[metric_choice],
            mode="lines",
            name=f"Effluent {metric_choice}",
            line=dict(color="#10b981", width=2.5),
        ))

        # Add corresponding influent or baseline
        inf_mapping = {"BOD_eff": "BOD_in", "COD_eff": "COD_in", "pH_eff": "pH_in", "DO_eff": "DO_aeration"}
        if overlay_influent and metric_choice in inf_mapping and inf_mapping[metric_choice] in sub_df.columns:
            inf_col = inf_mapping[metric_choice]
            fig_ts.add_trace(go.Scatter(
                x=sub_df["timestamp"],
                y=sub_df[inf_col],
                mode="lines",
                name=f"Influent {inf_col}",
                line=dict(color="#f97316", width=1.5, dash="dot"),
                yaxis="y2" if "pH" not in metric_choice else "y",
            ))

        # Add regulatory cap line
        thresh = REGULATORY_THRESHOLDS.get(metric_choice, {})
        if "normal_max" in thresh:
            fig_ts.add_hline(
                y=thresh["normal_max"],
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Regulatory Cap ({thresh['normal_max']} {thresh['unit']})",
                annotation_position="top right",
            )
        elif "normal_min" in thresh and thresh.get("direction") == "higher_is_better":
            fig_ts.add_hline(
                y=thresh["normal_min"],
                line_dash="dash",
                line_color="#3b82f6",
                annotation_text=f"Minimum Standard ({thresh['normal_min']} {thresh['unit']})",
                annotation_position="bottom right",
            )

        layout_kwargs = dict(
            title=f"Time-Series Trajectory: Effluent {metric_choice} ({time_window} Hours)",
            xaxis_title="Telemetry Timestamp",
            yaxis_title=f"{metric_choice} ({thresh.get('unit', '')})",
            hovermode="x unified",
            height=420,
            margin=dict(l=30, r=30, t=50, b=30),
        )
        if overlay_influent and metric_choice in inf_mapping and "pH" not in metric_choice:
            layout_kwargs["yaxis2"] = dict(
                title=f"Influent ({thresh.get('unit', '')})",
                overlaying="y",
                side="right",
            )

        fig_ts.update_layout(**layout_kwargs)
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    corr_c1, corr_c2 = st.columns([1.3, 1])

    with corr_c1:
        st.markdown("##### 🔀 Pearson Correlation Heatmap")
        corr_cols = [
            "Q_in", "Temp_water", "pH_in", "BOD_in", "COD_in", "TSS_in", "DO_aeration",
            "MLSS", "PAC_dosage", "BOD_COD_ratio_in", "DO_eff", "BOD_eff", "COD_eff", "Turbidity_eff"
        ]
        available_corr = [c for c in corr_cols if c in cleaned_df.columns]
        corr_matrix = cleaned_df[available_corr].corr().round(2)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            title="Operational Parameters vs Effluent Pollutants Correlation",
        )
        fig_corr.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with corr_c2:
        st.markdown("##### 🧪 Biodegradability Index (BOD/COD Ratio)")
        st.markdown(
            """
            The **BOD/COD ratio** measures the biodegradability of wastewater:
            * **> 0.50**: Readily biodegradable municipal sewage.
            * **0.30 - 0.50**: Moderately biodegradable; conventional activated sludge effective.
            * **< 0.30**: Refractory industrial waste or toxic bio-inhibition.
            """
        )
        fig_biodeg = px.histogram(
            cleaned_df,
            x="BOD_COD_ratio_in",
            nbins=35,
            color_discrete_sequence=["#0284c7"],
            title="Influent Biodegradability Ratio Distribution",
            labels={"BOD_COD_ratio_in": "BOD / COD Ratio"},
        )
        fig_biodeg.add_vline(x=0.30, line_dash="dash", line_color="#ef4444", annotation_text="Refractory Limit (<0.30)")
        fig_biodeg.add_vline(x=0.50, line_dash="dash", line_color="#10b981", annotation_text="Readily Biodegradable (>0.50)")
        fig_biodeg.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_biodeg, use_container_width=True)


# =========================================================================
# TAB 3: MULTI-OUTPUT ML MODEL METRICS
# =========================================================================
with tabs[2]:
    st.subheader("Multi-Output Machine Learning Model Evaluation")
    metrics_data = ml_pipeline.metrics

    if metrics_data:
        m_kpi1, m_kpi2, m_kpi3, m_kpi4 = st.columns(4)
        m_kpi1.metric("Model Architecture", "Random Forest Regressor")
        m_kpi2.metric("Overall Mean R²", f"{metrics_data.get('overall_mean_R2', 0.60):.4f}")
        m_kpi3.metric("Training Set Size", f"{metrics_data.get('train_samples', 1200)} samples")
        m_kpi4.metric("Holdout Test Size", f"{metrics_data.get('test_samples', 300)} samples")

        st.markdown("---")
        st.markdown("##### 🎯 Per-Target Holdout Evaluation Table")
        tgt_metrics = metrics_data.get("per_target_metrics", {})
        tgt_table_rows = []
        for tgt, vals in tgt_metrics.items():
            thresh = REGULATORY_THRESHOLDS.get(tgt, {})
            tgt_table_rows.append({
                "Effluent Target": tgt,
                "Unit": thresh.get("unit", ""),
                "R² Score": vals.get("R2_score"),
                "MAE": vals.get("MAE"),
                "RMSE": vals.get("RMSE"),
                "MAPE (%)": f"{vals.get('MAPE_pct')}%",
                "Mean Actual": vals.get("mean_actual"),
                "Mean Predicted": vals.get("mean_predicted"),
            })
        st.dataframe(pd.DataFrame(tgt_table_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        feat_col1, feat_col2 = st.columns([1.2, 1])

        with feat_col1:
            st.markdown("##### 🌟 Global Feature Importance Ranking")
            feat_imp = metrics_data.get("feature_importance_ranking", {})
            if feat_imp:
                top_feats = pd.DataFrame(
                    list(feat_imp.items())[:12],
                    columns=["Operational Feature", "Gini Importance"],
                ).sort_values("Gini Importance", ascending=True)

                fig_imp = px.bar(
                    top_feats,
                    x="Gini Importance",
                    y="Operational Feature",
                    orientation="h",
                    title="Top 12 Features Governing Effluent Pollution Abatement",
                    color="Gini Importance",
                    color_continuous_scale="Viridis",
                )
                fig_imp.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)

        with feat_col2:
            st.markdown("##### 📌 Scientific Validation Principles")
            st.markdown(
                """
                * **Biological Degradation Drivers**: Influent `COD_in` and `BOD_in` along with Hydraulic Retention Time (`HRT_hours`) and `DO_aeration` are the primary governing features for effluent organic carbon removal.
                * **Clarification Mechanics**: Effluent Turbidity is predominantly driven by Clarifier Surface Overflow Rate (`Clarifier_SOR_m_h`) and Coagulant dosage (`PAC_dosage`).
                * **Nitrification & Alkalinity Dynamics**: Effluent `pH_eff` correlates with influent Total Nitrogen (`TN_in`), capturing biological alkalinity consumption during nitrification.
                """
            )
    else:
        st.info("No saved model metrics found. Please train model via pipeline.")


# =========================================================================
# TAB 4: WHAT-IF SIMULATOR & ABATEMENT ADVISOR
# =========================================================================
with tabs[3]:
    st.subheader("🔬 Real-Time 'What-If' Prediction Simulator & Abatement Advisory Engine")
    st.markdown(
        "Adjust plant influent characteristics and operational control setpoints below. "
        "The trained Multi-Output Machine Learning model predicts effluent quality in real-time, "
        "and the environmental rule engine synthesizes actionable abatement protocols."
    )

    # Simulation Presets
    st.markdown("##### ⚡ Quick Operational Scenarios:")
    preset_cols = st.columns(4)
    preset_mode = None

    if preset_cols[0].button("🟢 Normal Nominal Operation", use_container_width=True):
        preset_mode = "nominal"
    if preset_cols[1].button("🌧️ Storm Water Surge", use_container_width=True):
        preset_mode = "storm"
    if preset_cols[2].button("🏭 Toxic Industrial Shock", use_container_width=True):
        preset_mode = "toxic"
    if preset_cols[3].button("⚠️ Blower Trip / Aeration Failure", use_container_width=True):
        preset_mode = "blower_trip"

    # Default preset values
    defaults = {
        "Q_in": 22000.0,
        "Temp_water": 20.0,
        "pH_in": 7.3,
        "BOD_in": 220.0,
        "COD_in": 460.0,
        "TSS_in": 230.0,
        "TN_in": 38.0,
        "TP_in": 7.0,
        "DO_aeration": 2.2,
        "Blower_speed_pct": 70.0,
        "RAS_flow": 14000.0,
        "WAS_flow": 240.0,
        "MLSS": 3200.0,
        "PAC_dosage": 12.0,
    }

    if preset_mode == "storm":
        defaults["Q_in"] = 35000.0
        defaults["BOD_in"] = 150.0
        defaults["COD_in"] = 310.0
        defaults["TSS_in"] = 380.0
        defaults["DO_aeration"] = 1.8
        defaults["PAC_dosage"] = 6.0
    elif preset_mode == "toxic":
        defaults["BOD_in"] = 390.0
        defaults["COD_in"] = 920.0
        defaults["pH_in"] = 5.9
        defaults["TN_in"] = 65.0
        defaults["DO_aeration"] = 1.2
        defaults["PAC_dosage"] = 8.0
    elif preset_mode == "blower_trip":
        defaults["DO_aeration"] = 0.8
        defaults["Blower_speed_pct"] = 45.0
        defaults["BOD_in"] = 280.0
        defaults["COD_in"] = 580.0

    # User Input Control Panels
    with st.expander("🎛️ Adjust Plant Influent & Operational Setpoints", expanded=True):
        p_c1, p_c2, p_c3 = st.columns(3)

        with p_c1:
            st.markdown("**Influent Hydraulic & Thermal Conditions**")
            in_q = st.slider("Influent Flow (Q_in, m³/day):", 10000.0, 40000.0, float(defaults["Q_in"]), 500.0)
            in_temp = st.slider("Water Temperature (°C):", 12.0, 32.0, float(defaults["Temp_water"]), 0.5)
            in_ph = st.slider("Influent pH:", 5.0, 9.5, float(defaults["pH_in"]), 0.1)

        with p_c2:
            st.markdown("**Influent Contaminant Loading**")
            in_bod = st.slider("Influent BOD (mg/L):", 100.0, 500.0, float(defaults["BOD_in"]), 10.0)
            in_cod = st.slider("Influent COD (mg/L):", 200.0, 1000.0, max(float(defaults["COD_in"]), in_bod * 1.2), 10.0)
            in_tss = st.slider("Influent TSS (mg/L):", 80.0, 550.0, float(defaults["TSS_in"]), 10.0)
            in_tn = st.slider("Total Nitrogen (TN_in, mg/L):", 15.0, 80.0, float(defaults["TN_in"]), 1.0)
            in_tp = st.slider("Total Phosphorus (TP_in, mg/L):", 2.0, 20.0, float(defaults["TP_in"]), 0.5)

        with p_c3:
            st.markdown("**Operational Controls & Dosing**")
            op_do_aero = st.slider("Aeration DO Setpoint (mg/L):", 0.5, 4.0, float(defaults["DO_aeration"]), 0.1)
            op_blower = st.slider("Blower Speed (%):", 40.0, 100.0, float(defaults["Blower_speed_pct"]), 1.0)
            op_ras = st.slider("RAS Flow (m³/day):", 5000.0, 28000.0, float(defaults["RAS_flow"]), 500.0)
            op_was = st.slider("WAS Sludge Wasting (m³/day):", 100.0, 500.0, float(defaults["WAS_flow"]), 10.0)
            op_mlss = st.slider("Aeration MLSS (mg/L):", 1800.0, 5000.0, float(defaults["MLSS"]), 50.0)
            op_pac = st.slider("PAC Coagulant Dose (mg/L):", 0.0, 35.0, float(defaults["PAC_dosage"]), 1.0)

    # Assemble Input Dict
    sim_inputs = {
        "Q_in": in_q,
        "Temp_water": in_temp,
        "pH_in": in_ph,
        "BOD_in": in_bod,
        "COD_in": in_cod,
        "TSS_in": in_tss,
        "TN_in": in_tn,
        "TP_in": in_tp,
        "DO_aeration": op_do_aero,
        "Blower_speed_pct": op_blower,
        "RAS_flow": op_ras,
        "WAS_flow": op_was,
        "MLSS": op_mlss,
        "PAC_dosage": op_pac,
    }

    # Execute ML Inference & Rule Evaluation
    predictions = ml_pipeline.predict_from_raw_inputs(sim_inputs)
    compliance = rule_engine.evaluate_compliance(predictions, sim_inputs)

    # Overall Compliance Banner
    st.markdown("---")
    status_str = compliance["overall_status"]
    score = compliance["plant_health_score"]

    if status_str == "CRITICAL_NON_COMPLIANT":
        st.markdown(
            f"""
            <div class="alert-critical">
                <h3 style="margin:0; color:#b91c1c;">🚨 CRITICAL ENVIRONMENTAL EXCEEDANCE (Health Score: {score}/100)</h3>
                <p style="margin:0.4rem 0 0 0; color:#7f1d1d; font-weight:500;">{compliance['summary_statement']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status_str == "WARNING_AT_RISK":
        st.markdown(
            f"""
            <div class="alert-warning">
                <h3 style="margin:0; color:#b45309;">⚠️ OPERATIONAL WARNING: APPROACHING DISCHARGE CAPS (Health Score: {score}/100)</h3>
                <p style="margin:0.4rem 0 0 0; color:#78350f; font-weight:500;">{compliance['summary_statement']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-optimal">
                <h3 style="margin:0; color:#047857;">✅ OPTIMAL COMPLIANT OPERATION (Health Score: {score}/100)</h3>
                <p style="margin:0.4rem 0 0 0; color:#064e3b; font-weight:500;">{compliance['summary_statement']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Predicted Effluent Parameter Cards
    st.markdown("##### 🧪 Predicted Effluent Quality vs Discharge Limits")
    res_c1, res_c2, res_c3, res_c4, res_c5 = st.columns(5)

    audits = compliance["parameter_audits"]

    # DO Card
    do_stat = audits["DO_eff"]["status"]
    do_icon = "🟢" if do_stat == "NORMAL" else ("🟡" if do_stat == "WARNING" else "🔴")
    res_c1.metric(
        f"{do_icon} DO Effluent",
        f"{predictions['DO_eff']:.2f} mg/L",
        delta=f"Limit >= 5.0",
        delta_color="normal" if do_stat == "NORMAL" else "inverse",
    )

    # BOD Card
    bod_stat = audits["BOD_eff"]["status"]
    bod_icon = "🟢" if bod_stat == "NORMAL" else ("🟡" if bod_stat == "WARNING" else "🔴")
    res_c2.metric(
        f"{bod_icon} BOD5 Effluent",
        f"{predictions['BOD_eff']:.1f} mg/L",
        delta=f"Cap <= 30.0",
        delta_color="normal" if bod_stat == "NORMAL" else "inverse",
    )

    # COD Card
    cod_stat = audits["COD_eff"]["status"]
    cod_icon = "🟢" if cod_stat == "NORMAL" else ("🟡" if cod_stat == "WARNING" else "🔴")
    res_c3.metric(
        f"{cod_icon} COD Effluent",
        f"{predictions['COD_eff']:.1f} mg/L",
        delta=f"Cap <= 125.0",
        delta_color="normal" if cod_stat == "NORMAL" else "inverse",
    )

    # pH Card
    ph_stat = audits["pH_eff"]["status"]
    ph_icon = "🟢" if ph_stat == "NORMAL" else ("🟡" if ph_stat == "WARNING" else "🔴")
    res_c4.metric(
        f"{ph_icon} pH Effluent",
        f"{predictions['pH_eff']:.2f}",
        delta=f"Range 6.5 - 8.5",
        delta_color="normal" if ph_stat == "NORMAL" else "inverse",
    )

    # Turbidity Card
    turb_stat = audits["Turbidity_eff"]["status"]
    turb_icon = "🟢" if turb_stat == "NORMAL" else ("🟡" if turb_stat == "WARNING" else "🔴")
    res_c5.metric(
        f"{turb_icon} Turbidity",
        f"{predictions['Turbidity_eff']:.2f} NTU",
        delta=f"Cap <= 10.0",
        delta_color="normal" if turb_stat == "NORMAL" else "inverse",
    )

    # Hydraulic & Kinetic Indicators
    st.markdown("##### ⚙️ Kinetic & Treatment Performance Indicators")
    k_c1, k_c2, k_c3, k_c4 = st.columns(4)
    k_c1.metric("BOD Removal Efficiency", f"{predictions.get('BOD_removal_pct', 0):.1f}%")
    k_c2.metric("COD Removal Efficiency", f"{predictions.get('COD_removal_pct', 0):.1f}%")
    k_c3.metric("Hydraulic Retention (HRT)", f"{predictions.get('HRT_hours', 0):.1f} hours")
    k_c4.metric("Clarifier Overflow (SOR)", f"{predictions.get('Clarifier_SOR_m_h', 0):.2f} m/h")

    # Abatement Action Advisor Cards
    st.markdown("---")
    st.markdown("##### 🛡️ Evidence-Based Abatement Action Recommendations")

    abatement_list = compliance.get("abatement_actions", [])
    if abatement_list:
        for idx, act in enumerate(abatement_list):
            border_col = "#ef4444" if act["level"] == "CRITICAL" else "#f59e0b"
            badge_bg = "#dc2626" if act["level"] == "CRITICAL" else "#d97706"
            st.markdown(
                f"""
                <div style="background:#ffffff; border: 1px solid #e2e8f0; border-left: 6px solid {border_col}; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#1e293b;">
                            Action #{idx+1}: {act['parameter']} (Predicted: {act['predicted_value']} vs Target {act['target_threshold']})
                        </h4>
                        <span style="background:{badge_bg}; color:white; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700;">
                            {act['level']}
                        </span>
                    </div>
                    <p style="margin: 0.5rem 0; color:#475569; font-size: 0.95rem;">
                        <strong>Root Cause Diagnosis:</strong> {act['root_cause']}
                    </p>
                    <div style="background:#f8fafc; border-radius:6px; padding:0.8rem 1rem; margin:0.6rem 0;">
                        <strong style="color:#0f172a;">Prescriptive Engineering & Chemical Interventions:</strong>
                        <ul style="margin: 0.4rem 0 0 1rem; color:#334155; padding-left:0;">
                            {''.join([f"<li>{step}</li>" for step in act['action_steps']])}
                        </ul>
                    </div>
                    <p style="margin: 0.4rem 0 0 0; color:#059669; font-weight:600; font-size:0.9rem;">
                        🎯 Projected Outcome: {act['expected_impact']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("🎉 All effluent metrics satisfy environmental discharge regulations. No active chemical or operational abatement actions required.")
