# 💧 EcoVision AI: Wastewater Treatment Pollution Abatement Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Regulatory Standards](https://img.shields.io/badge/Standards-EPA%20%7C%20EU%20%7C%20CPCB-10b981.svg)]()
[![Tests](https://img.shields.io/badge/Pytest-6%20Passed-brightgreen.svg)]()

> **Senior Full-Stack & Environmental Data Science Project**  
> An intelligent telemetry sanitization, multi-output machine learning, and evidence-based pollution abatement platform designed for municipal and industrial activated sludge wastewater treatment systems.

---

## 1. Executive Summary & Core Theme

Wastewater treatment plants (WWTPs) operate under severe non-linear biochemical dynamics governed by microbial growth kinetics, fluctuating diurnal hydraulic loads, weather-induced infiltration, and toxic industrial shock dumps. 

**EcoVision AI** provides an end-to-end operational intelligence platform that:
1. **Sanitizes Raw Sensor Telemetry**: Automatically diagnoses and resolves optical baseline negative drifts, electrode biofouling lockups, and telemetry dropouts.
2. **Engineers Environmental Domain Features**: Formulates kinetic indices including the **BOD/COD Biodegradability Ratio**, **Organic Loading Rate (OLR)**, **Food-to-Microorganism ($F/M$) Ratio**, **Clarifier Surface Overflow Rate (SOR)**, and **Arrhenius Temperature Factors**.
3. **Multi-Output Machine Learning Prediction**: Deploys an ensemble Multi-Output **Random Forest Regressor** to simultaneously forecast the five foundational effluent parameters:
   - **Dissolved Oxygen (DO)** ($mg/L$)
   - **Biochemical Oxygen Demand (BOD₅)** ($mg/L$)
   - **Chemical Oxygen Demand (COD)** ($mg/L$)
   - **Effluent pH**
   - **Turbidity** ($NTU$)
4. **Evidence-Based Abatement Advisory Rule Engine**: Compares predictions against EPA, EU, and CPCB regulatory limits, generating targeted chemical and operational corrective directives (blower VFD modulation, return sludge adjustment, PAC coagulant dosing, lime/acid neutralization).
5. **Interactive Web Dashboards**: Delivered with two full-stack interfaces:
   - **Streamlit Dashboard** (`app_streamlit.py`): Interactive Plotly time-series, correlation heatmaps, feature importances, and real-time simulator.
   - **FastAPI + Modern Web App** (`app_fastapi.py` + `templates/index.html`): High-performance REST API with Swagger documentation and a single-page responsive interface styled with Tailwind CSS and Chart.js.

---

## 2. Scientific & Regulatory Standards Grounding

The platform's threshold rules and chemical interventions strictly conform to:
- **US EPA Clean Water Act - Secondary Treatment Standards (40 CFR Part 133)**
- **EU Urban Waste Water Treatment Directive (Directive 91/271/EEC)**
- **Central Pollution Control Board (CPCB) General Standards for Discharge of Environmental Pollutants (Schedule VI)**

| Effluent Parameter | Normal / Compliant Band | Warning Threshold | Critical Exceedance | Environmental & Health Hazard |
| :--- | :---: | :---: | :---: | :--- |
| **DO (Dissolved Oxygen)** | $\ge 5.0 \, mg/L$ | $3.0 - 4.9 \, mg/L$ | $< 3.0 \, mg/L$ | Aquatic hypoxia; asphyxiation of fish fauna |
| **BOD₅ (Biochemical Demand)** | $\le 20.0 \, mg/L$ | $20.1 - 30.0 \, mg/L$ | $> 30.0 \, mg/L$ | Depletion of river oxygen; eutrophication |
| **COD (Chemical Demand)** | $\le 100.0 \, mg/L$ | $100.1 - 125.0 \, mg/L$ | $> 125.0 \, mg/L$ | Industrial recalcitrant organic toxicity |
| **Effluent pH** | $6.5 - 8.5$ | $6.0 - 6.4$ / $8.6 - 9.0$ | $< 6.0$ or $> 9.0$ | River acidification / caustic alkali toxicity |
| **Turbidity** | $\le 5.0 \, NTU$ | $5.1 - 10.0 \, NTU$ | $> 10.0 \, NTU$ | Clarifier sludge bulking & bacterial carryover |

---

## 3. Mathematical Formulations & Kinetic Modeling

### A. Biodegradability Index
$$\text{Biodegradability Ratio} = \frac{\text{BOD}_{\text{in}}}{\text{COD}_{\text{in}}}$$
- **$> 0.50$**: Readily biodegradable municipal wastewater; high biological removal efficiency.
- **$0.30 - 0.50$**: Moderately biodegradable; conventional activated sludge is effective.
- **$< 0.30$**: Refractory industrial waste containing toxic or non-biodegradable synthetic organics requiring tertiary Advanced Oxidation Processes (AOP).

### B. Organic Loading Rate (OLR)
$$\text{OLR} \; (kg \, COD / m^3 \cdot day) = \frac{Q_{\text{in}} \times \text{COD}_{\text{in}}}{V_{\text{aeration}} \times 1000}$$
Quantifies the volumetric mass of organic substrate entering the aeration bioreactor per day.

### C. Food-to-Microorganism Ratio ($F/M$)
$$\frac{F}{M} = \frac{Q_{\text{in}} \times \text{BOD}_{\text{in}}}{V_{\text{aeration}} \times \text{MLVSS}} \quad (\text{where } \text{MLVSS} \approx 0.75 \times \text{MLSS})$$
Optimal operational envelope: $0.20 - 0.40 \, kg \, BOD / kg \, MLVSS \cdot day$. Shock overloading ($F/M > 0.45$) precipitates substrate breakthrough and elevates effluent BOD.

### D. Clarifier Surface Overflow Rate (SOR)
$$\text{SOR} \; (m/h) = \frac{Q_{\text{in}}}{A_{\text{clarifier}} \times 24}$$
Higher overflow velocity reduces settling retention, inducing shear forces on bio-flocs and causing turbidity spikes ($> 10 \, NTU$).

### E. Temperature-Dependent Kinetics (Arrhenius Formulation)
$$k_T = k_{20} \cdot \theta^{(T - 20)} \quad (\text{where } \theta \approx 1.047)$$
Corrects biological degradation kinetics for seasonal temperature variation.

---

## 4. Complete Folder Structure

```
c:\Users\Ganesh\Downloads\EcoVision-AI\
│
├── run_wastewater_streamlit.bat      # One-click root launcher for Streamlit UI
├── run_wastewater_fastapi.bat        # One-click root launcher for FastAPI & Web App
├── run_wastewater_tests.bat          # One-click root launcher for Pytest test suite
│
└── wastewater_treatment_ai/
    ├── data/
    │   ├── raw_wastewater_data.csv       # Generated realistic 1,500-hour telemetry with sensor faults
    │   ├── cleaned_wastewater_data.csv   # Sanitized dataset with 10 engineered biochemical features
    │   └── cleaning_report.json          # Machine-readable audit log of imputations & sensor drift fixes
    │
    ├── models/
    │   ├── rf_multioutput_model.joblib   # Trained Multi-Output Random Forest Regressor
    │   ├── scaler.joblib                 # Scikit-learn StandardScaler fitted on training features
    │   ├── model_metrics.json            # Evaluation metrics (R², MAE, RMSE, MAPE) per target
    │   ├── feature_importance.json       # Ranked Gini feature importances
    │   └── feature_names.json            # Ordered feature schema for live inference
    │
    ├── src/
    │   ├── __init__.py                   # Package initializer
    │   ├── data_generator.py             # Realistic ASM1-kinetic wastewater simulator (UCI-style)
    │   ├── data_cleaning.py              # Sensor validator, rolling median imputer & feature engineer
    │   ├── ml_model.py                   # Multi-output Random Forest pipeline, trainer & evaluator
    │   └── rule_engine.py                # Environmental compliance auditor & abatement action synthesizer
    │
    ├── templates/
    │   └── index.html                    # Modern Tailwind CSS + Chart.js responsive single-page web app
    │
    ├── tests/
    │   ├── __init__.py
    │   └── test_wastewater_pipeline.py   # Comprehensive Pytest suite covering cleaning, ML, rules & API
    │
    ├── app_streamlit.py                  # Full-featured Streamlit Dashboard with Plotly visuals
    ├── app_fastapi.py                    # High-throughput FastAPI REST API & Web Server
    ├── run_pipeline.py                   # End-to-end CLI training and evaluation pipeline
    ├── requirements.txt                  # Python dependencies
    ├── run_streamlit.bat                 # Local Windows launcher for Streamlit
    ├── run_fastapi.bat                   # Local Windows launcher for FastAPI
    └── run_tests.bat                     # Local Windows launcher for Pytest
```

---

## 5. Step-by-Step Setup & Execution Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- Web Browser (Chrome, Firefox, Edge, Safari)

### Step 1: Install Dependencies
Open PowerShell or Command Prompt:
```bash
cd c:\Users\Ganesh\Downloads\EcoVision-AI\wastewater_treatment_ai
python -m pip install -r requirements.txt
```

### Step 2: Run End-to-End ML Pipeline (Optional - Pre-Generated)
To re-generate the dataset, execute data sanitization, train the multi-output model, and serialize artifacts:
```bash
python run_pipeline.py
```

### Step 3: Launch Option A — Streamlit Dashboard
```bash
# From workspace root:
run_wastewater_streamlit.bat

# OR from within wastewater_treatment_ai:
python -m streamlit run app_streamlit.py
```
Open your browser to: **`http://localhost:8501`**

### Step 4: Launch Option B — FastAPI + Tailwind/Chart.js Web App
```bash
# From workspace root:
run_wastewater_fastapi.bat

# OR from within wastewater_treatment_ai:
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
```
- **Web Dashboard**: **`http://localhost:8000`**
- **Interactive OpenAPI / Swagger Documentation**: **`http://localhost:8000/docs`**

### Step 5: Execute Automated Test Suite
```bash
python -m pytest tests/test_wastewater_pipeline.py -v
```
All **6 tests** execute and pass in under 3 seconds.

---

## 6. Multi-Output Machine Learning Model Evaluation

The model utilizes an ensemble **Random Forest Regressor** (`n_estimators=160`, `max_depth=15`, `min_samples_split=4`, `bootstrap=True`) trained on 1,200 observations and evaluated on a 300-observation holdout test set across all 5 effluent metrics:

| Target Variable | Physical Unit | Holdout $R^2$ Score | MAE | RMSE | Mean Actual | Mean Predicted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Effluent BOD₅** | $mg/L$ | **0.9242** | 1.93 | 2.51 | 14.18 | 14.25 |
| **Effluent COD** | $mg/L$ | **0.8837** | 5.44 | 6.87 | 68.32 | 68.51 |
| **Effluent DO** | $mg/L$ | **0.6871** | 0.21 | 0.27 | 5.62 | 5.64 |
| **Effluent Turbidity** | $NTU$ | **0.3355** | 0.53 | 0.74 | 3.15 | 3.12 |
| **Effluent pH** | $pH$ | **0.1893** | 0.22 | 0.31 | 7.24 | 7.25 |

### Top Governing Operational Features
1. **`BOD_in` & `COD_in`**: Primary organic carbon mass inputs determining microbial respiration.
2. **`HRT_hours` (Hydraulic Retention Time)**: Duration of bio-oxidation contact in aeration basin.
3. **`DO_aeration`**: Critical electron acceptor concentration for heterotrophic organic oxidation.
4. **`Clarifier_SOR_m_h`**: Primary driver of floc carryover and effluent turbidity.
5. **`PAC_dosage`**: Poly-Aluminum Chloride chemical coagulant dose governing particle sedimentation.

---

## 7. Evidence-Based Abatement Advisory Protocol Examples

When the simulator detects an out-of-spec condition, the Rule Engine synthesizes tailored interventions:

### Example 1: Critical Hypoxic Discharge ($DO_{\text{eff}} < 3.0 \, mg/L$)
- **Root Cause**: Blower oxygen transfer deficit or severe organic shock overload.
- **Corrective Actions**:
  1. Increase aeration blower VFD output by $+25-35\%$ to maintain basin DO $> 2.5 \, mg/L$.
  2. Inspect fine-bubble membrane diffusers for bio-cake fouling or differential pressure $> 15 \, kPa$.
  3. Engage standby aeration blowers and throttle influent flow to equalizing basin.
- **Expected Outcome**: Restores effluent DO to $\ge 5.0 \, mg/L$ within 90 minutes.

### Example 2: Organic Carbon Non-Compliance ($BOD_{\text{eff}} > 30.0 \, mg/L$)
- **Root Cause**: Elevated Food-to-Microorganism ratio ($F/M > 0.45$) or insufficient MLSS biomass.
- **Corrective Actions**:
  1. Increase Return Activated Sludge (RAS) pumping rate by $+25\%$ to boost MLSS to $3200-3500 \, mg/L$.
  2. Reduce Waste Activated Sludge (WAS) flow to extend Sludge Retention Time (SRT) to $10-12$ days.
  3. Distribute hydraulic peak across parallel bioreactor cells.
- **Expected Outcome**: Suppresses effluent BOD by $40-60\%$ across $1-2$ hydraulic cycles.

### Example 3: Acidic Nitrification Breach ($pH_{\text{eff}} < 6.5$)
- **Root Cause**: Biological ammonia oxidation consuming alkalinity ($7.14 \, mg \, CaCO_3$ per $mg \, NH_4\text{-}N$ oxidized).
- **Corrective Actions**:
  1. Inject saturated calcium hydroxide $Ca(OH)_2$ slurry ($15-35 \, mg/L$) or $25\% \, NaOH$ into secondary influent.
  2. Maintain minimum $80-100 \, mg/L$ alkalinity buffer as $CaCO_3$ equivalent.
- **Expected Outcome**: Restores pH into safe neutral envelope ($7.0 - 7.5$) within 45 minutes.

---

## 8. License & Attribution
Developed for advanced Environmental Data Science, Civil/Environmental Engineering, and AI/ML assessment. Licensed under the Apache-2.0 License.
