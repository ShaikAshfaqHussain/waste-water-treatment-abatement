"""Environmental Anomaly Detection & Pollution Abatement Rule Engine.

Grounded in:
- US EPA Clean Water Act Secondary Treatment Regulations (40 CFR Part 133)
- EU Urban Waste Water Treatment Directive (91/271/EEC)
- Central Pollution Control Board (CPCB) General Effluent Discharge Standards (Schedule VI)
"""

from typing import Dict, Any, List


REGULATORY_THRESHOLDS = {
    "DO_eff": {
        "unit": "mg/L",
        "description": "Effluent Dissolved Oxygen",
        "normal_min": 5.0,
        "warning_min": 4.0,
        "critical_min": 3.0,
        "direction": "higher_is_better",
        "standard_ref": "EPA 40 CFR 133 / CPCB Schedule VI (Minimum 5.0 mg/L to prevent aquatic hypoxia; < 3.0 mg/L critical)",
    },
    "BOD_eff": {
        "unit": "mg/L",
        "description": "5-Day Biochemical Oxygen Demand",
        "normal_max": 20.0,
        "warning_max": 30.0,
        "critical_max": 30.0,
        "direction": "lower_is_better",
        "standard_ref": "CPCB / EPA 30-Day Discharge Cap (Max 30 mg/L; optimal < 20 mg/L)",
    },
    "COD_eff": {
        "unit": "mg/L",
        "description": "Chemical Oxygen Demand",
        "normal_max": 100.0,
        "warning_max": 125.0,
        "critical_max": 125.0,
        "direction": "lower_is_better",
        "standard_ref": "EU Directive 91/271/EEC / CPCB (Discharge Cap 125 mg/L; warning 100 mg/L)",
    },
    "pH_eff": {
        "unit": "pH units",
        "description": "Effluent pH",
        "normal_min": 6.5,
        "normal_max": 8.5,
        "warning_min": 6.0,
        "warning_max": 9.0,
        "critical_min": 6.0,
        "critical_max": 9.0,
        "direction": "in_range",
        "standard_ref": "EPA / CPCB Effluent Range (Permitted 6.5 - 8.5; Excursions < 6.0 or > 9.0 violate permit)",
    },
    "Turbidity_eff": {
        "unit": "NTU",
        "description": "Effluent Turbidity",
        "normal_max": 5.0,
        "warning_max": 10.0,
        "critical_max": 10.0,
        "direction": "lower_is_better",
        "standard_ref": "WHO / EPA Clarification Benchmark (Max 10 NTU; pristine secondary effluent < 5 NTU)",
    },
}


class WastewaterAbatementEngine:
    """Evaluates multi-metric effluent predictions against environmental standards

    and synthesizes actionable engineering abatement protocols.
    """

    def evaluate_compliance(
        self,
        predictions: Dict[str, float],
        plant_inputs: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Runs rule engine against effluent metrics and operational plant inputs.

        Parameters
        ----------
        predictions : Dict[str, float]
            Predicted effluent values: DO_eff, BOD_eff, COD_eff, pH_eff, Turbidity_eff.
        plant_inputs : Dict[str, float], optional
            Plant operational settings (e.g. Q_in, DO_aeration, MLSS, PAC_dosage).

        Returns
        -------
        Dict[str, Any]
            Compliance evaluation summary, per-parameter audit, overall risk status,
            and prioritized abatement recommendations.
        """
        plant_inputs = plant_inputs or {}
        audit_results: Dict[str, Dict[str, Any]] = {}
        abatement_actions: List[Dict[str, Any]] = []

        critical_count = 0
        warning_count = 0
        health_penalties = 0

        # 1. Evaluate DO_eff (Dissolved Oxygen)
        do_val = float(predictions.get("DO_eff", 5.5))
        do_rules = REGULATORY_THRESHOLDS["DO_eff"]
        if do_val < do_rules["critical_min"]:
            status = "CRITICAL"
            critical_count += 1
            health_penalties += 35
            severity = "Severe Hypoxia Hazard"
            message = f"Effluent DO ({do_val:.2f} mg/L) is below emergency limit (3.0 mg/L), threatening immediate aquatic asphyxiation."
            action = {
                "parameter": "Dissolved Oxygen (DO)",
                "level": "CRITICAL",
                "predicted_value": f"{do_val:.2f} mg/L",
                "target_threshold": ">= 5.0 mg/L",
                "root_cause": "Aeration basin oxygen transfer deficit or excessive microbial respiration rate under high organic load.",
                "action_steps": [
                    "Immediately increase aeration blower variable frequency drive (VFD) output by +25-35%.",
                    "Engage standby auxiliary centrifugal blowers to bring aeration basin DO above 2.5 mg/L.",
                    "Inspect aeration diffuser grids for biofilm fouling or header pressure drops exceeding 15 kPa.",
                    "Temporarily throttle influent flow into equalization basin to alleviate microbial oxygen demand.",
                ],
                "expected_impact": "Raises effluent DO to >= 5.0 mg/L within 90 minutes, restoring aerobic discharge compliance.",
            }
            abatement_actions.append(action)
        elif do_val < do_rules["normal_min"]:
            status = "WARNING"
            warning_count += 1
            health_penalties += 15
            severity = "Marginal Oxygenation"
            message = f"Effluent DO ({do_val:.2f} mg/L) is below recommended standard (5.0 mg/L)."
            action = {
                "parameter": "Dissolved Oxygen (DO)",
                "level": "WARNING",
                "predicted_value": f"{do_val:.2f} mg/L",
                "target_threshold": ">= 5.0 mg/L",
                "root_cause": "Sub-optimal aeration control loop response or elevated water temperature reducing oxygen saturation.",
                "action_steps": [
                    "Increment aeration blower setpoint by +10-15%.",
                    "Verify dissolved oxygen optical probe calibration in the final discharge weir.",
                    "Optimize post-aeration cascade weir drop height to maximize mechanical surface re-aeration.",
                ],
                "expected_impact": "Brings effluent DO into compliant band (> 5.0 mg/L).",
            }
            abatement_actions.append(action)
        else:
            status = "NORMAL"
            severity = "Fully Compliant"
            message = f"Effluent DO ({do_val:.2f} mg/L) satisfies environmental criteria (>= 5.0 mg/L)."

        audit_results["DO_eff"] = {
            "value": do_val,
            "unit": "mg/L",
            "status": status,
            "severity": severity,
            "message": message,
            "standard": do_rules["standard_ref"],
        }

        # 2. Evaluate BOD_eff (Biochemical Oxygen Demand)
        bod_val = float(predictions.get("BOD_eff", 14.0))
        bod_rules = REGULATORY_THRESHOLDS["BOD_eff"]
        if bod_val > bod_rules["critical_max"]:
            status = "CRITICAL"
            critical_count += 1
            health_penalties += 40
            severity = "Severe Organic Non-Compliance"
            message = f"Effluent BOD ({bod_val:.1f} mg/L) exceeds the regulatory ceiling (30.0 mg/L)."
            action = {
                "parameter": "Biochemical Oxygen Demand (BOD5)",
                "level": "CRITICAL",
                "predicted_value": f"{bod_val:.1f} mg/L",
                "target_threshold": "<= 20.0 mg/L (Cap: 30.0 mg/L)",
                "root_cause": "Biological kinetic rate failure caused by insufficient HRT, organic shock overload (F/M > 0.45), or low MLSS.",
                "action_steps": [
                    "Increase Return Activated Sludge (RAS) pumping rate by +25% to boost aeration MLSS toward 3500 mg/L.",
                    "Reduce Waste Activated Sludge (WAS) flow to extend Mean Cell Residence Time (Sludge Age / SRT) to 10-12 days.",
                    "If plant receives high-strength industrial influent, divert excess flow to retention equalization tanks.",
                    "Dose supplemental readily biodegradable carbon (e.g. sodium acetate) only if carbon-deficiency is arresting biomass growth.",
                ],
                "expected_impact": "Reduces effluent BOD5 by 40-60% within 1-2 hydraulic cycles.",
            }
            abatement_actions.append(action)
        elif bod_val > bod_rules["normal_max"]:
            status = "WARNING"
            warning_count += 1
            health_penalties += 15
            severity = "Elevated Organic Load"
            message = f"Effluent BOD ({bod_val:.1f} mg/L) approaches the 30 mg/L regulatory cap."
            action = {
                "parameter": "Biochemical Oxygen Demand (BOD5)",
                "level": "WARNING",
                "predicted_value": f"{bod_val:.1f} mg/L",
                "target_threshold": "<= 20.0 mg/L",
                "root_cause": "Hydraulic peak load reducing effective bioreactor retention time.",
                "action_steps": [
                    "Optimize hydraulic loading by distributing flow evenly across parallel aeration basins.",
                    "Trim RAS recycle ratio to match influent diurnal flux.",
                ],
                "expected_impact": "Stabilizes effluent BOD comfortably below 20 mg/L.",
            }
            abatement_actions.append(action)
        else:
            status = "NORMAL"
            severity = "Fully Compliant"
            message = f"Effluent BOD ({bod_val:.1f} mg/L) is within safe ecological parameters (<= 20 mg/L)."

        audit_results["BOD_eff"] = {
            "value": bod_val,
            "unit": "mg/L",
            "status": status,
            "severity": severity,
            "message": message,
            "standard": bod_rules["standard_ref"],
        }

        # 3. Evaluate COD_eff (Chemical Oxygen Demand)
        cod_val = float(predictions.get("COD_eff", 65.0))
        cod_rules = REGULATORY_THRESHOLDS["COD_eff"]
        if cod_val > cod_rules["critical_max"]:
            status = "CRITICAL"
            critical_count += 1
            health_penalties += 30
            severity = "Industrial Refractory Exceedance"
            message = f"Effluent COD ({cod_val:.1f} mg/L) breaches the 125.0 mg/L environmental limit."
            action = {
                "parameter": "Chemical Oxygen Demand (COD)",
                "level": "CRITICAL",
                "predicted_value": f"{cod_val:.1f} mg/L",
                "target_threshold": "<= 100.0 mg/L (Cap: 125.0 mg/L)",
                "root_cause": "High proportion of non-biodegradable soluble inerts (S_I) or industrial chemical dumping.",
                "action_steps": [
                    "Engage tertiary Advanced Oxidation Process (AOP: UV / Fenton / Ozone) to break refractory organics.",
                    "Inject powdered activated carbon (PAC, 20-30 mg/L) directly into aeration contact zone for physical adsorption.",
                    "Notify upstream industrial dischargers to enforce pretreatment compliance.",
                ],
                "expected_impact": "Destroys non-biodegradable complex aromatics, reducing effluent COD by 35-50 mg/L.",
            }
            abatement_actions.append(action)
        elif cod_val > cod_rules["normal_max"]:
            status = "WARNING"
            warning_count += 1
            health_penalties += 12
            severity = "Marginal Chemical Load"
            message = f"Effluent COD ({cod_val:.1f} mg/L) exceeds warning threshold (100.0 mg/L)."
            action = {
                "parameter": "Chemical Oxygen Demand (COD)",
                "level": "WARNING",
                "predicted_value": f"{cod_val:.1f} mg/L",
                "target_threshold": "<= 100.0 mg/L",
                "root_cause": "Elevated particulate COD carryover or partial bio-oxidation kinetics.",
                "action_steps": [
                    "Improve primary and secondary sedimentation settling times.",
                    "Verify coagulation dosing upstream of tertiary sand filters.",
                ],
                "expected_impact": "Maintains effluent COD safely below 100 mg/L.",
            }
            abatement_actions.append(action)
        else:
            status = "NORMAL"
            severity = "Fully Compliant"
            message = f"Effluent COD ({cod_val:.1f} mg/L) complies with discharge norms (<= 100 mg/L)."

        audit_results["COD_eff"] = {
            "value": cod_val,
            "unit": "mg/L",
            "status": status,
            "severity": severity,
            "message": message,
            "standard": cod_rules["standard_ref"],
        }

        # 4. Evaluate pH_eff
        ph_val = float(predictions.get("pH_eff", 7.2))
        ph_rules = REGULATORY_THRESHOLDS["pH_eff"]
        if ph_val < ph_rules["critical_min"]:
            status = "CRITICAL"
            critical_count += 1
            health_penalties += 35
            severity = "Severe Acidic Effluent"
            message = f"Effluent pH ({ph_val:.2f}) violates permit minimum (6.0), risking ecological toxicity and pipe corrosion."
            action = {
                "parameter": "Effluent pH (Acidic)",
                "level": "CRITICAL",
                "predicted_value": f"{ph_val:.2f}",
                "target_threshold": "6.5 - 8.5",
                "root_cause": "Nitrification alkalinity depletion (7.14 mg CaCO3 consumed per mg NH4+ oxidized) or acidic industrial influent shock.",
                "action_steps": [
                    "Immediately initiate chemical neutralization: dose hydrated lime Ca(OH)2 slurry (20-35 mg/L) or 25% NaOH into secondary influent channel.",
                    "Monitor aeration alkalinity to ensure minimum 80-100 mg/L as CaCO3 residual.",
                    "Temporarily trim excessive aeration if over-nitrification is stripping buffering capacity.",
                ],
                "expected_impact": "Restores pH into neutral buffer zone (7.0 - 7.5) within 45 minutes.",
            }
            abatement_actions.append(action)
        elif ph_val > ph_rules["critical_max"]:
            status = "CRITICAL"
            critical_count += 1
            health_penalties += 35
            severity = "Severe Caustic Effluent"
            message = f"Effluent pH ({ph_val:.2f}) exceeds permit maximum (9.0), causing alkaline shock to receiving water."
            action = {
                "parameter": "Effluent pH (Alkaline)",
                "level": "CRITICAL",
                "predicted_value": f"{ph_val:.2f}",
                "target_threshold": "6.5 - 8.5",
                "root_cause": "Industrial caustic CIP (clean-in-place) washdown breach or chemical overdose.",
                "action_steps": [
                    "Initiate acid neutralization: inject dilute sulfuric acid (0.5N H2SO4) or submerged carbon dioxide (CO2) sparging.",
                    "Verify influent pH sensor and trace caustic industrial discharge source.",
                ],
                "expected_impact": "Neutralizes alkaline stream to pH 7.4 - 7.8 safely.",
            }
            abatement_actions.append(action)
        elif ph_val < ph_rules["normal_min"] or ph_val > ph_rules["normal_max"]:
            status = "WARNING"
            warning_count += 1
            health_penalties += 12
            severity = "Sub-Optimal pH Band"
            message = f"Effluent pH ({ph_val:.2f}) is outside optimal 6.5 - 8.5 envelope."
            action = {
                "parameter": "Effluent pH",
                "level": "WARNING",
                "predicted_value": f"{ph_val:.2f}",
                "target_threshold": "6.5 - 8.5",
                "root_cause": "Alkalinity drift or coagulant-induced acidic suppression.",
                "action_steps": [
                    "Adjust coagulant (PAC) dosing rate, which slightly acidifies water.",
                    "Check lime feeder readiness.",
                ],
                "expected_impact": "Maintains effluent pH within standard 6.5 - 8.5 span.",
            }
            abatement_actions.append(action)
        else:
            status = "NORMAL"
            severity = "Fully Compliant"
            message = f"Effluent pH ({ph_val:.2f}) is within standard regulatory band (6.5 - 8.5)."

        audit_results["pH_eff"] = {
            "value": ph_val,
            "unit": "pH",
            "status": status,
            "severity": severity,
            "message": message,
            "standard": ph_rules["standard_ref"],
        }

        # 5. Evaluate Turbidity_eff
        turb_val = float(predictions.get("Turbidity_eff", 3.2))
        turb_rules = REGULATORY_THRESHOLDS["Turbidity_eff"]
        if turb_val > turb_rules["critical_max"]:
            status = "CRITICAL"
            critical_count += 1
            health_penalties += 30
            severity = "Biomass Clarifier Carryover"
            message = f"Effluent Turbidity ({turb_val:.2f} NTU) exceeds critical discharge cap (10.0 NTU)."
            action = {
                "parameter": "Turbidity & Suspended Solids",
                "level": "CRITICAL",
                "predicted_value": f"{turb_val:.2f} NTU",
                "target_threshold": "<= 5.0 NTU (Cap: 10.0 NTU)",
                "root_cause": "Secondary clarifier hydraulic overload (SOR > 1.0 m/h), sludge bulking (filamentous bacteria), or coagulant underdosing.",
                "action_steps": [
                    "Increase Poly-Aluminum Chloride (PAC) coagulant dosing by +5 to 10 mg/L at clarifier inlet well.",
                    "Inject anionic polymer flocculant (0.5 - 1.0 mg/L) to strengthen pin-point floc settleability.",
                    "Increase Waste Activated Sludge (WAS) rate to lower sludge blanket depth in secondary clarifier.",
                    "Initiate backwash cycle for tertiary dual-media pressure filters.",
                ],
                "expected_impact": "Rapidly suppresses floc carryover, restoring effluent turbidity to < 4.0 NTU.",
            }
            abatement_actions.append(action)
        elif turb_val > turb_rules["normal_max"]:
            status = "WARNING"
            warning_count += 1
            health_penalties += 12
            severity = "Moderate Turbidity Fluctuation"
            message = f"Effluent Turbidity ({turb_val:.2f} NTU) is above optimal clarity target (5.0 NTU)."
            action = {
                "parameter": "Turbidity",
                "level": "WARNING",
                "predicted_value": f"{turb_val:.2f} NTU",
                "target_threshold": "<= 5.0 NTU",
                "root_cause": "Minor clarifier surface turbulence or mild floc shearing.",
                "action_steps": [
                    "Fine-tune coagulant dosing (+2-3 mg/L).",
                    "Check clarifier scum baffle and effluent v-notch weir cleanliness.",
                ],
                "expected_impact": "Reduces turbidity to pristine levels (< 5 NTU).",
            }
            abatement_actions.append(action)
        else:
            status = "NORMAL"
            severity = "Fully Compliant"
            message = f"Effluent Turbidity ({turb_val:.2f} NTU) is crystal clear (<= 5.0 NTU)."

        audit_results["Turbidity_eff"] = {
            "value": turb_val,
            "unit": "NTU",
            "status": status,
            "severity": severity,
            "message": message,
            "standard": turb_rules["standard_ref"],
        }

        # Calculate Overall Plant Health & Compliance Status
        plant_health_score = max(0, min(100, 100 - health_penalties))
        if critical_count > 0:
            overall_status = "CRITICAL_NON_COMPLIANT"
            overall_color = "red"
            summary_statement = f"EMERGENCY: {critical_count} regulatory parameter(s) breach critical environmental discharge thresholds. Immediate abatement intervention required."
        elif warning_count > 0:
            overall_status = "WARNING_AT_RISK"
            overall_color = "amber"
            summary_statement = f"CAUTION: {warning_count} operational metric(s) in warning zone approaching regulatory caps. Preventative action advised."
        else:
            overall_status = "OPTIMAL_COMPLIANT"
            overall_color = "emerald"
            summary_statement = "OPTIMAL: All effluent parameters fully comply with EPA / EU / CPCB environmental discharge standards."

        return {
            "overall_status": overall_status,
            "overall_color": overall_color,
            "plant_health_score": plant_health_score,
            "critical_violations": critical_count,
            "warning_conditions": warning_count,
            "summary_statement": summary_statement,
            "parameter_audits": audit_results,
            "abatement_actions": abatement_actions,
        }


if __name__ == "__main__":
    engine = WastewaterAbatementEngine()
    test_preds = {
        "DO_eff": 2.4,  # Critical low
        "BOD_eff": 34.2,  # Critical high
        "COD_eff": 115.0,  # Warning
        "pH_eff": 6.1,  # Warning low
        "Turbidity_eff": 12.8,  # Critical high
    }
    result = engine.evaluate_compliance(test_preds)
    print("Overall Status:", result["overall_status"])
    print("Health Score:", result["plant_health_score"])
    print("Abatement Actions Count:", len(result["abatement_actions"]))
    for act in result["abatement_actions"]:
        print(f"[{act['level']}] {act['parameter']}: {act['action_steps'][0]}")
