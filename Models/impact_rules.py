"""Dependency-weighted downstream impact rules (Section 3.6).
Shared by ImpactAssessment.py (batch results) and app.py (dashboard What-If tester),
so thresholds and advisory wording live in exactly one place."""

import pandas as pd

# Confirmed disruption = Isolation Forest flag that persisted >= 2 consecutive hours (raw flags: Predicted_Anomaly)
CONFIRMED_COL = "Predicted_Anomaly_2h"

BASELINE_FLOW = 200  # normal pipeline baseline MMSCFD

dependency_weights = {
    "LNG": 0.50,
    "COGEN": 0.30,
    "PCG": 0.20
}

BASELINE_EFFICIENCY = {"LNG": 90, "COGEN": 85, "PCG": 82}  # normal plant efficiency means from the generator

# A localized fault leaves gas supply intact and is recoverable on-site, so its efficiency drop is scaled down:
# ~20% efficiency (≈76% drop) x 0.20 ≈ 15 -> Medium for that single plant
LOCAL_FAULT_FACTOR = 0.20

# Per-plant impact level cut-offs (dependency-weighted scores)
HIGH_THRESHOLD = 18
MEDIUM_THRESHOLD = 8

# System severity for supply disruptions uses the unweighted pipeline supply drop (%):
# Low_Pressure (~50% drop) -> Medium, Compressor_Trip (~100% drop) -> High
SYSTEM_HIGH_DROP = 75
SYSTEM_MEDIUM_DROP = 25

PLANTS = ["LNG", "COGEN", "PCG"]
SUPPLY_DISRUPTIONS = {"Low_Pressure", "Compressor_Trip"}
LOCAL_FAULTS = {f"{p}_Underperformance": p for p in PLANTS}
LEVEL_ORDER = ["No Impact", "Low", "Medium", "High"]
LEVEL_COLS = [f"{p}_Impact_Level" for p in PLANTS]


def classify_impact(score, high=HIGH_THRESHOLD, medium=MEDIUM_THRESHOLD):
    if score >= high:
        return "High"
    elif score >= medium:
        return "Medium"
    elif score > 0:
        return "Low"
    else:
        return "No Impact"


def calculate_impact_scores(row):
    """Per-plant impact scores driven by the XGBoost disruption type."""
    disruption = row["Predicted_Anomaly_Type"]
    supply_drop = max(((BASELINE_FLOW - row["Midstream_Pipeline_Flow"]) / BASELINE_FLOW) * 100, 0)
    scores = {p: 0.0 for p in PLANTS}

    if disruption in SUPPLY_DISRUPTIONS:
        scores = {p: supply_drop * dependency_weights[p] for p in PLANTS}
    elif disruption in LOCAL_FAULTS:
        plant = LOCAL_FAULTS[disruption]
        eff_drop = max((BASELINE_EFFICIENCY[plant] - row[f"{plant}_Plant_Efficiency"])
                       / BASELINE_EFFICIENCY[plant] * 100, 0)
        scores[plant] = eff_drop * LOCAL_FAULT_FACTOR

    return supply_drop, scores


def system_severity(data, plant_levels, factor=1.0):
    """Supply disruptions: tier of the total supply drop. Localized faults: the affected plant's level.
    IF-only confirmed anomalies (XGBoost says Normal) have no known disruption type: lowest actionable tier."""
    severity = plant_levels.apply(lambda levels: max(levels, key=LEVEL_ORDER.index), axis=1)

    supply = data["Predicted_Anomaly_Type"].isin(SUPPLY_DISRUPTIONS)
    severity[supply] = data.loc[supply, "Supply_Drop_Percent"].apply(
        classify_impact, high=SYSTEM_HIGH_DROP * factor, medium=SYSTEM_MEDIUM_DROP * factor)

    severity[(data["Predicted_Anomaly_Type"] == "Normal") & (data[CONFIRMED_COL] == 1)] = "Low"
    return severity


RECOMMENDATIONS = {
    "High": "Escalate to operations planner; review gas allocation and production plans of affected facilities.",
    "Medium": "Notify affected facility operations; review production plan and continue close monitoring.",
    "Low": "Continue monitoring and verify sensor readings.",
    "No Impact": "No action required.",
}

CAUSES = {
    "Low_Pressure": "Upstream low pressure reducing pipeline supply",
    "Compressor_Trip": "Midstream compressor trip blocking pipeline supply",
    "LNG_Underperformance": "LNG plant underperformance (localized)",
    "COGEN_Underperformance": "COGEN plant underperformance (localized)",
    "PCG_Underperformance": "PCG plant underperformance (localized)",
}


def get_recommendation(row):
    """Advisory text only: monitoring / escalation, never autonomous control actions."""
    disruption = row["Predicted_Anomaly_Type"]
    if disruption == "Normal":
        if row[CONFIRMED_COL] == 1:
            return ("Unclassified anomaly: Isolation Forest anomaly persisted 2+ hours; verify sensor readings and "
                    "continue close monitoring.")
        return RECOMMENDATIONS["No Impact"]
    return f"{CAUSES[disruption]}. {RECOMMENDATIONS[row['System_Severity']]}"


def assess_impact(df):
    """Add impact scores/levels, affected facilities, system severity and advisory to a copy of df.
    df needs: Predicted_Anomaly_Type, CONFIRMED_COL, Midstream_Pipeline_Flow and the three *_Plant_Efficiency columns."""
    df = df.copy()
    impact = df.apply(calculate_impact_scores, axis=1)
    df["Supply_Drop_Percent"] = [drop for drop, _ in impact]

    for p in PLANTS:
        df[f"{p}_Impact_Score"] = [scores[p] for _, scores in impact]
        df[f"{p}_Impact_Level"] = df[f"{p}_Impact_Score"].apply(classify_impact)

    df["Affected_Facilities"] = df[LEVEL_COLS].apply(
        lambda levels: ", ".join(f"{p}_Plant" for p, lvl in zip(PLANTS, levels) if lvl != "No Impact") or "None",
        axis=1
    )
    df["System_Severity"] = system_severity(df, df[LEVEL_COLS])
    df["Recommendation"] = df.apply(get_recommendation, axis=1)
    return df
