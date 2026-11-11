import pandas as pd


# =========================
# 1. Load Data
# =========================

df = pd.read_csv("test_dataset_with_anomaly_predictions.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])


# =========================
# 2. Define Impact Logic
# =========================

BASELINE_FLOW = 200  # normal upstream baseline MMSCFD

dependency_weights = {
    "LNG": 0.50,
    "Power": 0.30,
    "PCG": 0.20
}


def calculate_supply_drop(actual_flow):
    drop = ((BASELINE_FLOW - actual_flow) / BASELINE_FLOW) * 100
    return max(drop, 0)


def classify_impact(score):
    if score >= 10:
        return "High"
    elif score >= 5:
        return "Medium"
    elif score > 0:
        return "Low"
    else:
        return "No Impact"


def get_recommendation(lng_level, power_level, pcg_level):
    levels = [lng_level, power_level, pcg_level]

    if "High" in levels:
        return "Escalate to operations planner and review downstream allocation."

    elif "Medium" in levels:
        return "Review downstream production plan and continue close monitoring."

    elif "Low" in levels:
        return "Continue monitoring."

    else:
        return "No action required."


# =========================
# 3. Calculate Impact
# =========================

df["Supply_Drop_Percent"] = df["Upstream_Flow"].apply(calculate_supply_drop)

df["LNG_Impact_Score"] = df["Supply_Drop_Percent"] * dependency_weights["LNG"]
df["Power_Impact_Score"] = df["Supply_Drop_Percent"] * dependency_weights["Power"]
df["PCG_Impact_Score"] = df["Supply_Drop_Percent"] * dependency_weights["PCG"]

df["LNG_Impact_Level"] = df["LNG_Impact_Score"].apply(classify_impact)
df["Power_Impact_Level"] = df["Power_Impact_Score"].apply(classify_impact)
df["PCG_Impact_Level"] = df["PCG_Impact_Score"].apply(classify_impact)

df["Recommendation"] = df.apply(
    lambda row: get_recommendation(
        row["LNG_Impact_Level"],
        row["Power_Impact_Level"],
        row["PCG_Impact_Level"]
    ),
    axis=1
)


# =========================
# 4. Keep Only Detected Anomalies
# =========================

impact_df = df[df["Predicted_Anomaly"] == 1].copy()


# =========================
# 5. Save Impact Results
# =========================

impact_df.to_csv("impact_assessment_results.csv", index=False)


# =========================
# 6. Display Summary
# =========================

print("\n===== Impact Assessment Completed =====")
print(f"Total detected anomalies: {len(impact_df)}")

print("\n===== Impact Level Summary =====")
print("\nLNG Impact:")
print(impact_df["LNG_Impact_Level"].value_counts())

print("\nPower Impact:")
print(impact_df["Power_Impact_Level"].value_counts())

print("\nPCG Impact:")
print(impact_df["PCG_Impact_Level"].value_counts())

print("\n===== Sample Impact Assessment Results =====")
print(
    impact_df[
        [
            "Timestamp",
            "Upstream_Flow",
            "Supply_Drop_Percent",
            "LNG_Impact_Level",
            "Power_Impact_Level",
            "PCG_Impact_Level",
            "Recommendation"
        ]
    ].head(20)
)

print("\nFile saved: impact_assessment_results.csv")