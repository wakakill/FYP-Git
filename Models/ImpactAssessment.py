"""Objective 5: dependency-weighted downstream impact assessment + SHAP explainability.
Inputs (from Train_Anomaly.py and Train_Classifier.py, in result/):
    test_dataset_with_anomaly_predictions.csv, xgboost_classification_results.csv,
    xgboost_anomaly_classifier.pkl, xgboost_label_encoder.pkl
Outputs (result/): impact_assessment_results.csv, dashboard_master_test.csv, shap_global_importance.csv,
    shap_global_summary.png, shap_artifacts.pkl, impact_threshold_sensitivity.csv"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from impact_rules import (CONFIRMED_COL, HIGH_THRESHOLD, MEDIUM_THRESHOLD, SYSTEM_HIGH_DROP, SYSTEM_MEDIUM_DROP,
                          PLANTS, LEVEL_ORDER, LEVEL_COLS, classify_impact, system_severity, assess_impact)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"


# =========================
# 1. Load & Align IF + XGBoost Test Predictions
# =========================

anomaly_df = pd.read_csv(RESULT_DIR / "test_dataset_with_anomaly_predictions.csv", parse_dates=["Timestamp"])

# Confirmed disruption = Isolation Forest after the 2-hour persistence rule (raw flags stay in Predicted_Anomaly)
if CONFIRMED_COL not in anomaly_df.columns:
    raise ValueError(f"{CONFIRMED_COL} missing from test_dataset_with_anomaly_predictions.csv. "
                     "Re-run Train_Anomaly.py.")

xgb_df = pd.read_csv(RESULT_DIR / "xgboost_classification_results.csv", parse_dates=["Timestamp"])

xgb_cols = ["Timestamp", "Prediction_Confidence", "Predicted_Label_Encoded", "Predicted_Anomaly_Type"]
df = anomaly_df.merge(xgb_df[xgb_cols], on="Timestamp", how="inner", validate="one_to_one")
df = df.sort_values("Timestamp").reset_index(drop=True)

if not (len(df) == len(anomaly_df) == len(xgb_df)):
    raise ValueError(f"Test rows do not align 1-to-1: IF={len(anomaly_df)}, XGB={len(xgb_df)}, merged={len(df)}. "
                     "Re-run Train_Anomaly.py and Train_Classifier.py.")

print(f"===== Aligned Test Predictions: {len(df)} rows "
      f"({df['Timestamp'].min()} to {df['Timestamp'].max()}) =====")


# =========================
# 2-3. Impact Assessment for Every Test Row (rules in impact_rules.py, Section 3.6)
# =========================

df = assess_impact(df)

is_anomalous = (df[CONFIRMED_COL] == 1) | (df["Predicted_Anomaly_Type"] != "Normal")

print("\n===== Impact Assessment =====")
print(f"Anomalous records (IF confirmed >=2h or XGBoost non-Normal): {is_anomalous.sum()}")
print("\nSystem severity by predicted disruption type:")
print(pd.crosstab(df.loc[is_anomalous, "Predicted_Anomaly_Type"], df.loc[is_anomalous, "System_Severity"]))
print("\nPer-plant impact level by predicted disruption type:")
for p in PLANTS:
    print(f"\n{p}:")
    print(pd.crosstab(df.loc[is_anomalous, "Predicted_Anomaly_Type"], df.loc[is_anomalous, f"{p}_Impact_Level"]))


# =========================
# 4. Threshold Sensitivity Analysis (±15%)
# =========================

anom = df[is_anomalous]
baseline_severity = anom["System_Severity"]
sensitivity_rows = []

for label, factor in [("-15%", 0.85), ("Baseline", 1.00), ("+15%", 1.15)]:
    high, medium = HIGH_THRESHOLD * factor, MEDIUM_THRESHOLD * factor
    plant_levels = pd.DataFrame({
        p: anom[f"{p}_Impact_Score"].apply(classify_impact, high=high, medium=medium) for p in PLANTS
    })
    severity = system_severity(anom, plant_levels, factor)

    counts = severity.value_counts()
    sensitivity_rows.append({
        "Scenario": label,
        "Plant_High_Cutoff": round(high, 2),
        "Plant_Medium_Cutoff": round(medium, 2),
        "System_High_Drop_%": round(SYSTEM_HIGH_DROP * factor, 2),
        "System_Medium_Drop_%": round(SYSTEM_MEDIUM_DROP * factor, 2),
        **{f"{lvl}_Records": int(counts.get(lvl, 0)) for lvl in reversed(LEVEL_ORDER)},
        "System_Severity_Changed": int((severity != baseline_severity).sum()),
        "Plant_Levels_Changed": int((plant_levels.to_numpy() != anom[LEVEL_COLS].to_numpy()).sum()),
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
print("\n===== Threshold Sensitivity (+/-15%) =====")
print(sensitivity_df.to_string(index=False))


# =========================
# 5. SHAP Explainability (Section 3.7)
# =========================

xgb_model = joblib.load(RESULT_DIR / "xgboost_anomaly_classifier.pkl")
label_encoder = joblib.load(RESULT_DIR / "xgboost_label_encoder.pkl")
class_names = list(label_encoder.classes_)
features = list(xgb_model.get_booster().feature_names)  # exact training feature order

X_test = df[features]
explainer = shap.TreeExplainer(xgb_model)
shap_values = np.asarray(explainer.shap_values(X_test))
if shap_values.shape != (len(X_test), len(features), len(class_names)):  # older SHAP: (classes, rows, features)
    shap_values = np.transpose(shap_values, (1, 2, 0))

# Sanity check: the model's own predictions match the stored XGBoost predictions
if not (xgb_model.predict(X_test) == df["Predicted_Label_Encoded"]).all():
    raise ValueError("Loaded XGBoost model does not reproduce stored predictions. Re-run Train_Classifier.py.")

# Global importance: mean |SHAP| per feature, per class and overall
mean_abs = np.abs(shap_values).mean(axis=0)  # (features, classes)
importance_df = pd.DataFrame(mean_abs, index=features, columns=class_names)
importance_df.insert(0, "Mean_Abs_SHAP_All_Classes", mean_abs.mean(axis=1))
importance_df = importance_df.sort_values("Mean_Abs_SHAP_All_Classes", ascending=False)
importance_df.index.name = "Feature"

plt.figure(figsize=(10, 6))
plt.barh(importance_df.index[::-1], importance_df["Mean_Abs_SHAP_All_Classes"][::-1])
plt.xlabel("Mean |SHAP value| (averaged over all classes)")
plt.title("XGBoost Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(RESULT_DIR / "shap_global_summary.png", dpi=150)
plt.close()

print("\n===== SHAP Global Feature Importance =====")
print(importance_df["Mean_Abs_SHAP_All_Classes"].round(4).to_string())

# Top 3 sensors pushing each row toward its predicted class
pred_idx = df["Predicted_Label_Encoded"].to_numpy()
row_shap = shap_values[np.arange(len(df)), :, pred_idx]  # (rows, features)
top3 = np.argsort(-row_shap, axis=1)[:, :3]
X_values = X_test.to_numpy()

df["SHAP_Top_Contributors"] = [
    ", ".join(f"{features[j]}: {X_values[i, j]:.1f} ({row_shap[i, j]:+.2f})" for j in top3[i]
              if row_shap[i, j] >= 0.005) or "None"  # only sensors visibly pushing toward the predicted class
    for i in range(len(df))
]


# =========================
# 6. Sample Anomaly Events
# =========================

print("\n===== Sample Anomaly Events =====")
for disruption in ["Low_Pressure", "Compressor_Trip", "LNG_Underperformance"]:
    matches = df[(df["Predicted_Anomaly_Type"] == disruption) & (df["System_Status"] == disruption)]
    if matches.empty:
        print(f"\n{disruption}: no correctly predicted rows in the test period")
        continue
    # Highest-impact hour = full-severity hour, not a ramp hour
    row = matches.loc[matches[[f"{p}_Impact_Score" for p in PLANTS]].sum(axis=1).idxmax()]
    print(f"\n[{row['Timestamp']}] Predicted: {disruption} (confidence {row['Prediction_Confidence']:.3f}, "
          f"actual: {row['System_Status']})")
    print(f"  Supply drop        : {row['Supply_Drop_Percent']:.1f}%")
    print("  Impact levels      : " + ", ".join(
        f"{p} {row[f'{p}_Impact_Level']} ({row[f'{p}_Impact_Score']:.1f})" for p in PLANTS))
    print(f"  Affected facilities: {row['Affected_Facilities']}")
    print(f"  System severity    : {row['System_Severity']}")
    print(f"  Recommendation     : {row['Recommendation']}")
    print(f"  Top SHAP sensors   : {row['SHAP_Top_Contributors']}")


# =========================
# 7. Save Outputs for Streamlit
# =========================

df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

df[is_anomalous].to_csv(RESULT_DIR / "impact_assessment_results.csv", index=False)
df.to_csv(RESULT_DIR / "dashboard_master_test.csv", index=False)
sensitivity_df.to_csv(RESULT_DIR / "impact_threshold_sensitivity.csv", index=False)
importance_df.to_csv(RESULT_DIR / "shap_global_importance.csv")

joblib.dump({
    "shap_values": shap_values,              # (rows, features, classes), rows aligned with dashboard_master_test.csv
    "expected_value": np.asarray(explainer.expected_value),
    "feature_names": features,
    "class_names": class_names,
    "timestamps": df["Timestamp"].tolist(),
    "X_test": X_test.reset_index(drop=True),
}, RESULT_DIR / "shap_artifacts.pkl")

print(f"\nFiles saved in {RESULT_DIR}:")
for name in ["impact_assessment_results.csv", "dashboard_master_test.csv", "impact_threshold_sensitivity.csv",
             "shap_global_importance.csv", "shap_global_summary.png", "shap_artifacts.pkl"]:
    print(f"- {name}")
