import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

from xgboost import XGBClassifier
import joblib

# Same paths as Train_Anomaly.py so both scripts use the identical master file and split
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "dataset" / "master_gas_supply_chain_dataset.csv"
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)


# =========================
# 1. Load Master Dataset
# =========================

df = pd.read_csv(DATA_PATH)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)


# =========================
# 2-3. Select Features (target is the 6-class System_Status built in Merge.py)
# =========================

features = [
    "Upstream_Flow",
    "Upstream_Pressure",

    "Midstream_Pipeline_Flow",
    "Midstream_Compressor_Load",

    "LNG_Gas_Supply_Received",
    "LNG_Plant_Efficiency",
    "LNG_Plant_Output",

    "COGEN_Gas_Supply_Received",
    "COGEN_Plant_Efficiency",
    "COGEN_Plant_Output",

    "PCG_Gas_Supply_Received",
    "PCG_Plant_Efficiency",
    "PCG_Plant_Output"
]

X = df[features]
y = df["System_Status"]


# =========================
# 4. Encode Target Labels
# =========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
class_ids = np.arange(len(class_names))

print("\n===== Class Mapping =====")
for class_id, class_name in enumerate(class_names):
    print(f"{class_id}: {class_name}")


# =========================
# 5. Chronological Train-Test Split 70/30 (identical to Train_Anomaly.py)
# =========================

split_index = int(len(df) * 0.7)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y_encoded[:split_index], y_encoded[split_index:]
test_timestamps = df["Timestamp"].iloc[split_index:]

print("\n===== Dataset Split =====")
print(f"Training rows : {len(X_train)}  ({df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[split_index - 1]})")
print(f"Testing rows  : {len(X_test)}  ({test_timestamps.iloc[0]} to {test_timestamps.iloc[-1]})")

distribution = pd.DataFrame({
    "Train": pd.Series(label_encoder.inverse_transform(y_train)).value_counts(),
    "Test": pd.Series(label_encoder.inverse_transform(y_test)).value_counts(),
}).reindex(class_names).fillna(0).astype(int)

print("\n===== Class Distribution per Split =====")
print(distribution)

missing = distribution[(distribution == 0).any(axis=1)].index.tolist()
if missing:
    raise ValueError(f"Classes missing from a split: {missing}. Spread events more evenly in Dataset.py.")


# =========================
# 6. Feature Scaling (SVM only)
# =========================

# Fitted on X_train only (identical to Train_Anomaly.py's scaler.pkl, same rows and features).
# Tree models (XGBoost, Random Forest) use unscaled X_train/X_test; scaling does not change tree splits.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 7. Train the Three Classifiers
# =========================

models = {
    "XGBoost": (XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    ), X_train, X_test),
    "Random Forest": (RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ), X_train, X_test),
    "SVM (RBF)": (SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ), X_train_scaled, X_test_scaled),
}

for name, (model, X_tr, _) in models.items():
    model.fit(X_tr, y_train)


# =========================
# 8. Evaluation & Comparison
# =========================

comparison_rows = []
predictions = {}

for name, (model, _, X_te) in models.items():
    start = time.perf_counter()
    y_pred = model.predict(X_te)
    inference_ms = (time.perf_counter() - start) * 1000
    predictions[name] = y_pred

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro_Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Inference_Time_ms": inference_ms,
    }
    comparison_rows.append(row)

    print(f"\n===== {name} =====")
    for metric in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"]:
        print(f"{metric:16}: {row[metric]:.4f}")
    print(f"Inference time  : {inference_ms:.2f} ms ({len(X_te)} test rows)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=class_ids, target_names=class_names, zero_division=0))

    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=class_ids),
                       index=class_names, columns=class_ids).to_string())

comparison_df = pd.DataFrame(comparison_rows).round(4)

print("\n===== Classification Comparison =====")
print(comparison_df.to_string(index=False))


# =========================
# 9. Stress Test: Partial Compressor Degradation (unseen scenario)
# =========================

rng = np.random.default_rng(42)
trip_rows = X_test[label_encoder.inverse_transform(y_test) == "Compressor_Trip"]
altered = trip_rows.copy()
n_alt = len(altered)

# Compressor degrades to ~45% instead of tripping; pipeline still carries ~115 instead of ~0
altered["Midstream_Compressor_Load"] = rng.normal(45, 2, n_alt)
altered["Midstream_Pipeline_Flow"] = rng.normal(115, 2, n_alt)

for plant, weight in [("LNG", 0.50), ("COGEN", 0.30), ("PCG", 0.20)]:
    altered[f"{plant}_Gas_Supply_Received"] = altered["Midstream_Pipeline_Flow"] * weight
    altered[f"{plant}_Plant_Output"] = altered[f"{plant}_Gas_Supply_Received"] * altered[f"{plant}_Plant_Efficiency"] / 100

altered_scaled = scaler.transform(altered)

print(f"\n===== Stress Test: Partial Compressor Degradation ({n_alt} altered Compressor_Trip rows) =====")
for name, (model, _, _) in models.items():
    X_alt = altered_scaled if name.startswith("SVM") else altered
    breakdown = pd.Series(label_encoder.inverse_transform(model.predict(X_alt))).value_counts()
    print(f"\n{name}:")
    print(breakdown.to_string())


# =========================
# 10. Save Prediction Results (primary model = XGBoost)
# =========================

xgb_model = models["XGBoost"][0]
xgb_pred = predictions["XGBoost"]
xgb_confidence = xgb_model.predict_proba(X_test).max(axis=1)

result_df = X_test.copy()
result_df.insert(0, "Timestamp", test_timestamps.dt.strftime("%Y-%m-%d %H:%M:%S").values)
result_df["Prediction_Confidence"] = xgb_confidence
result_df["Actual_Label_Encoded"] = y_test
result_df["Predicted_Label_Encoded"] = xgb_pred
result_df["Actual_Anomaly_Type"] = label_encoder.inverse_transform(y_test)
result_df["Predicted_Anomaly_Type"] = label_encoder.inverse_transform(xgb_pred)

comparison_df.to_csv(RESULT_DIR / "classification_comparison.csv", index=False)
result_df.to_csv(RESULT_DIR / "xgboost_classification_results.csv", index=False)

joblib.dump(xgb_model, RESULT_DIR / "xgboost_anomaly_classifier.pkl")
joblib.dump(label_encoder, RESULT_DIR / "xgboost_label_encoder.pkl")

print(f"\nFiles saved in {RESULT_DIR}:")
print("- classification_comparison.csv")
print("- xgboost_classification_results.csv")
print("- xgboost_anomaly_classifier.pkl")
print("- xgboost_label_encoder.pkl")


# =========================
# 11. Combined Isolation Forest + XGBoost Predictions (for SHAP, Impact Module, Streamlit)
# =========================

anomaly_path = RESULT_DIR / "test_dataset_with_anomaly_predictions.csv"

if anomaly_path.exists():
    anomaly_df = pd.read_csv(anomaly_path, parse_dates=["Timestamp"])
    xgb_cols = result_df[["Prediction_Confidence", "Predicted_Label_Encoded", "Predicted_Anomaly_Type"]].copy()
    xgb_cols["Timestamp"] = test_timestamps.values

    full_df = anomaly_df.merge(xgb_cols, on="Timestamp", how="inner", validate="one_to_one")
    if len(full_df) != len(result_df):
        raise ValueError("Isolation Forest and XGBoost test rows do not align. Re-run Train_Anomaly.py.")

    full_df.to_csv(RESULT_DIR / "test_dataset_full_predictions.csv", index=False)
    print("- test_dataset_full_predictions.csv")
else:
    print("\nSkipped test_dataset_full_predictions.csv: run Train_Anomaly.py first to create "
          "test_dataset_with_anomaly_predictions.csv.")
