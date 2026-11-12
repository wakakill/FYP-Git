import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
import joblib

# Paths relative to this script, so it runs from any working directory
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "dataset" / "master_gas_supply_chain_dataset.csv"
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)


# =========================
# 1. Load Dataset
# =========================

master_df = pd.read_csv(DATA_PATH)

# Convert timestamp and sort chronologically
master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"])
master_df = master_df.sort_values("Timestamp").reset_index(drop=True)


# =========================
# 2. Select Features
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

# Check missing columns
missing_cols = [col for col in features if col not in master_df.columns]

if missing_cols:
    print("Missing columns:")
    print(missing_cols)
    print("\nAvailable columns:")
    print(master_df.columns.tolist())
    raise ValueError("Some selected features are missing from the dataset.")


# =========================
# 3. Chronological Train-Test Split 70/30
# =========================

split_index = int(len(master_df) * 0.7)

train_df = master_df.iloc[:split_index].copy()
test_df = master_df.iloc[split_index:].copy()

# Only sensor features go to the models; no Status / label columns
X_train = train_df[features]
X_test = test_df[features]

y_test = test_df["GroundTruth_Anomaly"]

# Contamination from the training period only (no test-set leakage)
train_anomaly_rate = train_df["GroundTruth_Anomaly"].mean()

print("\n===== Dataset Split =====")
print(f"Total rows    : {len(master_df)}")
print(f"Training rows : {len(train_df)}  ({train_df['Timestamp'].min()} to {train_df['Timestamp'].max()})")
print(f"Testing rows  : {len(test_df)}  ({test_df['Timestamp'].min()} to {test_df['Timestamp'].max()})")
print(f"Training anomaly rate (contamination): {train_anomaly_rate:.4f}")

print("\n===== Test Set Anomaly Distribution =====")
print(y_test.value_counts())


# =========================
# 4. Feature Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 5. Train the Three Detectors (on X_train_scaled only)
# =========================

# a) Isolation Forest (primary candidate)
iso_forest = IsolationForest(
    n_estimators=300,
    contamination=train_anomaly_rate,
    random_state=42
)
iso_forest.fit(X_train_scaled)

# b) Local Outlier Factor (density-based baseline)
lof = LocalOutlierFactor(
    n_neighbors=20,
    novelty=True,
    contamination=train_anomaly_rate
)
lof.fit(X_train_scaled)

# c) Autoencoder (reconstruction-based baseline), trained on normal rows only
X_train_normal = X_train_scaled[(train_df["GroundTruth_Anomaly"] == 0).to_numpy()]

autoencoder = MLPRegressor(
    hidden_layer_sizes=(10, 5, 10),
    max_iter=500,
    random_state=42
)
autoencoder.fit(X_train_normal, X_train_normal)

train_normal_mse = np.mean((autoencoder.predict(X_train_normal) - X_train_normal) ** 2, axis=1)
ae_threshold = np.quantile(train_normal_mse, 1 - train_anomaly_rate)


# =========================
# 6. Predict on Test Data (timed)
# =========================

def predict_sklearn_detector(model):
    """IF / LOF: predict() gives 1 = Normal, -1 = Anomaly; higher -decision_function = more anomalous."""
    start = time.perf_counter()
    y_pred = (model.predict(X_test_scaled) == -1).astype(int)
    scores = -model.decision_function(X_test_scaled)
    return y_pred, scores, (time.perf_counter() - start) * 1000


def predict_autoencoder():
    start = time.perf_counter()
    scores = np.mean((autoencoder.predict(X_test_scaled) - X_test_scaled) ** 2, axis=1)
    y_pred = (scores > ae_threshold).astype(int)
    return y_pred, scores, (time.perf_counter() - start) * 1000


predictions = {
    "Isolation Forest": predict_sklearn_detector(iso_forest),
    "Local Outlier Factor": predict_sklearn_detector(lof),
    "Autoencoder (MLP)": predict_autoencoder(),
}


# =========================
# 6b. 2-Hour Alarm Persistence Rule
# =========================

# Rule lives in scoring.py (shared with the dashboard's Upload & Score page); it self-checks on import
from scoring import apply_persistence  # noqa: E402

for name in list(predictions):
    y_pred, scores, inference_ms = predictions[name]
    start = time.perf_counter()
    y_pred_2h = apply_persistence(y_pred)
    # Scores are unchanged by the rule, so ROC-AUC stays the same as the raw model
    predictions[f"{name} + 2h rule"] = (y_pred_2h, scores, inference_ms + (time.perf_counter() - start) * 1000)


# =========================
# 7. Standardized Evaluation & Comparison
# =========================

# Ground-truth events = runs of consecutive anomaly hours (events are >= 20 h apart)
y_test_arr = y_test.to_numpy()
event_edges = np.flatnonzero(np.diff(np.r_[0, y_test_arr, 0]))
events = list(zip(event_edges[::2], event_edges[1::2]))
test_days = len(y_test_arr) / 24

comparison_rows = []

for name, (y_pred, scores, inference_ms) in predictions.items():
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    events_detected = sum(y_pred[start:end].any() for start, end in events)
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Anomaly_Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Anomaly_Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Anomaly_F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, scores),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Events_Detected": events_detected,
        "Total_Events": len(events),
        "False_Alarm_Hours_Per_Day": fp / test_days,
        "Inference_Time_ms": inference_ms,
    }
    comparison_rows.append(row)

    print(f"\n===== {name} =====")
    for metric in ["Accuracy", "Anomaly_Precision", "Anomaly_Recall", "Anomaly_F1", "ROC_AUC"]:
        print(f"{metric:18}: {row[metric]:.4f}")
    print(f"Events detected   : {events_detected}/{len(events)}")
    print(f"False alarms/day  : {row['False_Alarm_Hours_Per_Day']:.2f} hours")
    print(f"Inference time    : {inference_ms:.2f} ms ({len(X_test)} test rows)")
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))

comparison_df = pd.DataFrame(comparison_rows).round(4)

print("\n===== Anomaly Detection Comparison =====")
print(comparison_df.to_string(index=False))


# =========================
# 8. Save Results (primary model = Isolation Forest)
# =========================

if_pred, if_scores, _ = predictions["Isolation Forest"]
test_df["Predicted_Anomaly"] = if_pred
test_df["Anomaly_Score"] = if_scores
test_df["Detection_Result"] = test_df["Predicted_Anomaly"].map({0: "Normal", 1: "Anomaly"})
test_df["Predicted_Anomaly_2h"] = predictions["Isolation Forest + 2h rule"][0]  # alarm after 2h persistence

comparison_df.to_csv(RESULT_DIR / "anomaly_detection_comparison.csv", index=False)
test_df.to_csv(RESULT_DIR / "test_dataset_with_anomaly_predictions.csv", index=False)

joblib.dump(iso_forest, RESULT_DIR / "isolation_forest_model.pkl")
joblib.dump(scaler, RESULT_DIR / "scaler.pkl")

print(f"\nFiles saved in {RESULT_DIR}:")
print("- anomaly_detection_comparison.csv")
print("- test_dataset_with_anomaly_predictions.csv")
print("- isolation_forest_model.pkl")
print("- scaler.pkl")
