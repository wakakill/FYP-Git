import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# =========================
# 1. Load Dataset
# =========================

master_df = pd.read_csv("master_gas_supply_chain_dataset.csv")

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

    "Power_Gas_Supply_Received",
    "Power_Plant_Efficiency",
    "Power_Plant_Output",

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
# 3. Train-Test Split 80/20
# =========================

split_index = int(len(master_df) * 0.8)

train_df = master_df.iloc[:split_index].copy()
test_df = master_df.iloc[split_index:].copy()

X_train = train_df[features]
X_test = test_df[features]

y_test = test_df["GroundTruth_Anomaly"]


# =========================
# 4. Feature Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 5. Train Isolation Forest
# =========================

model = IsolationForest(
    n_estimators=300,
    contamination=0.28,
    random_state=42
)

model.fit(X_train_scaled)


# =========================
# 6. Predict on Test Data
# =========================

predictions = model.predict(X_test_scaled)

# Isolation Forest output:
#  1 = Normal
# -1 = Anomaly
test_df["Predicted_Anomaly"] = predictions
test_df["Predicted_Anomaly"] = test_df["Predicted_Anomaly"].map({
    1: 0,
    -1: 1
})

test_df["Detection_Result"] = test_df["Predicted_Anomaly"].map({
    0: "Normal",
    1: "Anomaly"
})


# =========================
# 7. Evaluation
# =========================

y_pred = test_df["Predicted_Anomaly"]

print("\n===== Dataset Split =====")
print(f"Total rows    : {len(master_df)}")
print(f"Training rows : {len(train_df)}")
print(f"Testing rows  : {len(test_df)}")

print("\n===== Test Set Anomaly Distribution =====")
print(test_df["GroundTruth_Anomaly"].value_counts())

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

print("\n===== Accuracy =====")
print(accuracy_score(y_test, y_pred))


# =========================
# 8. Save Results
# =========================

test_df.to_csv("test_dataset_with_anomaly_predictions.csv", index=False)

joblib.dump(model, "isolation_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nFiles saved:")
print("- test_dataset_with_anomaly_predictions.csv")
print("- isolation_forest_model.pkl")
print("- scaler.pkl")