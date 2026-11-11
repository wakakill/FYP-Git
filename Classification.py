import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier
import joblib


# =========================
# 1. Load Master Dataset
# =========================

df = pd.read_csv("master_gas_supply_chain_dataset.csv")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)


# =========================
# 2. Create Primary Anomaly Type
# =========================

def get_primary_anomaly(row):
    if row["Upstream_Status"] != "Normal":
        return row["Upstream_Status"]

    elif row["Midstream_Status"] != "Normal":
        return row["Midstream_Status"]

    elif row["LNG_Status"] != "Normal":
        return row["LNG_Status"]

    elif row["Power_Status"] != "Normal":
        return row["Power_Status"]

    elif row["PCG_Status"] != "Normal":
        return row["PCG_Status"]

    else:
        return "Normal"


df["Primary_Anomaly_Type"] = df.apply(get_primary_anomaly, axis=1)


# =========================
# 3. Select Features
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

X = df[features]
y = df["Primary_Anomaly_Type"]


# =========================
# 4. Encode Target Labels
# =========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n===== Class Mapping =====")
for class_id, class_name in enumerate(label_encoder.classes_):
    print(f"{class_id}: {class_name}")


print("\n===== Class Distribution =====")
print(y.value_counts())


# =========================
# 5. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================
# 6. Train XGBoost Classifier
# =========================

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="multi:softmax",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 7. Predict
# =========================

y_pred = model.predict(X_test)


# =========================
# 8. Evaluation
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== Accuracy =====")
print(accuracy)

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== Classification Report =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# =========================
# 9. Save Prediction Results
# =========================

result_df = X_test.copy()

result_df["Actual_Label_Encoded"] = y_test
result_df["Predicted_Label_Encoded"] = y_pred

result_df["Actual_Anomaly_Type"] = label_encoder.inverse_transform(y_test)
result_df["Predicted_Anomaly_Type"] = label_encoder.inverse_transform(y_pred)

result_df.to_csv(
    "xgboost_classification_results.csv",
    index=False
)


# =========================
# 10. Save Model and Encoder
# =========================

joblib.dump(model, "xgboost_anomaly_classifier.pkl")
joblib.dump(label_encoder, "xgboost_label_encoder.pkl")

print("\nFiles saved:")
print("- xgboost_classification_results.csv")
print("- xgboost_anomaly_classifier.pkl")
print("- xgboost_label_encoder.pkl")