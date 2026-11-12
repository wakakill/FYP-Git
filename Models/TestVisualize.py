import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Load Prediction Dataset
# =========================

df = pd.read_csv("test_dataset_with_anomaly_predictions.csv")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])


# =========================
# 2. Create Useful Columns
# =========================

df["Total_Downstream_Supply"] = (
    df["LNG_Gas_Supply_Received"] +
    df["COGEN_Gas_Supply_Received"] +
    df["PCG_Gas_Supply_Received"]
)

df["Total_Downstream_Output"] = (
    df["LNG_Plant_Output"] +
    df["COGEN_Plant_Output"] +
    df["PCG_Plant_Output"]
)

anomalies = df[df["Predicted_Anomaly"] == 1]


# =========================
# 3. Plot Upstream Flow
# =========================

plt.figure(figsize=(15, 6))

plt.plot(
    df["Timestamp"],
    df["Upstream_Flow"],
    label="Upstream Flow"
)

plt.scatter(
    anomalies["Timestamp"],
    anomalies["Upstream_Flow"],
    color="red",
    label="Detected Anomaly"
)

plt.title("Detected Anomalies in Upstream Flow")
plt.xlabel("Timestamp")
plt.ylabel("Flow (MMSCFD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 4. Plot Midstream Pipeline Flow
# =========================

plt.figure(figsize=(15, 6))

plt.plot(
    df["Timestamp"],
    df["Midstream_Pipeline_Flow"],
    label="Midstream Pipeline Flow"
)

plt.scatter(
    anomalies["Timestamp"],
    anomalies["Midstream_Pipeline_Flow"],
    color="red",
    label="Detected Anomaly"
)

plt.title("Detected Anomalies in Midstream Pipeline Flow")
plt.xlabel("Timestamp")
plt.ylabel("Pipeline Flow (MMSCFD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 5. Plot Downstream Supply
# =========================

plt.figure(figsize=(15, 6))

plt.plot(
    df["Timestamp"],
    df["LNG_Gas_Supply_Received"],
    label="LNG Supply Received"
)

plt.plot(
    df["Timestamp"],
    df["COGEN_Gas_Supply_Received"],
    label="COGEN Supply Received"
)

plt.plot(
    df["Timestamp"],
    df["PCG_Gas_Supply_Received"],
    label="PCG Supply Received"
)

plt.scatter(
    anomalies["Timestamp"],
    anomalies["Total_Downstream_Supply"],
    color="red",
    label="Detected Anomaly"
)

plt.title("Detected Anomalies Across Downstream Supply")
plt.xlabel("Timestamp")
plt.ylabel("Gas Supply Received (MMSCFD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 6. Plot Total Supply Chain Flow
# =========================

plt.figure(figsize=(15, 6))

plt.plot(
    df["Timestamp"],
    df["Upstream_Flow"],
    label="Upstream Flow"
)

plt.plot(
    df["Timestamp"],
    df["Midstream_Pipeline_Flow"],
    label="Midstream Pipeline Flow"
)

plt.plot(
    df["Timestamp"],
    df["Total_Downstream_Supply"],
    label="Total Downstream Supply"
)

plt.plot(
    df["Timestamp"],
    df["Total_Downstream_Output"],
    label="Total Downstream Output"
)

plt.scatter(
    anomalies["Timestamp"],
    anomalies["Upstream_Flow"],
    color="red",
    label="Detected Anomaly"
)

plt.title("Gas Supply Chain Flow Monitoring with Detected Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Flow / Output (MMSCFD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 7. Plot Anomaly Timeline
# =========================

plt.figure(figsize=(15, 4))

plt.scatter(
    df["Timestamp"],
    df["Predicted_Anomaly"],
    label="Predicted Anomaly"
)

plt.title("Anomaly Detection Timeline")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Status")
plt.yticks([0, 1], ["Normal", "Anomaly"])
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 8. Print Anomaly Summary
# =========================

print("\n===== Anomaly Summary =====")
print(f"Total records: {len(df)}")
print(f"Detected anomalies: {len(anomalies)}")
print(f"Normal records: {len(df) - len(anomalies)}")

print("\n===== Detected Anomaly Records =====")
print(
    anomalies[
        [
            "Timestamp",
            "Upstream_Flow",
            "Midstream_Pipeline_Flow",
            "LNG_Gas_Supply_Received",
            "COGEN_Gas_Supply_Received",
            "PCG_Gas_Supply_Received",
            "GroundTruth_Status_Summary",
            "Detection_Result"
        ]
    ]
)