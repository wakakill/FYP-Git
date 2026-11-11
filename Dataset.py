import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Generate hourly timestamps starting Jan 1, 2026
def generate_timestamps(days=60):
    start = datetime(2026, 1, 1, 0, 0, 0)
    return [start + timedelta(hours=i) for i in range(days * 24)]

# Inject anomalies with consistent drop values
def inject_flow_anomalies(df, feature, label_col="Status"):
    n = len(df)
    anomaly_events = random.randint(8, 12)  # number of anomaly events
    for _ in range(anomaly_events):
        start_idx = random.randint(0, n-6)
        duration = random.randint(2, 5)
        drop_percent = random.choice([0.2, 0.3, 0.5])  # 20%, 30%, 50%
        anomaly_value = 200 * (1 - drop_percent)
        anomaly_series = np.random.normal(anomaly_value, 2, duration)
        df.loc[start_idx:start_idx+duration-1, feature] = anomaly_series
        df.loc[start_idx:start_idx+duration-1, label_col] = "Flow Drop"
    return df

# Upstream dataset
def generate_upstream():
    ts = generate_timestamps()
    df = pd.DataFrame({
        "Timestamp": ts,
        "Unit_ID": "Upstream",
        "Flow": np.random.normal(200, 2, len(ts)),
        "Pressure": np.random.normal(1000, 2, len(ts)),
        "Status": "Normal"
    })
    df = inject_flow_anomalies(df, "Flow")
    return df

# Midstream dataset (inherits upstream anomalies, adds compressor faults)
def generate_midstream(upstream_df):
    df = pd.DataFrame({
        "Timestamp": upstream_df["Timestamp"],
        "Unit_ID": "Midstream",
        "Pipeline_Flow": upstream_df["Flow"],  # propagate upstream flow directly
        "Compressor_Load": np.random.normal(75, 2, len(upstream_df)),
        "Status": "Normal"
    })
    # Propagation: if upstream anomaly, mark midstream anomaly
    df.loc[upstream_df["Status"].str.contains("Flow Drop"), "Status"] = "Flow Restriction"
    # Local anomaly: compressor fault
    df = inject_flow_anomalies(df, "Compressor_Load", label_col="Status")
    return df

# Downstream dataset (allocated by weightage)
def generate_downstream(midstream_df, unit_name, weight, efficiency_mean):
    df = pd.DataFrame({
        "Timestamp": midstream_df["Timestamp"],
        "Unit_ID": unit_name,
        "Gas_Supply_Received": midstream_df["Pipeline_Flow"] * weight,
        "Plant_Efficiency": np.random.normal(efficiency_mean, 2, len(midstream_df)),
        "Status": "Normal"
    })
    # Output calculation
    df["Plant_Output"] = df["Gas_Supply_Received"] * (df["Plant_Efficiency"] / 100)
    # Propagation: if midstream anomaly, mark downstream anomaly
    df.loc[midstream_df["Status"].str.contains("Flow Restriction"), "Status"] = "Supply Shortage"
    # Local anomaly: underperformance
    df = inject_flow_anomalies(df, "Plant_Efficiency", label_col="Status")
    # Recalculate output after efficiency anomalies
    df["Plant_Output"] = df["Gas_Supply_Received"] * (df["Plant_Efficiency"] / 100)
    return df

# Generate datasets
upstream_df = generate_upstream()
midstream_df = generate_midstream(upstream_df)
lng_df = generate_downstream(midstream_df, "LNG_Plant", 0.50, 90)
power_df = generate_downstream(midstream_df, "Power_Plant", 0.30, 85)
pcg_df = generate_downstream(midstream_df, "Petrochemical_Plant", 0.20, 80)

# Convert timestamps to ISO format
for df in [upstream_df, midstream_df, lng_df, power_df, pcg_df]:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

# Save to CSV
upstream_df.to_csv("upstream_dataset.csv", index=False)
midstream_df.to_csv("midstream_dataset.csv", index=False)
lng_df.to_csv("downstream_LNG.csv", index=False)
power_df.to_csv("downstream_power.csv", index=False)
pcg_df.to_csv("downstream_pcg.csv", index=False)
