import numpy as np
import pandas as pd

# ==========================
# 1. Read CSV Files
# ==========================

upstream = pd.read_csv("upstream_dataset.csv")
midstream = pd.read_csv("midstream_dataset.csv")
lng = pd.read_csv("downstream_LNG.csv")
cogen = pd.read_csv("downstream_cogen.csv")
pcg = pd.read_csv("downstream_pcg.csv")


# ==========================
# 2. Rename Columns
# ==========================

upstream = upstream.rename(columns={
    "Flow": "Upstream_Flow",
    "Pressure(bar)": "Upstream_Pressure",
    "Status": "Upstream_Status"
})

midstream = midstream.rename(columns={
    "Pipeline_Flow": "Midstream_Pipeline_Flow",
    "Compressor_Load(%)": "Midstream_Compressor_Load",
    "Status": "Midstream_Status"
})

lng = lng.rename(columns={
    "Gas_Supply_Received": "LNG_Gas_Supply_Received",
    "Plant_Efficiency": "LNG_Plant_Efficiency",
    "Plant_Output": "LNG_Plant_Output",
    "Status": "LNG_Status"
})

cogen = cogen.rename(columns={
    "Gas_Supply_Received": "COGEN_Gas_Supply_Received",
    "Plant_Efficiency": "COGEN_Plant_Efficiency",
    "Plant_Output": "COGEN_Plant_Output",
    "Status": "COGEN_Status"
})

pcg = pcg.rename(columns={
    "Gas_Supply_Received": "PCG_Gas_Supply_Received",
    "Plant_Efficiency": "PCG_Plant_Efficiency",
    "Plant_Output": "PCG_Plant_Output",
    "Status": "PCG_Status"
})


# ==========================
# 3. Remove Unnecessary Columns
# ==========================

# Remove Unit_ID columns (constant per file, not a sensor or label)
upstream = upstream.drop(columns=["Unit_ID"], errors="ignore")
midstream =midstream.drop(columns=["Unit_ID"], errors="ignore")
lng = lng.drop(columns=["Unit_ID"], errors="ignore")
cogen = cogen.drop(columns=["Unit_ID"], errors="ignore")
pcg = pcg.drop(columns=["Unit_ID"], errors="ignore")


# ==========================
# 4. Convert Timestamp
# ==========================

upstream["Timestamp"] = pd.to_datetime(upstream["Timestamp"])
midstream["Timestamp"] = pd.to_datetime(midstream["Timestamp"])
lng["Timestamp"] = pd.to_datetime(lng["Timestamp"])
cogen["Timestamp"] = pd.to_datetime(cogen["Timestamp"])
pcg["Timestamp"] = pd.to_datetime(pcg["Timestamp"])


# ==========================
# 5. Merge All Datasets
# ==========================

master_df = upstream.merge(
    midstream,
    on="Timestamp",
    how="inner"
)

master_df = master_df.merge(
    lng,
    on="Timestamp",
    how="inner"
)

master_df = master_df.merge(
    cogen,
    on="Timestamp",
    how="inner"
)

master_df = master_df.merge(
    pcg,
    on="Timestamp",
    how="inner"
)

master_df = master_df.sort_values("Timestamp").reset_index(drop=True)


# ==========================
# 6. Create Ground Truth Label
# ==========================

master_df["GroundTruth_Anomaly"] = (
    (master_df["Upstream_Status"] != "Normal") |
    (master_df["Midstream_Status"] != "Normal") |
    (master_df["LNG_Status"] != "Normal") |
    (master_df["COGEN_Status"] != "Normal") |
    (master_df["PCG_Status"] != "Normal")
).astype(int)


# ==========================
# 7. Create Summary Column
# ==========================

status_columns = [
    "Upstream_Status",
    "Midstream_Status",
    "LNG_Status",
    "COGEN_Status",
    "PCG_Status"
]


def summarize_status(row):
    anomalies = []

    for col in status_columns:
        if row[col] != "Normal":
            anomalies.append(row[col])

    if len(anomalies) == 0:
        return "Normal"

    return ", ".join(sorted(set(anomalies)))


master_df["GroundTruth_Status_Summary"] = master_df.apply(
    summarize_status,
    axis=1
)


# ==========================
# 7b. Create 6-Class Root-Cause Label
# ==========================

# Events never overlap, so each anomalous row has exactly one root cause
master_df["System_Status"] = np.select(
    [
        master_df["Upstream_Status"] == "Low_Pressure",
        master_df["Midstream_Status"] == "Compressor_Trip",
        master_df["LNG_Status"] == "Plant_Underperformance",
        master_df["COGEN_Status"] == "Plant_Underperformance",
        master_df["PCG_Status"] == "Plant_Underperformance",
    ],
    ["Low_Pressure", "Compressor_Trip", "LNG_Underperformance", "COGEN_Underperformance", "PCG_Underperformance"],
    "Normal"
)

assert ((master_df["System_Status"] != "Normal") == master_df["GroundTruth_Anomaly"].astype(bool)).all()


# ==========================
# 8. Save Master Dataset
# ==========================

master_df.to_csv(
    "master_gas_supply_chain_dataset.csv",
    index=False
)


# ==========================
# 9. Display Information
# ==========================

print("\n===== Merge Completed =====")
print("Master dataset shape:", master_df.shape)

print("\nColumns:")
print(master_df.columns.tolist())

print("\nGround Truth Distribution:")
print(master_df["GroundTruth_Anomaly"].value_counts())

print("\nSystem_Status Distribution:")
print(master_df["System_Status"].value_counts())

print("\nMaster dataset saved as:")
print("master_gas_supply_chain_dataset.csv")