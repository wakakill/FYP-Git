"""Realistic variant of Dataset.py: higher SCADA sensor noise, 1-hour normal transients and
1-hour ramp-down / recovery transitions. Fault severity, timeline, gap and event counts are unchanged.
Saves the same 5 CSVs, then runs Merge.py -> Train_Anomaly.py -> Train_Classifier.py."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# 1. Configuration
# =========================

SEED = 42
DAYS = 365            # 1 year = 8,760 hourly rows
MIN_GAP = 20           # normal hours required between the end of one event and the start of the next, across ALL stages
EVENT_HOURS = (5, 7)   # every event lasts 5-7 hours (inclusive)

# Same balanced counts as Dataset.py: 90 events x ~6 h ≈ 6.2% of the year
EVENT_COUNTS = {
    "Upstream": 18,     # Low_Pressure
    "Midstream": 18,    # Compressor_Trip
    "LNG_Plant": 18,    # Plant_Underperformance -> LNG_Underperformance
    "COGEN_Plant": 18,  # -> COGEN_Underperformance
    "PCG_Plant": 18,    # -> PCG_Underperformance
}

# (unit, allocation weight, normal efficiency mean, output file)
PLANTS = [
    ("LNG_Plant", 0.50, 90, "downstream_LNG.csv"),
    ("COGEN_Plant", 0.30, 85, "downstream_cogen.csv"),
    ("PCG_Plant", 0.20, 82, "downstream_pcg.csv"),
]

# Higher SCADA sensor noise (std), applied to normal and abnormal readings alike
PRESSURE_STD = 3.5
FLOW_STD = 5.5
COMPRESSOR_STD = 3.5
EFFICIENCY_STD = 3.5

# 1-hour normal transients (stay labelled Normal, do not propagate)
N_TRANSIENTS = 20
TRANSIENT_PRESSURE = (58, 65)     # bar
TRANSIENT_EFFICIENCY = (62, 70)   # %
TRANSIENT_CLEARANCE = 3           # transients stay at least 3 rows away from any event so they never touch a ramp

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "Models"

rng = np.random.default_rng(SEED)
timestamps = pd.date_range("2025-01-01 00:00:00", periods=DAYS * 24, freq="h")
n = len(timestamps)


# =========================
# 2. Event Scheduler (shared timeline, same as Dataset.py)
# =========================

def schedule_events():
    """Place every event on one timeline so that no two events are within MIN_GAP rows of each other."""
    events = [(unit, int(rng.integers(EVENT_HOURS[0], EVENT_HOURS[1] + 1)))
              for unit, count in EVENT_COUNTS.items() for _ in range(count)]
    rng.shuffle(events)

    durations = [d for _, d in events]
    slack = n - sum(durations) - MIN_GAP * (len(events) - 1)
    if slack < 0:
        raise ValueError("Too many events to fit in the timeline with the required gap.")

    # Spread the spare hours randomly: sorted offsets keep each event >= MIN_GAP after the previous one
    offsets = np.sort(rng.integers(0, slack + 1, len(events)))

    masks = {unit: np.zeros(n, dtype=bool) for unit in EVENT_COUNTS}
    used = 0
    for i, ((unit, dur), offset) in enumerate(zip(events, offsets)):
        start = offset + used + i * MIN_GAP
        masks[unit][start:start + dur] = True
        used += dur

    # Self-check: events are separate (none merged/overlapped) and every gap is >= MIN_GAP
    any_event = np.logical_or.reduce(list(masks.values()))
    edges = np.flatnonzero(np.diff(np.r_[0, any_event.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    assert len(starts) == len(events), "Events overlapped or touched"
    assert (starts[1:] - ends[:-1] >= MIN_GAP).all(), "Gap between events is below MIN_GAP"
    return masks, any_event


def severity(mask):
    """1.0 inside an event, 0.5 on its first (ramp-down) and last (recovery) hour, 0.0 elsewhere."""
    sev = mask.astype(float)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    sev[edges[::2]] = 0.5        # first hour of each event
    sev[edges[1::2] - 1] = 0.5   # last hour of each event
    return sev


def blend(normal, fault, sev):
    """Normal reading where sev=0, full fault where sev=1, halfway on ramp hours."""
    return normal + sev * (fault - normal)


masks, any_event = schedule_events()
low_pressure = masks["Upstream"]
compressor_trip = masks["Midstream"]


# =========================
# 3. Pick 1-Hour Normal Transients
# =========================

near_event = np.convolve(any_event, np.ones(2 * TRANSIENT_CLEARANCE + 1), mode="same") > 0
transient_rows = rng.choice(np.flatnonzero(~near_event), N_TRANSIENTS, replace=False)
pressure_transients = transient_rows[:N_TRANSIENTS // 2]
efficiency_transients = transient_rows[N_TRANSIENTS // 2:]
efficiency_transient_plant = rng.integers(0, len(PLANTS), len(efficiency_transients))


# =========================
# 4. Upstream
# =========================

lp_sev = severity(low_pressure)

pressure = blend(rng.normal(78.5, PRESSURE_STD, n), rng.normal(38.5, PRESSURE_STD, n), lp_sev)
pressure[pressure_transients] = rng.uniform(*TRANSIENT_PRESSURE, len(pressure_transients))  # flow unaffected

upstream_df = pd.DataFrame({
    "Timestamp": timestamps,
    "Unit_ID": "Upstream",
    "Flow": blend(rng.normal(200, FLOW_STD, n), rng.normal(100, FLOW_STD, n), lp_sev),
    "Pressure(bar)": pressure,
    "Status": np.where(low_pressure, "Low_Pressure", "Normal"),
})


# =========================
# 5. Midstream
# =========================

trip_sev = severity(compressor_trip)

midstream_df = pd.DataFrame({
    "Timestamp": timestamps,
    "Unit_ID": "Midstream",
    # Pipeline carries upstream flow; a full trip blocks it to a 0-3 residual (a stopped machine reads ~0, so no extra noise)
    "Pipeline_Flow": blend(upstream_df["Flow"].to_numpy(), rng.uniform(0, 3, n), trip_sev),
    "Compressor_Load(%)": blend(rng.normal(80, COMPRESSOR_STD, n), rng.uniform(0, 2, n), trip_sev),
    "Status": np.select([low_pressure, compressor_trip], ["Flow_Restriction", "Compressor_Trip"], "Normal"),
})


# =========================
# 6. Downstream
# =========================

supply_shortage = low_pressure | compressor_trip
downstream = {}

for plant_idx, (unit, weight, eff_mean, path) in enumerate(PLANTS):
    underperformance = masks[unit]
    # Allocation follows midstream flow, so shortages (and their ramps) propagate proportionally
    supply = midstream_df["Pipeline_Flow"].to_numpy() * weight * rng.uniform(0.98, 1.02, n)
    efficiency = blend(rng.normal(eff_mean, EFFICIENCY_STD, n), rng.normal(20, EFFICIENCY_STD, n),
                       severity(underperformance))

    rows = efficiency_transients[efficiency_transient_plant == plant_idx]
    efficiency[rows] = rng.uniform(*TRANSIENT_EFFICIENCY, len(rows))

    downstream[path] = pd.DataFrame({
        "Timestamp": timestamps,
        "Unit_ID": unit,
        "Gas_Supply_Received": supply,
        "Plant_Efficiency": efficiency,
        "Plant_Output": supply * (efficiency / 100),
        "Status": np.select([supply_shortage, underperformance],
                            ["Supply_Shortage", "Plant_Underperformance"], "Normal"),
    })


# =========================
# 7. Save and Summarize
# =========================

datasets = {"upstream_dataset.csv": upstream_df, "midstream_dataset.csv": midstream_df, **downstream}

# Self-check: every transient row is still Normal in all 5 datasets
assert all((df["Status"].to_numpy()[transient_rows] == "Normal").all() for df in datasets.values())

print("===== Status Summary =====")
for path, df in datasets.items():
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(BASE_DIR / path, index=False)
    print(f"\n{path} ({len(df)} rows)")
    print(df["Status"].value_counts().to_string())

print(f"\nNormal transients: {len(pressure_transients)} pressure dips, {len(efficiency_transients)} efficiency dips")


# =========================
# 8. Re-Run Pipeline
# =========================

# for script, cwd in [(BASE_DIR / "Merge.py", BASE_DIR),
#                     (MODELS_DIR / "Train_Anomaly.py", MODELS_DIR),
#                     (MODELS_DIR / "Train_Classifier.py", MODELS_DIR)]:
#     print(f"\n>>> Running {script.name} ...")
#     # Output captured so only the comparison tables are shown; a failing step prints its error and stops
#     result = subprocess.run([sys.executable, str(script)], cwd=cwd, capture_output=True, text=True)
#     if result.returncode != 0:
#         print(result.stdout[-3000:], result.stderr[-3000:])
#         raise SystemExit(f"{script.name} failed")

# for name in ["anomaly_detection_comparison.csv", "classification_comparison.csv"]:
#     print(f"\n===== {name} =====")
#     print(pd.read_csv(MODELS_DIR / "result" / name).to_string(index=False))
