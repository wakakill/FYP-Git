import numpy as np
import pandas as pd

# =========================
# 1. Configuration
# =========================

SEED = 42
DAYS = 365            # 1 year = 8,760 hourly rows
MIN_GAP = 20           # normal hours required between the end of one event and the start of the next, across ALL stages
EVENT_HOURS = (5, 7)   # every event lasts 5-7 hours (inclusive)

# Independent root-cause events per System_Status class: equal counts keep the 5 fault classes balanced,
# and 90 events x ~6 h ≈ 540 rows ≈ 6.2% of the year
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

rng = np.random.default_rng(SEED)
timestamps = pd.date_range("2025-01-01 00:00:00", periods=DAYS * 24, freq="h")
n = len(timestamps)


# =========================
# 2. Event Scheduler (shared timeline)
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
    return masks


masks = schedule_events()
low_pressure = masks["Upstream"]
compressor_trip = masks["Midstream"]


# =========================
# 3. Upstream
# =========================

upstream_df = pd.DataFrame({
    "Timestamp": timestamps,
    "Unit_ID": "Upstream",
    "Flow": np.where(low_pressure, rng.normal(100, 2, n), rng.normal(200, 2, n)),
    "Pressure(bar)": np.where(low_pressure, rng.normal(38.5, 1.2, n), rng.normal(78.5, 1.5, n)),
    "Status": np.where(low_pressure, "Low_Pressure", "Normal"),
})


# =========================
# 4. Midstream
# =========================

midstream_df = pd.DataFrame({
    "Timestamp": timestamps,
    "Unit_ID": "Midstream",
    # Pipeline carries upstream flow; a compressor trip blocks it to a 0-3 residual
    "Pipeline_Flow": np.where(compressor_trip, rng.uniform(0, 3, n), upstream_df["Flow"]),
    "Compressor_Load(%)": np.where(compressor_trip, rng.uniform(0, 2, n), rng.normal(80, 2, n)),
    "Status": np.select([low_pressure, compressor_trip], ["Flow_Restriction", "Compressor_Trip"], "Normal"),
})


# =========================
# 5. Downstream
# =========================

supply_shortage = low_pressure | compressor_trip
downstream = {}

for unit, weight, eff_mean, path in PLANTS:
    underperformance = masks[unit]
    # Allocation follows midstream flow, so shortages are proportional (~half or ~0) automatically
    supply = midstream_df["Pipeline_Flow"] * weight * rng.uniform(0.98, 1.02, n)
    efficiency = np.where(underperformance, rng.normal(20, 2, n), rng.normal(eff_mean, 1.5, n))

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
# 6. Save and Summarize
# =========================

datasets = {"upstream_dataset.csv": upstream_df, "midstream_dataset.csv": midstream_df, **downstream}

print("===== Status Summary =====")
for path, df in datasets.items():
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(path, index=False)
    print(f"\n{path} ({len(df)} rows)")
    print(df["Status"].value_counts().to_string())
