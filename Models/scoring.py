"""Score a new dataset with the already-trained models (no retraining).
Shared by Train_Anomaly.py (persistence rule) and app.py (Upload & Score page).

Pipeline per hour: scaler -> Isolation Forest -> 2-hour persistence filter -> XGBoost -> impact rules -> SHAP."""

import numpy as np
import pandas as pd

from impact_rules import CONFIRMED_COL, assess_impact

# Sensor features, in the order the models were trained on (Train_Anomaly.py / Train_Classifier.py)
FEATURES = [
    "Upstream_Flow", "Upstream_Pressure",
    "Midstream_Pipeline_Flow", "Midstream_Compressor_Load",
    "LNG_Gas_Supply_Received", "LNG_Plant_Efficiency", "LNG_Plant_Output",
    "COGEN_Gas_Supply_Received", "COGEN_Plant_Efficiency", "COGEN_Plant_Output",
    "PCG_Gas_Supply_Received", "PCG_Plant_Efficiency", "PCG_Plant_Output",
]

PERSISTENCE_HOURS = 2


def apply_persistence(y_pred, min_hours=PERSISTENCE_HOURS):
    """Keep only flags that belong to a run of >= min_hours consecutive flagged hours.
    In live monitoring the alarm is raised at the 2nd consecutive hour (1-hour confirmation delay);
    once confirmed, the first hour is marked as part of the same alarm."""
    y_pred = np.asarray(y_pred)
    edges = np.flatnonzero(np.diff(np.r_[0, y_pred, 0]))
    filtered = np.zeros_like(y_pred)
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= min_hours:
            filtered[start:end] = 1
    return filtered


# Self-check: single-hour flags are removed, runs of 2+ hours are kept
assert (apply_persistence(np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1]))
        == np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])).all()


def validate(raw):
    """Check an uploaded table. Returns (clean DataFrame sorted by time, list of errors, list of warnings)."""
    errors, warnings = [], []
    missing = [c for c in ["Timestamp"] + FEATURES if c not in raw.columns]
    if missing:
        return None, [f"Missing required column(s): {', '.join(missing)}"], warnings

    df = raw.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        errors.append(f"{int(df['Timestamp'].isna().sum())} row(s) have a Timestamp that cannot be read "
                      "(expected e.g. 2025-01-01 00:00:00).")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[FEATURES].isna().sum()
    if bad.any():
        errors.append("Empty or non-numeric sensor values in: "
                      + ", ".join(f"{c} ({n} rows)" for c, n in bad[bad > 0].items()))
    if len(df) < PERSISTENCE_HOURS:
        errors.append(f"The file needs at least {PERSISTENCE_HOURS} hourly rows.")
    if errors:
        return None, errors, warnings

    df = df.sort_values("Timestamp").reset_index(drop=True)
    dupes = df["Timestamp"].duplicated()
    if dupes.any():
        return None, [f"{int(dupes.sum())} duplicate Timestamp(s), e.g. {df.loc[dupes, 'Timestamp'].iloc[0]}. "
                      "Each hour must appear once."], warnings

    step = df["Timestamp"].diff().dropna()
    if (step < pd.Timedelta(hours=1)).any():
        return None, ["Rows are closer together than 1 hour. The models expect one row per hour."], warnings
    gaps = int((step > pd.Timedelta(hours=1)).sum())
    if gaps:
        warnings.append(f"{gaps} gap(s) in the hourly timeline. The 2-hour rule is applied separately on each "
                        "side of a gap, so hours across a gap are never joined into one disruption.")
    return df, errors, warnings


def score(df, scaler, iso_forest, xgb_model, label_encoder, explainer=None):
    """Add the same prediction / impact columns as dashboard_master_test.csv to a validated DataFrame."""
    df = df.copy()
    X = df[FEATURES]
    X_scaled = scaler.transform(X[list(scaler.feature_names_in_)])
    df["Predicted_Anomaly"] = (iso_forest.predict(X_scaled) == -1).astype(int)
    df["Anomaly_Score"] = -iso_forest.decision_function(X_scaled)

    # 2-hour rule within each unbroken run of hourly rows, so a timeline gap never joins two flags
    segment = (df["Timestamp"].diff() != pd.Timedelta(hours=1)).cumsum()
    df[CONFIRMED_COL] = df.groupby(segment)["Predicted_Anomaly"].transform(lambda s: apply_persistence(s.to_numpy()))

    proba = xgb_model.predict_proba(X)
    pred = proba.argmax(axis=1)
    df["Prediction_Confidence"] = proba.max(axis=1)
    df["Predicted_Anomaly_Type"] = label_encoder.inverse_transform(pred)
    df = assess_impact(df)

    # Top 3 sensors pushing each confirmed hour toward its predicted class (SHAP), same format as ImpactAssessment.py
    df["SHAP_Top_Contributors"] = "None"
    confirmed = np.flatnonzero(df[CONFIRMED_COL].to_numpy() == 1)
    if explainer is not None and len(confirmed):
        shap_values = shap_matrix(explainer, X.iloc[confirmed], len(label_encoder.classes_))
        row_shap = shap_values[np.arange(len(confirmed)), :, pred[confirmed]]
        values = X.iloc[confirmed].to_numpy()
        df.loc[df.index[confirmed], "SHAP_Top_Contributors"] = [
            ", ".join(f"{FEATURES[j]}: {values[i, j]:.1f} ({row_shap[i, j]:+.2f})"
                      for j in np.argsort(-row_shap[i])[:3] if row_shap[i, j] >= 0.005) or "None"
            for i in range(len(confirmed))]
    return df


def shap_matrix(explainer, X, n_classes):
    """SHAP values as (rows, features, classes), whatever layout the installed SHAP version returns."""
    values = np.asarray(explainer.shap_values(X))
    if values.shape != (len(X), X.shape[1], n_classes):  # older SHAP: (classes, rows, features)
        values = np.transpose(values, (1, 2, 0))
    return values
