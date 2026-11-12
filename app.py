"""Historical Gas Supply Chain Disruption Analysis & Decision Support Dashboard.

Offline analytical tool over the model test period (no live data, no control actions).
Shows confirmed disruptions only: Isolation Forest anomalies that persisted for 2+ consecutive hours.
Run from the project root:  streamlit run app.py
Requires the result files produced by Models/Train_Anomaly.py -> Train_Classifier.py -> ImpactAssessment.py.
"""

import html
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
RESULT_DIR = MODELS_DIR / "result"
sys.path.insert(0, str(MODELS_DIR))

from impact_rules import CONFIRMED_COL, PLANTS, LEVEL_ORDER, dependency_weights, assess_impact  # noqa: E402

st.set_page_config(page_title="Gas Supply Chain Disruption Analysis", layout="wide")


# =========================
# Theme constants & labels
# =========================

NAVY = "#1E3A5F"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
LINE = "#3B82F6"
CONFIRMED_RED = "#EF4444"
POSITIVE_SHAP = "#EF4444"
NEGATIVE_SHAP = "#93C5FD"
PLOT_CONFIG = {"displayModeBar": False}

# (text colour, soft background)
LEVEL_STYLE = {
    "Normal": ("#059669", "#ECFDF5"),
    "No Impact": ("#059669", "#ECFDF5"),
    "Low": ("#B45309", "#FEF3C7"),
    "Medium": ("#C2410C", "#FFEDD5"),
    "High": ("#B91C1C", "#FEE2E2"),
    "Confirmed": ("#B91C1C", "#FEE2E2"),
    "Filtered": ("#475569", "#F1F5F9"),
}

CONFIRMED_LABEL = "Confirmed Disruption (≥2h)"
TRANSIENT_LABEL = "Single-Hour Transient Spike (Filtered)"
NOT_FLAGGED_LABEL = "Not Flagged"

DISRUPTION_LABELS = {
    "Low_Pressure": "Upstream Low Pressure",
    "Compressor_Trip": "Midstream Compressor Trip",
    "LNG_Underperformance": "LNG Plant Underperformance",
    "COGEN_Underperformance": "COGEN / Power Plant Underperformance",
    "PCG_Underperformance": "PCG Plant Underperformance",
    "Unclassified": "Unclassified Anomaly (Isolation Forest only)",
    "Normal": "Normal Operation",
}

PLANT_LABELS = {"LNG": "LNG", "COGEN": "COGEN / Power", "PCG": "PCG"}

FEATURE_UNITS = {
    "Upstream_Pressure": "bar", "Midstream_Compressor_Load": "%",
    "LNG_Plant_Efficiency": "%", "COGEN_Plant_Efficiency": "%", "PCG_Plant_Efficiency": "%",
}

# Short sidebar label -> full page title
PAGES = {
    "Overview": "Supply Chain Telemetry Overview",
    "Event Inspector": "Event Diagnosis, Impact & SHAP Inspector",
    "Disruption Log & What-If": "Disruption Log & Offline Scenario Tester",
    "Model Benchmarks": "Model Evaluation & Temporal Filtering Benchmarks",
}

st.markdown(f"""
<style>
.block-container {{max-width: 1280px; padding-top: 2.4rem; padding-bottom: 3rem;}}
.page-title {{font-size: 1.65rem; font-weight: 700; color: {NAVY}; margin: 0;}}
.page-sub {{color: {MUTED}; font-size: 0.95rem; margin: 0.25rem 0 1.6rem 0;}}
.section {{font-size: 1.02rem; font-weight: 600; color: {TEXT}; margin: 0 0 0.5rem 0;}}
.kpi {{background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 20px;}}
.kpi-label {{font-size: 0.82rem; color: {MUTED}; font-weight: 500;}}
.kpi-value {{font-size: 2rem; font-weight: 700; color: {TEXT}; line-height: 1.25; margin-top: 4px;}}
.kpi-sub {{font-size: 0.8rem; color: {MUTED};}}
.pill {{display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        white-space: nowrap;}}
.chain {{display: flex; align-items: stretch; gap: 10px; flex-wrap: wrap; margin: 0.4rem 0 1.2rem 0;}}
.node {{flex: 1 1 160px; background: #FFFFFF; border: 1px solid {BORDER}; border-top: 4px solid;
        border-radius: 12px; padding: 14px 16px;}}
.node-title {{font-weight: 600; color: {TEXT}; font-size: 0.92rem;}}
.node-status {{font-size: 0.8rem; font-weight: 600; margin: 2px 0 10px 0;}}
.node-reading {{font-size: 0.82rem; color: {MUTED}; line-height: 1.7;}}
.node-reading b {{color: {TEXT}; font-weight: 600;}}
.node-recorded {{font-size: 0.72rem; color: #94A3B8; margin-top: 8px;}}
.arrow {{align-self: center; color: #CBD5E1; font-size: 1.3rem;}}
.stat-label {{font-size: 0.76rem; color: {MUTED};}}
.stat-value {{font-size: 1rem; font-weight: 600; color: {TEXT}; margin: 2px 0 14px 0;}}
.note {{background: #F8FAFC; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px;
        color: {TEXT}; font-size: 0.9rem; line-height: 1.55;}}
.impact-table {{width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 16px;}}
.impact-table th {{text-align: left; color: {MUTED}; font-weight: 500; font-size: 0.76rem;
                   border-bottom: 1px solid {BORDER}; padding: 6px 4px;}}
.impact-table td {{border-bottom: 1px solid #F1F5F9; padding: 9px 4px; color: {TEXT};}}
.advisory {{border-left: 4px solid; border-radius: 8px; padding: 12px 14px; font-size: 0.9rem; color: {TEXT};
            line-height: 1.55;}}
</style>
""", unsafe_allow_html=True)


# =========================
# Data & model loading
# =========================

REQUIRED_FILES = [
    "dashboard_master_test.csv", "shap_artifacts.pkl", "anomaly_detection_comparison.csv",
    "classification_comparison.csv", "impact_threshold_sensitivity.csv", "shap_global_importance.csv",
    "scaler.pkl", "isolation_forest_model.pkl", "xgboost_anomaly_classifier.pkl", "xgboost_label_encoder.pkl",
]
missing = [f for f in REQUIRED_FILES if not (RESULT_DIR / f).exists()]
if missing:
    st.error(f"Missing result files in {RESULT_DIR}: {', '.join(missing)}. Run Models/Train_Anomaly.py, "
             "Models/Train_Classifier.py and Models/ImpactAssessment.py first.")
    st.stop()


@st.cache_data
def load_dashboard():
    df = pd.read_csv(RESULT_DIR / "dashboard_master_test.csv", parse_dates=["Timestamp"])
    # pandas reads the literal text "None" as missing; restore it
    df[["Affected_Facilities", "SHAP_Top_Contributors"]] = df[["Affected_Facilities", "SHAP_Top_Contributors"]].fillna("None")
    return df


@st.cache_data
def build_events(df):
    """Confirmed hours grouped into events: each unbroken run of confirmed (>=2h) hours is one event."""
    hours = df[df[CONFIRMED_COL] == 1].copy()
    hours["Disruption"] = hours["Predicted_Anomaly_Type"].where(hours["Predicted_Anomaly_Type"] != "Normal",
                                                                "Unclassified")
    hours["Event"] = (hours["Timestamp"].diff() != pd.Timedelta(hours=1)).cumsum()

    def main_type(s):  # ramp hours can be classified Normal; name the event by its classified hours
        typed = s[s != "Unclassified"]
        return typed.mode().iloc[0] if len(typed) else "Unclassified"

    events = hours.groupby("Event").agg(
        Start=("Timestamp", "min"), End=("Timestamp", "max"), Hours=("Timestamp", "size"),
        Disruption=("Disruption", main_type),
        Severity=("System_Severity", lambda s: max(s, key=LEVEL_ORDER.index)),
        Peak_Supply_Drop=("Supply_Drop_Percent", "max"),
        Peak_Confidence=("Prediction_Confidence", "max"),
        Affected=("Affected_Facilities", lambda s: s.mode().iloc[0]),
    )
    return hours, events


@st.cache_data
def load_result_csv(name):
    return pd.read_csv(RESULT_DIR / name)


@st.cache_resource
def load_models():
    artifacts = joblib.load(RESULT_DIR / "shap_artifacts.pkl")
    xgb_model = joblib.load(RESULT_DIR / "xgboost_anomaly_classifier.pkl")
    return {
        "scaler": joblib.load(RESULT_DIR / "scaler.pkl"),
        "iso_forest": joblib.load(RESULT_DIR / "isolation_forest_model.pkl"),
        "xgb": xgb_model,
        "label_encoder": joblib.load(RESULT_DIR / "xgboost_label_encoder.pkl"),
        "explainer": shap.TreeExplainer(xgb_model),
        "shap_values": artifacts["shap_values"],       # (rows, features, classes), rows aligned with dashboard CSV
        "features": artifacts["feature_names"],
        "classes": artifacts["class_names"],
    }


df = load_dashboard()
hours, events = build_events(df)
models = load_models()
FEATURES = models["features"]
CLASSES = models["classes"]

if len(df) != models["shap_values"].shape[0]:
    st.error("dashboard_master_test.csv and shap_artifacts.pkl are out of sync. Re-run Models/ImpactAssessment.py.")
    st.stop()

PERIOD = f"{df['Timestamp'].min():%d %b %Y} to {df['Timestamp'].max():%d %b %Y}"


# =========================
# HTML / chart helpers
# =========================

def pill(text, level):
    color, bg = LEVEL_STYLE[level]
    return f'<span class="pill" style="color:{color}; background:{bg};">{html.escape(text)}</span>'


def kpi_card(label, value, sub):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def fmt_feature(feature, value):
    unit = FEATURE_UNITS.get(feature, "MMSCFD")
    return f"{feature.replace('_', ' ')} = {value:.1f} {unit}"


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12), margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor=BORDER, zeroline=False)
    return fig


def detection_label(record):
    if record[CONFIRMED_COL] == 1:
        return CONFIRMED_LABEL, "Confirmed"
    if record["Predicted_Anomaly"] == 1:
        return TRANSIENT_LABEL, "Filtered"
    return NOT_FLAGGED_LABEL, "Normal"


def event_label(event):
    end = event.End + pd.Timedelta(hours=1)
    return (f"{event.Start:%d %b %Y, %H:%M} to {end:%H:%M}  ·  {DISRUPTION_LABELS[event.Disruption]}  ·  "
            f"{event.Severity} severity")


def render_chain(record):
    """5-node horizontal supply chain: Upstream -> Midstream -> LNG / COGEN / PCG."""
    disruption = record["Predicted_Anomaly_Type"]

    if disruption == "Low_Pressure":
        upstream = ("Disrupted: Low Pressure", "High")
        midstream = ("Restricted Flow", "Medium")
    elif disruption == "Compressor_Trip":
        upstream = ("Normal", "Normal")
        midstream = ("Disrupted: Compressor Trip", "High")
    else:
        upstream = midstream = ("Normal", "Normal")

    def node(title, status, readings, recorded):
        text, level = status
        color, _ = LEVEL_STYLE[level]
        rec = f'<div class="node-recorded">Recorded: {html.escape(str(recorded))}</div>' if recorded else ""
        lines = "<br>".join(f"{k} <b>{v}</b>" for k, v in readings)
        return (f'<div class="node" style="border-top-color:{color};"><div class="node-title">{title}</div>'
                f'<div class="node-status" style="color:{color};">{html.escape(text)}</div>'
                f'<div class="node-reading">{lines}</div>{rec}</div>')

    nodes = [
        node("Upstream", upstream,
             [("Pressure", f"{record['Upstream_Pressure']:.1f} bar"),
              ("Flow", f"{record['Upstream_Flow']:.1f} MMSCFD")],
             record.get("Upstream_Status")),
        node("Midstream", midstream,
             [("Pipeline flow", f"{record['Midstream_Pipeline_Flow']:.1f} MMSCFD"),
              ("Compressor load", f"{record['Midstream_Compressor_Load']:.1f}%")],
             record.get("Midstream_Status")),
    ]
    for p in PLANTS:
        level = record[f"{p}_Impact_Level"]
        status = ("No Impact", "No Impact") if level == "No Impact" else (f"{level} Impact", level)
        nodes.append(node(
            f"{PLANT_LABELS[p]} · {dependency_weights[p]:.0%} share", status,
            [("Supply", f"{record[f'{p}_Gas_Supply_Received']:.1f} MMSCFD"),
             ("Efficiency", f"{record[f'{p}_Plant_Efficiency']:.1f}%"),
             ("Output", f"{record[f'{p}_Plant_Output']:.1f}")],
            record.get(f"{p}_Status")))

    arrow = '<div class="arrow">&#8594;</div>'
    chain = nodes[0] + arrow + nodes[1] + arrow + "".join(nodes[2:])
    st.markdown(f'<div class="chain">{chain}</div>', unsafe_allow_html=True)


def render_diagnosis(record, shap_vector):
    """Left card: Isolation Forest + persistence, XGBoost type/confidence, SHAP bar chart and plain-English summary."""
    disruption = record["Predicted_Anomaly_Type"]
    det_text, det_level = detection_label(record)

    with st.container(border=True):
        section("AI Diagnosis & SHAP Explanation")
        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="stat-label">Isolation Forest anomaly score (&gt; 0 = anomalous)</div>'
            f'<div class="stat-value">{record["Anomaly_Score"]:+.3f}</div>'
            f'<div class="stat-label">Temporal persistence</div>'
            f'<div class="stat-value">{pill(det_text, det_level)}</div>', unsafe_allow_html=True)
        type_level = "Normal" if disruption == "Normal" else "High"
        c2.markdown(
            f'<div class="stat-label">XGBoost predicted disruption</div>'
            f'<div class="stat-value">{pill(DISRUPTION_LABELS[disruption], type_level)}</div>'
            f'<div class="stat-label">Prediction confidence</div>'
            f'<div class="stat-value">{record["Prediction_Confidence"]:.1%}</div>', unsafe_allow_html=True)

        order = np.argsort(-np.abs(shap_vector))[:6][::-1]  # top 6 by magnitude, largest at the top
        names = [FEATURES[j].replace("_", " ") for j in order]
        values = shap_vector[order]
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h",
            marker_color=[POSITIVE_SHAP if v > 0 else NEGATIVE_SHAP for v in values],
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.update_layout(title=dict(text="Top sensor contributions (SHAP)", font=dict(size=13)),
                          xaxis_title="red = supports the prediction · blue = argues against")
        st.plotly_chart(style_fig(fig, height=280), width="stretch", config=PLOT_CONFIG)

        drivers = [j for j in np.argsort(-shap_vector)[:3] if shap_vector[j] >= 0.005]
        evidence = "; ".join(f"{fmt_feature(FEATURES[j], record[FEATURES[j]])} (SHAP {shap_vector[j]:+.2f})"
                             for j in drivers) or "no single sensor dominated"
        summary = (f"XGBoost classified this hour as <b>{DISRUPTION_LABELS[disruption]}</b> with "
                   f"{record['Prediction_Confidence']:.1%} confidence. Strongest supporting evidence: {evidence}.")
        st.markdown(f'<div class="note">{summary}</div>', unsafe_allow_html=True)


def render_impact(record):
    """Right card: per-facility impact table, system severity and monitoring advisory."""
    severity = record["System_Severity"]
    with st.container(border=True):
        section("Downstream Impact Assessment")
        rows = "".join(
            f"<tr><td><b>{PLANT_LABELS[p]}</b></td><td>{dependency_weights[p]:.0%}</td>"
            f"<td>{record[f'{p}_Impact_Score']:.1f}</td>"
            f"<td>{pill(record[f'{p}_Impact_Level'], record[f'{p}_Impact_Level'])}</td></tr>"
            for p in PLANTS)
        st.markdown(
            f'<table class="impact-table"><tr><th>Facility</th><th>Gas share</th><th>Impact score</th>'
            f'<th>Impact level</th></tr>{rows}</table>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="stat-label">System severity</div>'
                    f'<div class="stat-value">{pill(severity, severity)}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-label">Pipeline supply drop</div>'
                    f'<div class="stat-value">{record["Supply_Drop_Percent"]:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-label">Affected facilities</div>'
                    f'<div class="stat-value">{html.escape(str(record["Affected_Facilities"]).replace("_", " "))}'
                    f'</div>', unsafe_allow_html=True)

        color, bg = LEVEL_STYLE[severity]
        st.markdown(f'<div class="stat-label" style="margin-bottom:4px;">Monitoring Advisory</div>'
                    f'<div class="advisory" style="border-left-color:{color}; background:{bg};">'
                    f'{html.escape(record["Recommendation"])}</div>', unsafe_allow_html=True)
        st.caption("Advisory only: supports monitoring and escalation decisions; no control actions are issued.")


def color_severity(value):
    color, bg = LEVEL_STYLE.get(value, (TEXT, "#FFFFFF"))
    return f"color: {color}; background-color: {bg}; font-weight: 600;"


# =========================
# Sidebar: navigation only
# =========================

with st.sidebar:
    st.markdown(f"<div style='font-size:1.15rem; font-weight:700; color:{NAVY};'>Gas Supply Chain</div>"
                f"<div style='color:{MUTED}; font-size:0.85rem; margin-bottom:1.2rem;'>"
                f"Disruption Analysis & Decision Support</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"Offline analysis of the test period, {PERIOD} (synthetic dataset).")
    st.caption(f"Showing **confirmed disruptions only**: anomalies that persisted for 2 or more consecutive hours. "
               f"Single-hour transient spikes are filtered out.")


# =========================
# Page 1: Supply Chain Telemetry Overview
# =========================

def telemetry_figure(series):
    """Stacked panels sharing one time axis; red dots mark confirmed disruption hours."""
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[label for _, label in series])
    for i, (col, label) in enumerate(series, start=1):
        fig.add_trace(go.Scatter(x=df["Timestamp"], y=df[col], mode="lines", name=label, showlegend=False,
                                 line=dict(color=LINE, width=1),
                                 hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f}<extra></extra>"), row=i, col=1)
        fig.add_trace(go.Scatter(x=hours["Timestamp"], y=hours[col], mode="markers", name=CONFIRMED_LABEL,
                                 legendgroup="confirmed", showlegend=(i == 1),
                                 marker=dict(color=CONFIRMED_RED, size=5),
                                 customdata=hours["Disruption"].map(DISRUPTION_LABELS),
                                 hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f}<br>%{customdata}<extra></extra>"),
                      row=i, col=1)
    for annotation in fig.layout.annotations:  # left-align panel titles
        annotation.update(x=0, xanchor="left", font=dict(size=12, color=MUTED))
    return style_fig(fig, height=190 * len(series) + 50)


def page_overview():
    page_header(PAGES["Overview"], f"Historical telemetry and confirmed disruptions, {PERIOD}")

    high = events[events["Severity"] == "High"]
    cols = st.columns(4)
    cols[0].markdown(kpi_card("Hours analysed", f"{len(df):,}", PERIOD), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Confirmed disruption events", f"{len(events)}",
                              f"{len(hours)} hours in total"), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("High-severity events", f"{len(high)}",
                              "reached High system severity"), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Average event duration", f"{events['Hours'].mean():.1f} h",
                              "consecutive confirmed hours"), unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        section("Sensor telemetry")
        tabs = st.tabs(["Upstream", "Midstream", "Downstream"])
        panels = [
            [("Upstream_Pressure", "Upstream pressure (bar)"), ("Upstream_Flow", "Upstream flow (MMSCFD)")],
            [("Midstream_Pipeline_Flow", "Pipeline flow (MMSCFD)"),
             ("Midstream_Compressor_Load", "Compressor load (%)")],
            [(f"{p}_Plant_Output", f"{PLANT_LABELS[p]} plant output") for p in PLANTS],
        ]
        for tab, series in zip(tabs, panels):
            tab.plotly_chart(telemetry_figure(series), width="stretch", config=PLOT_CONFIG)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        section("Events by disruption type")
        counts = events["Disruption"].map(DISRUPTION_LABELS).value_counts().sort_values()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=NAVY,
                               text=counts.values, textposition="outside",
                               hovertemplate="%{y}: %{x} events<extra></extra>"))
        fig.update_xaxes(visible=False)
        st.plotly_chart(style_fig(fig, height=260), width="stretch", config=PLOT_CONFIG)
    with c2, st.container(border=True):
        section("Events by system severity")
        counts = events["Severity"].value_counts().reindex(LEVEL_ORDER[::-1]).dropna()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[LEVEL_STYLE[l][0] for l in counts.index],
                               text=counts.values.astype(int), textposition="outside",
                               hovertemplate="%{x}: %{y} events<extra></extra>"))
        fig.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig, height=260), width="stretch", config=PLOT_CONFIG)


# =========================
# Page 2: Event Diagnosis, Impact & SHAP Inspector
# =========================

def page_inspector():
    page_header(PAGES["Event Inspector"],
                "Pick a confirmed disruption to see what the models detected, why, and which facilities are affected")

    if events.empty:
        st.info("No confirmed disruptions in the test period.")
        return

    # Default: the Compressor Trip event with the largest supply drop, else the first event
    trips = events[events["Disruption"] == "Compressor_Trip"]
    default = trips["Peak_Supply_Drop"].idxmax() if len(trips) else events.index[0]
    options = list(events.index)
    c1, c2 = st.columns([2.2, 1])
    event_id = c1.selectbox("Confirmed disruption event", options, index=options.index(default),
                            format_func=lambda e: event_label(events.loc[e]))

    event_hours = hours[hours["Event"] == event_id]
    peak = event_hours[[f"{p}_Impact_Score" for p in PLANTS]].sum(axis=1).idxmax()  # highest-impact hour
    selected = c2.select_slider("Hour within event", list(event_hours.index), value=peak,
                                format_func=lambda i: f"{df.at[i, 'Timestamp']:%H:%M}")

    record = df.loc[selected]
    pred_idx = CLASSES.index(record["Predicted_Anomaly_Type"])
    shap_vector = models["shap_values"][selected, :, pred_idx]

    render_chain(record)
    left, right = st.columns([1.15, 1])
    with left:
        render_diagnosis(record, shap_vector)
    with right:
        render_impact(record)


# =========================
# Page 3: Disruption Log & Offline Scenario Tester
# =========================

SCENARIO_PRESETS = {
    "Typical normal hour": dict(pressure=78.5, flow=200.0, pipeline=200.0, load=80.0, lng=90.0, cogen=85.0,
                                pcg=82.0),
    "Upstream low pressure": dict(pressure=38.5, flow=100.0, pipeline=100.0, load=80.0, lng=90.0, cogen=85.0,
                                  pcg=82.0),
    "Midstream compressor trip": dict(pressure=78.5, flow=200.0, pipeline=1.5, load=1.0, lng=90.0, cogen=85.0,
                                      pcg=82.0),
    "LNG plant underperformance": dict(pressure=78.5, flow=200.0, pipeline=200.0, load=80.0, lng=20.0, cogen=85.0,
                                       pcg=82.0),
    "COGEN / Power plant underperformance": dict(pressure=78.5, flow=200.0, pipeline=200.0, load=80.0, lng=90.0,
                                                 cogen=20.0, pcg=82.0),
    "PCG plant underperformance": dict(pressure=78.5, flow=200.0, pipeline=200.0, load=80.0, lng=90.0, cogen=85.0,
                                       pcg=20.0),
}


def apply_preset():
    for key, value in SCENARIO_PRESETS[st.session_state["wi_preset"]].items():
        st.session_state[f"wi_{key}"] = value


def evaluate_scenario(inputs, persists):
    """Run one hypothetical hour through the scaler, Isolation Forest, XGBoost, impact rules and SHAP."""
    record = {
        "Upstream_Flow": inputs["flow"], "Upstream_Pressure": inputs["pressure"],
        "Midstream_Pipeline_Flow": inputs["pipeline"], "Midstream_Compressor_Load": inputs["load"],
    }
    for p, key in [("LNG", "lng"), ("COGEN", "cogen"), ("PCG", "pcg")]:
        supply = inputs["pipeline"] * dependency_weights[p]
        record[f"{p}_Gas_Supply_Received"] = supply
        record[f"{p}_Plant_Efficiency"] = inputs[key]
        record[f"{p}_Plant_Output"] = supply * inputs[key] / 100

    X = pd.DataFrame([record])[FEATURES]
    X_scaled = models["scaler"].transform(X[list(models["scaler"].feature_names_in_)])
    flagged = int(models["iso_forest"].predict(X_scaled)[0] == -1)
    score = float(-models["iso_forest"].decision_function(X_scaled)[0])

    proba = models["xgb"].predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    disruption = str(models["label_encoder"].classes_[pred_idx])

    row = pd.DataFrame([{**record, "Predicted_Anomaly_Type": disruption, "Predicted_Anomaly": flagged,
                         CONFIRMED_COL: int(flagged and persists), "Anomaly_Score": score,
                         "Prediction_Confidence": float(proba[pred_idx])}])
    result = assess_impact(row).iloc[0]

    shap_values = np.asarray(models["explainer"].shap_values(X))
    if shap_values.shape != (1, len(FEATURES), len(CLASSES)):  # older SHAP: (classes, rows, features)
        shap_values = np.transpose(shap_values, (1, 2, 0))
    return result, shap_values[0, :, pred_idx]


def page_log_and_tester():
    page_header(PAGES["Disruption Log & What-If"],
                "Every confirmed disruption in the test period, plus a what-if tester for hypothetical readings")

    tab_log, tab_whatif = st.tabs(["Disruption Log", "What-If Scenario Tester"])

    with tab_log:
        summary = pd.DataFrame({
            "Start": events["Start"].dt.strftime("%Y-%m-%d %H:%M"),
            "End": (events["End"] + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d %H:%M"),
            "Duration (h)": events["Hours"],
            "Disruption": events["Disruption"].map(DISRUPTION_LABELS),
            "Peak Severity": events["Severity"],
            "Affected Facilities": events["Affected"].str.replace("_", " "),
            "Peak Supply Drop (%)": events["Peak_Supply_Drop"].round(1),
            "Peak Confidence": events["Peak_Confidence"],
        })
        st.markdown(f"**{len(summary)} confirmed disruption events** · {len(hours)} hours")
        st.dataframe(summary.style.map(color_severity, subset=["Peak Severity"])
                     .format({"Peak Confidence": "{:.1%}", "Peak Supply Drop (%)": "{:.1f}"}),
                     hide_index=True, width="stretch")

        hourly = pd.DataFrame({
            "Timestamp": hours["Timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
            "Event": hours["Event"],
            "Predicted Disruption": hours["Disruption"].map(DISRUPTION_LABELS),
            "Anomaly Score": hours["Anomaly_Score"],
            "Confidence": hours["Prediction_Confidence"],
            "System Severity": hours["System_Severity"],
            "Affected Facilities": hours["Affected_Facilities"].str.replace("_", " "),
            "Monitoring Advisory": hours["Recommendation"],
            "Top SHAP Sensors": hours["SHAP_Top_Contributors"],
            "Recorded Status": hours["System_Status"],
        })
        with st.expander("Hourly detail"):
            st.dataframe(hourly.style.map(color_severity, subset=["System Severity"])
                         .format({"Anomaly Score": "{:+.3f}", "Confidence": "{:.1%}"}),
                         hide_index=True, width="stretch")

        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("Download events (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           file_name="confirmed_disruption_events.csv", mime="text/csv", width="stretch")
        c2.download_button("Download hourly log (CSV)", hourly.to_csv(index=False).encode("utf-8"),
                           file_name="confirmed_disruption_hourly_log.csv", mime="text/csv", width="stretch")

    with tab_whatif:
        st.caption("Evaluate a hypothetical hour through the trained Isolation Forest, XGBoost classifier, "
                   "impact rules and SHAP explainer. Gas supply = pipeline flow × allocation share; "
                   "output = supply × efficiency. Readings far outside the training data may give unreliable results.")

        if "wi_preset" not in st.session_state:
            st.session_state["wi_preset"] = "Midstream compressor trip"
            apply_preset()

        with st.container(border=True):
            st.selectbox("Start from preset", list(SCENARIO_PRESETS), key="wi_preset", on_change=apply_preset)
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                section("Upstream")
                st.slider("Pressure (bar)", 0.0, 100.0, step=0.5, key="wi_pressure")
                st.slider("Flow (MMSCFD)", 0.0, 250.0, step=1.0, key="wi_flow")
            with c2:
                section("Midstream")
                st.slider("Pipeline flow (MMSCFD)", 0.0, 250.0, step=1.0, key="wi_pipeline")
                st.slider("Compressor load (%)", 0.0, 100.0, step=0.5, key="wi_load")
            with c3:
                section("Downstream efficiency")
                st.slider("LNG (%)", 0.0, 100.0, step=0.5, key="wi_lng")
                st.slider("COGEN / Power (%)", 0.0, 100.0, step=0.5, key="wi_cogen")
                st.slider("PCG (%)", 0.0, 100.0, step=0.5, key="wi_pcg")
            persists = st.checkbox("Assume this condition persists for ≥2 consecutive hours", value=True,
                                   help="A single hour cannot show persistence; this sets whether an Isolation "
                                        "Forest flag counts as a Confirmed Disruption (≥2h) or a filtered "
                                        "transient spike.")

        inputs = {key: st.session_state[f"wi_{key}"] for key in SCENARIO_PRESETS["Typical normal hour"]}
        result, shap_vector = evaluate_scenario(inputs, persists)

        st.write("")
        section("Scenario result")
        render_chain(result)
        left, right = st.columns([1.15, 1])
        with left:
            render_diagnosis(result, shap_vector)
        with right:
            render_impact(result)


# =========================
# Page 4: Model Evaluation & Temporal Filtering Benchmarks
# =========================

def page_benchmarks():
    page_header(PAGES["Model Benchmarks"], "Test-period performance of the detection, classification and impact modules")

    anomaly = load_result_csv("anomaly_detection_comparison.csv").copy()
    anomaly["Model"] = anomaly["Model"].str.replace("+ 2h rule", "+ 2h persistence filter", regex=False)
    anomaly = anomaly.rename(columns={"False_Alarm_Hours_Per_Day": "False_Positive_Hours_Per_Day"})
    anomaly["Base_Model"] = anomaly["Model"].str.replace(" + 2h persistence filter", "", regex=False)
    anomaly["Variant"] = np.where(anomaly["Model"].str.contains("persistence"), "+ 2h persistence filter", "Raw")

    iso = anomaly[anomaly["Base_Model"] == "Isolation Forest"].set_index("Variant")
    if {"Raw", "+ 2h persistence filter"} <= set(iso.index):
        raw, filt = iso.loc["Raw"], iso.loc["+ 2h persistence filter"]
        cols = st.columns(4)
        cols[0].markdown(kpi_card("False positives (Isolation Forest)",
                                  f"{int(raw['FP'])} &#8594; {int(filt['FP'])}", "raw vs 2h persistence filter"),
                         unsafe_allow_html=True)
        cols[1].markdown(kpi_card("Anomaly F1-score", f"{raw['Anomaly_F1']:.2f} &#8594; {filt['Anomaly_F1']:.2f}",
                                  "raw vs 2h persistence filter"), unsafe_allow_html=True)
        cols[2].markdown(kpi_card("Events detected", f"{int(filt['Events_Detected'])}/{int(filt['Total_Events'])}",
                                  "with the 2h persistence filter"), unsafe_allow_html=True)
        cols[3].markdown(kpi_card("ROC-AUC", f"{raw['ROC_AUC']:.3f}", "Isolation Forest anomaly score"),
                         unsafe_allow_html=True)
        st.write("")

    with st.container(border=True):
        section("Anomaly detection: raw vs 2h persistence filter")
        c1, c2 = st.columns(2)
        for container, metric, title in [(c1, "FP", "False positives (hours)"), (c2, "Anomaly_F1", "Anomaly F1-score")]:
            fig = go.Figure()
            for variant, color in [("Raw", "#CBD5E1"), ("+ 2h persistence filter", NAVY)]:
                part = anomaly[anomaly["Variant"] == variant]
                fig.add_trace(go.Bar(x=part["Base_Model"], y=part[metric], name=variant, marker_color=color,
                                     text=part[metric].round(2), textposition="outside"))
            fig.update_layout(title=dict(text=title, font=dict(size=13)), barmode="group")
            container.plotly_chart(style_fig(fig, height=320), width="stretch", config=PLOT_CONFIG)

        with st.expander("Full anomaly detection table"):
            st.dataframe(
                anomaly.drop(columns=["Base_Model", "Variant"]).style.format({
                    "Accuracy": "{:.3f}", "Anomaly_Precision": "{:.3f}", "Anomaly_Recall": "{:.3f}",
                    "Anomaly_F1": "{:.3f}", "ROC_AUC": "{:.3f}", "False_Positive_Hours_Per_Day": "{:.2f}",
                    "Inference_Time_ms": "{:.1f}"}),
                hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        section("Disruption classification (6 classes)")
        classification = load_result_csv("classification_comparison.csv")
        st.dataframe(classification.style.format({c: "{:.4f}" for c in classification.columns
                                                  if c not in ("Model", "Inference_Time_ms")}
                                                 | {"Inference_Time_ms": "{:.1f}"}),
                     hide_index=True, width="stretch")
    with c2, st.container(border=True):
        section("Impact threshold sensitivity (±15%)")
        st.dataframe(load_result_csv("impact_threshold_sensitivity.csv"), hide_index=True, width="stretch")
        st.caption("System_Severity_Changed counts anomalous records whose overall severity changes "
                   "when all cut-offs move by ±15%.")

    with st.container(border=True):
        section("Global SHAP feature importance (XGBoost)")
        importance = load_result_csv("shap_global_importance.csv")
        class_choice = st.selectbox("Class", ["Mean_Abs_SHAP_All_Classes"] + CLASSES,
                                    format_func=lambda c: "All classes (mean)" if c.startswith("Mean_")
                                    else DISRUPTION_LABELS.get(c, c))
        ranked = importance.sort_values(class_choice)
        fig = go.Figure(go.Bar(x=ranked[class_choice], y=ranked["Feature"].str.replace("_", " "), orientation="h",
                               marker_color=LINE, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(xaxis_title="Mean |SHAP value|")
        st.plotly_chart(style_fig(fig, height=420), width="stretch", config=PLOT_CONFIG)


{
    "Overview": page_overview,
    "Event Inspector": page_inspector,
    "Disruption Log & What-If": page_log_and_tester,
    "Model Benchmarks": page_benchmarks,
}[page]()
