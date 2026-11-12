"""Historical Gas Supply Chain Disruption Analysis & Decision Support Dashboard.

Offline analytical tool over the model test period (no live data, no control actions).
Shows confirmed disruptions only: Isolation Forest anomalies that persisted for 2+ consecutive hours.
Run from the project root:  streamlit run app.py
Requires the result files produced by Models/Train_Anomaly.py -> Train_Classifier.py -> ImpactAssessment.py.
"""

import base64
import html
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
RESULT_DIR = MODELS_DIR / "result"
PICTURE_DIR = BASE_DIR / "Picture"
sys.path.insert(0, str(MODELS_DIR))

from impact_rules import CONFIRMED_COL, PLANTS, LEVEL_ORDER, dependency_weights  # noqa: E402
from scoring import FEATURES as MODEL_FEATURES, validate, score, shap_matrix  # noqa: E402

# Browser-tab icon: the AEGIS shield mark
st.set_page_config(page_title="AEGIS - Gas Supply Chain Monitoring", page_icon=str(PICTURE_DIR / "aegis-mark.png"),
                   layout="wide")


# =========================
# Theme constants & labels
# =========================

NAVY = "#2B4A6F"
TEXT = "#1E293B"
MUTED = "#64748B"
BORDER = "#E3E8EF"
FONT = "Inter, 'Source Sans Pro', sans-serif"
PRIMARY_LINE = "#5B8DEF"
CONFIRMED_RED = "#E05A5A"
POSITIVE_SHAP = "#E05A5A"
NEGATIVE_SHAP = "#8FB3F0"

# (text/border colour, soft background)
LEVEL_STYLE = {
    "Normal": ("#3FA27F", "#EEF8F3"),
    "No Impact": ("#3FA27F", "#EEF8F3"),
    "Low": ("#C99A2E", "#FCF6E8"),
    "Medium": ("#D9773A", "#FDF1E8"),
    "High": ("#D04848", "#FCEEEE"),
    "Confirmed": ("#D04848", "#FCEEEE"),
}

CONFIRMED_LABEL = "Confirmed Disruption (≥2h)"

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

# Confusion-matrix axis order and short labels
CLASS_ORDER = ["Normal", "Low_Pressure", "Compressor_Trip", "LNG_Underperformance", "COGEN_Underperformance",
               "PCG_Underperformance"]
CLASS_SHORT = {"Normal": "Normal", "Low_Pressure": "Low Pressure", "Compressor_Trip": "Compressor Trip",
               "LNG_Underperformance": "LNG Under.", "COGEN_Underperformance": "COGEN Under.",
               "PCG_Underperformance": "PCG Under."}

FEATURE_UNITS = {
    "Upstream_Pressure": "bar", "Midstream_Compressor_Load": "%",
    "LNG_Plant_Efficiency": "%", "COGEN_Plant_Efficiency": "%", "PCG_Plant_Efficiency": "%",
}

# Page name -> sidebar icon
PAGES = {
    "Supply Chain Overview": ":material/dashboard:",
    "Event Inspector": ":material/search:",
    "Disruption Logs": ":material/list_alt:",
    "Upload & Score": ":material/upload_file:",
    "Model Performance": ":material/insights:",
}

st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem; padding-bottom: 2.5rem;}}
h1, h2, h3, h4 {{color: {NAVY}; font-weight: 600;}}
.page-sub {{color: {MUTED}; margin-top: -0.5rem; margin-bottom: 1.4rem;}}
.kpi {{background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 20px; height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);}}
.kpi-label {{font-size: 0.8rem; color: {MUTED}; font-weight: 500;}}
.kpi-value {{font-size: 1.85rem; font-weight: 600; color: {TEXT}; line-height: 1.35;}}
.kpi-sub {{font-size: 0.8rem; color: {MUTED};}}
.pill {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        white-space: nowrap;}}
.chain {{display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; margin-bottom: 1rem;}}
.node {{flex: 1 1 150px; background: #FFFFFF; border: 1.5px solid; border-radius: 12px; padding: 12px 14px;}}
.node-title {{font-weight: 600; color: {NAVY}; font-size: 0.95rem;}}
.node-status {{font-size: 0.8rem; font-weight: 600; margin: 2px 0 6px 0;}}
.node-reading {{font-size: 0.8rem; color: {TEXT}; line-height: 1.6;}}
.node-recorded {{font-size: 0.72rem; color: {MUTED}; margin-top: 6px;}}
.arrow {{align-self: center; color: #CBD5E1; font-size: 1.4rem;}}
.stat-label {{font-size: 0.76rem; color: {MUTED}; font-weight: 500;}}
.stat-value {{font-size: 1.02rem; font-weight: 600; color: {TEXT}; margin-bottom: 10px;}}
.summary {{background: #F2F6FC; border: 1px solid #DCE6F5; border-radius: 12px; padding: 12px 14px;
           color: {TEXT}; font-size: 0.9rem; line-height: 1.55;}}
.impact-table {{width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 12px;}}
.impact-table th {{text-align: left; color: {MUTED}; font-weight: 500; font-size: 0.76rem;
                   border-bottom: 1px solid {BORDER}; padding: 6px 4px;}}
.impact-table td {{border-bottom: 1px solid #F1F5F9; padding: 8px 4px;}}
.advisory {{border-radius: 12px; padding: 12px 14px; border-left: 4px solid; font-size: 0.9rem; line-height: 1.55;}}
/* Sidebar navigation buttons: left-aligned, quiet until hovered or selected */
[data-testid="stSidebar"] .stButton button {{justify-content: flex-start; padding: 0.55rem 0.9rem;
    border-radius: 10px; font-weight: 500;}}
[data-testid="stSidebar"] .stButton button > div {{justify-content: flex-start; width: 100%;}}
[data-testid="stSidebar"] .stButton button p {{text-align: left;}}
/* Hide the Deploy button and the three-dot main menu (top right) */
[data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stMainMenu"] {{display: none !important;}}
[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"] {{background: transparent;
    border-color: transparent; color: {TEXT};}}
[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"]:hover {{background: #EEF3F9;
    color: {NAVY};}}
</style>
""", unsafe_allow_html=True)


# =========================
# Data loading
# =========================

REQUIRED_FILES = [
    "dashboard_master_test.csv", "shap_artifacts.pkl", "anomaly_detection_comparison.csv",
    "classification_comparison.csv", "impact_threshold_sensitivity.csv", "shap_global_importance.csv",
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
def confirmed_hours(df):
    """Confirmed (>=2h) hours only, grouped into events: each unbroken run of confirmed hours is one event."""
    hours = df[df[CONFIRMED_COL] == 1].copy()
    hours["Disruption"] = hours["Predicted_Anomaly_Type"].where(hours["Predicted_Anomaly_Type"] != "Normal",
                                                                "Unclassified")
    hours["Event"] = (hours["Timestamp"].diff() != pd.Timedelta(hours=1)).cumsum()

    def main_type(s):  # ramp hours can be classified Normal; name the event by its classified hours
        typed = s[s != "Unclassified"]
        return typed.mode().iloc[0] if len(typed) else "Unclassified"

    hours["Event_Type"] = hours.groupby("Event")["Disruption"].transform(main_type)
    return hours


@st.cache_data
def load_result_csv(name):
    return pd.read_csv(RESULT_DIR / name)


@st.cache_resource
def load_shap():
    return joblib.load(RESULT_DIR / "shap_artifacts.pkl")  # shap_values: (rows, features, classes)


df = load_dashboard()
hours = confirmed_hours(df)
shap_artifacts = load_shap()
FEATURES = shap_artifacts["feature_names"]
CLASSES = shap_artifacts["class_names"]

if len(df) != shap_artifacts["shap_values"].shape[0]:
    st.error("dashboard_master_test.csv and shap_artifacts.pkl are out of sync. Re-run Models/ImpactAssessment.py.")
    st.stop()

PERIOD = f"{df['Timestamp'].min():%d %b %Y} to {df['Timestamp'].max():%d %b %Y}"


# =========================
# Uploaded dataset (Upload & Score page)
# =========================

MODEL_FILES = ["scaler.pkl", "isolation_forest_model.pkl", "xgboost_anomaly_classifier.pkl",
               "xgboost_label_encoder.pkl"]


@st.cache_resource
def load_models():
    """Trained models, loaded only when a dataset is uploaded."""
    scaler, iso_forest, xgb_model, label_encoder = (joblib.load(RESULT_DIR / f) for f in MODEL_FILES)
    return dict(scaler=scaler, iso_forest=iso_forest, xgb_model=xgb_model, label_encoder=label_encoder,
                explainer=shap.TreeExplainer(xgb_model))


@st.cache_data(show_spinner=False)
def score_bytes(data):
    """Read, validate and score uploaded CSV bytes. Returns (scored DataFrame or None, errors, warnings)."""
    try:
        raw = pd.read_csv(io.BytesIO(data))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        return None, [f"The file could not be read as a CSV: {exc}"], []
    clean, errors, warnings = validate(raw)
    if errors:
        return None, errors, warnings
    return score(clean, **load_models()), [], warnings


def on_upload():
    """File-uploader callback: score the new file and, if it is valid, show it in Overview, Event Inspector and Disruption Logs.
    The result lives in session_state, so it survives page changes (the uploader widget itself does not)."""
    file = st.session_state.get("upload_csv")
    if file is None:  # file removed from the uploader: keep the dataset already loaded
        return
    scored, errors, warnings = score_bytes(file.getvalue())
    st.session_state["upload_errors"] = (file.name, errors)
    if errors:
        return
    start, end = scored["Timestamp"].min(), scored["Timestamp"].max()
    st.session_state["upload"] = dict(name=file.name, df=scored, hours=confirmed_hours(scored), warnings=warnings,
                                      period=f"{start:%d %b %Y} to {end:%d %b %Y}")
    st.session_state["data_source"] = "uploaded"


def clear_upload():
    for key in ("upload", "upload_errors"):
        st.session_state.pop(key, None)
    st.session_state["data_source"] = "original"


# Dataset shown in Overview, Event Inspector and Disruption Logs: original test period, or the uploaded file
upload = st.session_state.get("upload")
USE_UPLOAD = upload is not None and st.session_state.get("data_source") == "uploaded"
if USE_UPLOAD:
    view_df, view_hours, view_period = upload["df"], upload["hours"], upload["period"]
else:
    view_df, view_hours, view_period = df, hours, PERIOD


# =========================
# HTML / chart helpers
# =========================

def pill(text, level):
    color, bg = LEVEL_STYLE[level]
    return f'<span class="pill" style="color:{color}; background:{bg};">{html.escape(text)}</span>'


def kpi_card(label, value, sub, accent):
    return (f'<div class="kpi" style="border-left: 4px solid {accent};">'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def page_header(title, subtitle):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def fmt_feature(feature, value):
    unit = FEATURE_UNITS.get(feature, "MMSCFD")
    return f"{feature.replace('_', ' ')} = {value:.1f} {unit}"


def style_fig(fig, height=330):
    fig.update_layout(
        height=height, template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, color=TEXT, size=12), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_family=FONT, bgcolor="#FFFFFF", bordercolor=BORDER),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor="#EEF2F6", linecolor=BORDER)
    return fig


def color_severity(value):
    color, bg = LEVEL_STYLE.get(value, (TEXT, "#FFFFFF"))
    return f"color: {color}; background-color: {bg}; font-weight: 600;"


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
        rec = f'<div class="node-recorded">Recorded status: {html.escape(str(recorded))}</div>' if recorded else ""
        return (f'<div class="node" style="border-color:{color};"><div class="node-title">{title}</div>'
                f'<div class="node-status" style="color:{color};">{html.escape(text)}</div>'
                f'<div class="node-reading">{"<br>".join(readings)}</div>{rec}</div>')

    nodes = [
        node("Upstream", upstream,
             [f"Pressure: {record['Upstream_Pressure']:.1f} bar", f"Flow: {record['Upstream_Flow']:.1f} MMSCFD"],
             record.get("Upstream_Status")),
        node("Midstream", midstream,
             [f"Pipeline flow: {record['Midstream_Pipeline_Flow']:.1f} MMSCFD",
              f"Compressor load: {record['Midstream_Compressor_Load']:.1f}%"],
             record.get("Midstream_Status")),
    ]
    for p in PLANTS:
        level = record[f"{p}_Impact_Level"]
        status = ("Normal", "No Impact") if level == "No Impact" else (f"Impact: {level}", level)
        nodes.append(node(
            f"{PLANT_LABELS[p]} ({dependency_weights[p]:.0%})", status,
            [f"Supply: {record[f'{p}_Gas_Supply_Received']:.1f} MMSCFD",
             f"Efficiency: {record[f'{p}_Plant_Efficiency']:.1f}%",
             f"Output: {record[f'{p}_Plant_Output']:.1f}"],
            record.get(f"{p}_Status")))

    arrow = '<div class="arrow">&#8594;</div>'
    chain = nodes[0] + arrow + nodes[1] + arrow + "".join(nodes[2:])
    st.markdown(f'<div class="chain">{chain}</div>', unsafe_allow_html=True)


def render_diagnosis(record, shap_vector):
    """Left card: Isolation Forest + persistence, XGBoost type/confidence, SHAP bar chart and plain-English summary."""
    disruption = record["Predicted_Anomaly_Type"]

    with st.container(border=True):
        st.markdown("#### AI Diagnosis & SHAP Explanation")
        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="stat-label">Isolation Forest anomaly score</div>'
            f'<div class="stat-value">{record["Anomaly_Score"]:+.3f} '
            f'<span style="color:{MUTED}; font-size:0.8rem; font-weight:400;">(&gt; 0 = anomalous)</span></div>'
            f'<div class="stat-label">Temporal persistence</div>'
            f'<div class="stat-value">{pill(CONFIRMED_LABEL, "Confirmed")}</div>', unsafe_allow_html=True)
        type_level = "Normal" if disruption == "Normal" else "High"
        c2.markdown(
            f'<div class="stat-label">XGBoost predicted disruption</div>'
            f'<div class="stat-value">{pill(DISRUPTION_LABELS[disruption], type_level)}</div>'
            f'<div class="stat-label">Prediction confidence</div>'
            f'<div class="stat-value">{record["Prediction_Confidence"]:.1%}</div>', unsafe_allow_html=True)

        order = np.argsort(-np.abs(shap_vector))[:8][::-1]  # top 8 by magnitude, largest at the top of the chart
        names = [FEATURES[j].replace("_", " ") for j in order]
        values = shap_vector[order]
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h",
            marker_color=[POSITIVE_SHAP if v > 0 else NEGATIVE_SHAP for v in values],
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.update_layout(title=f"SHAP contributions toward '{DISRUPTION_LABELS[disruption]}'",
                          xaxis_title="SHAP value (red = supports prediction, blue = argues against)")
        st.plotly_chart(style_fig(fig, height=320), width="stretch")

        drivers = [j for j in np.argsort(-shap_vector)[:3] if shap_vector[j] >= 0.005]
        evidence = "; ".join(f"{fmt_feature(FEATURES[j], record[FEATURES[j]])} (SHAP {shap_vector[j]:+.2f})"
                             for j in drivers) or "no single sensor dominated"
        summary = (f"XGBoost classified this hour as <b>{DISRUPTION_LABELS[disruption]}</b> with "
                   f"{record['Prediction_Confidence']:.1%} confidence. Strongest supporting evidence: {evidence}. "
                   f"Isolation Forest scored the hour {record['Anomaly_Score']:+.3f} and the anomaly persisted "
                   f"for at least 2 consecutive hours.")
        st.markdown(f'<div class="summary">{summary}</div>', unsafe_allow_html=True)


def render_impact(record):
    """Right card: per-facility impact table, system severity and monitoring advisory."""
    severity = record["System_Severity"]
    with st.container(border=True):
        st.markdown("#### Downstream Impact Assessment")
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
        st.markdown(f'<div class="stat-label">Monitoring Advisory</div>'
                    f'<div class="advisory" style="border-left-color:{color}; background:{bg};">'
                    f'{html.escape(record["Recommendation"])}</div>', unsafe_allow_html=True)
        st.caption("Advisory only: supports monitoring and escalation decisions; no control actions are issued.")


# =========================
# Sidebar: page navigation
# =========================

if "page" not in st.session_state:
    st.session_state["page"] = next(iter(PAGES))


def go_to(name):
    st.session_state["page"] = name


with st.sidebar:
    logo = base64.b64encode((PICTURE_DIR / "aegis-logo-sidebar.svg").read_bytes()).decode()
    st.markdown(f'<img src="data:image/svg+xml;base64,{logo}" alt="AEGIS: AI-Enhanced Gas Supply Chain Impact '
                f'& Disruption Assessment System" style="width:100%; margin:0 0 1.2rem 0;">',
                unsafe_allow_html=True)
    for name, icon in PAGES.items():
        st.button(name, icon=icon, key=f"nav_{name}", width="stretch", on_click=go_to, args=(name,),
                  type="primary" if st.session_state["page"] == name else "secondary")
    st.divider()
    if upload is not None:
        st.selectbox("Data shown in Overview, Inspector & Logs", ["uploaded", "original"], key="data_source",
                     format_func=lambda s: f"Uploaded: {upload['name']}" if s == "uploaded"
                     else "Original test period", filter_mode=None)
    if USE_UPLOAD:
        st.caption(f"Overview, Event Inspector and Disruption Logs show **{upload['name']}** ({view_period}). "
                   f"Model Performance shows the original test period, {PERIOD}.")
    else:
        st.caption(f"Offline analysis of the model test period: {PERIOD} (synthetic dataset).")
    st.caption("Showing confirmed disruptions only: anomalies that persisted for 2 or more consecutive hours.")


def upload_banner():
    if USE_UPLOAD:
        st.info(f"Showing your uploaded dataset **{upload['name']}**. Switch back to the original test period "
                "in the sidebar.", icon=":material/upload_file:")


# =========================
# Page 1: Supply Chain Overview
# =========================

def page_overview():
    page_header("Supply Chain Overview", f"Historical telemetry and confirmed disruptions, {view_period}")
    upload_banner()

    event_severity = view_hours.groupby("Event")["System_Severity"].agg(lambda s: max(s, key=LEVEL_ORDER.index))
    n_events = view_hours["Event"].nunique()

    cols = st.columns(4)
    cols[0].markdown(kpi_card("Total Analyzed Hours", f"{len(view_df):,}", view_period, NAVY), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Confirmed Disruption Hours", f"{len(view_hours):,}",
                              "persisted for 2+ consecutive hours", CONFIRMED_RED), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Confirmed Disruption Events", f"{n_events:,}",
                              f"{len(view_hours) / max(n_events, 1):.1f} hours per event on average", "#D9773A"),
                     unsafe_allow_html=True)
    cols[3].markdown(kpi_card("High-Severity Events", f"{(event_severity == 'High').sum():,}",
                              "events reaching High system severity", "#D04848"), unsafe_allow_html=True)
    st.write("")

    def telemetry_chart(lines, knock_on=()):
        """Red dots: confirmed disruptions of this stage/plant itself.
        Orange rings: knock-on effect here of a confirmed disruption further up the chain (`knock_on` types)."""
        fig = go.Figure()
        colors = [PRIMARY_LINE, "#3FB6A8", "#9B87E8"]
        for i, ((col, label, fault), color) in enumerate(zip(lines, colors)):
            fig.add_trace(go.Scatter(x=view_df["Timestamp"], y=view_df[col], name=label, mode="lines",
                                     line=dict(color=color, width=1.2), legendrank=1 + i,  # lines first
                                     hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.1f}<extra>" + label + "</extra>"))
            rows = view_hours[view_hours["Event_Type"] == fault]
            fig.add_trace(go.Scatter(x=rows["Timestamp"], y=rows[col], mode="markers", name=CONFIRMED_LABEL,
                                     marker=dict(color=CONFIRMED_RED, size=7), legendgroup="confirmed",
                                     showlegend=(i == 0), legendrank=10,
                                     hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.1f}<extra>"
                                                   + DISRUPTION_LABELS[fault] + "</extra>"))
            if knock_on:
                rows = view_hours[view_hours["Event_Type"].isin(knock_on)]
                fig.add_trace(go.Scatter(
                    x=rows["Timestamp"], y=rows[col], mode="markers", name="Knock-on from upstream disruption",
                    marker=dict(color="rgba(0,0,0,0)", size=9, line=dict(color="#E8894A", width=2)),
                    legendgroup="knock_on", showlegend=(i == 0), legendrank=11,
                    customdata=rows["Event_Type"].map(DISRUPTION_LABELS),
                    hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.1f}<extra>Knock-on from %{customdata}</extra>"))
        fig.update_yaxes(title_text="Flow (MMSCFD)" if len(lines) == 1 else "Plant output")
        st.plotly_chart(style_fig(fig, height=400), width="stretch")

    with st.container(border=True):
        st.markdown("#### Supply Chain Telemetry")
        up, mid, down = st.tabs(["Upstream", "Midstream", "Downstream"])
        with up:
            telemetry_chart([("Upstream_Flow", "Upstream Flow (MMSCFD)", "Low_Pressure")])
        with mid:
            telemetry_chart([("Midstream_Pipeline_Flow", "Pipeline Flow (MMSCFD)", "Compressor_Trip")],
                            knock_on=("Low_Pressure",))
        with down:
            telemetry_chart([(f"{p}_Plant_Output", f"{PLANT_LABELS[p]} Output", f"{p}_Underperformance")
                             for p in PLANTS], knock_on=("Low_Pressure", "Compressor_Trip"))


# =========================
# Page 2: Event Inspector
# =========================

def page_inspector():
    page_header("Event Inspector",
                "Select a confirmed disruption hour to see what the models detected, why, and which facilities "
                "are affected")
    upload_banner()

    if view_hours.empty:
        st.info("No confirmed disruptions in this dataset.")
        return

    labels = {idx: f"{row.Timestamp:%Y-%m-%d %H:%M}  |  {DISRUPTION_LABELS[row.Disruption]}  |  "
                   f"Severity: {row.System_Severity}"
              for idx, row in view_hours.iterrows()}
    # Default: the highest-impact Compressor Trip hour, else Low Pressure, else the first listed hour
    default = view_hours.index[0]
    for fault in ["Compressor_Trip", "Low_Pressure"]:
        matches = view_hours[view_hours["Predicted_Anomaly_Type"] == fault]
        if not matches.empty:
            default = matches["Supply_Drop_Percent"].idxmax()
            break
    options = list(labels)
    selected = st.selectbox("Confirmed disruption hour", options, index=options.index(default),
                            format_func=labels.get, filter_mode=None)  # select only, no typing

    record = view_df.loc[selected]
    pred_idx = CLASSES.index(record["Predicted_Anomaly_Type"])
    if USE_UPLOAD:  # uploaded data: explain this one hour on demand
        shap_vector = shap_matrix(load_models()["explainer"], view_df.loc[[selected], FEATURES],
                                  len(CLASSES))[0, :, pred_idx]
    else:
        shap_vector = shap_artifacts["shap_values"][selected, :, pred_idx]

    render_chain(record)
    left, right = st.columns([1.15, 1])
    with left:
        render_diagnosis(record, shap_vector)
    with right:
        render_impact(record)


# =========================
# Page 3: Upload & Score
# =========================

def event_summary(event_hours):
    """One row per confirmed disruption event (from confirmed_hours())."""
    summary = event_hours.groupby("Event").agg(
        Start=("Timestamp", "min"), End=("Timestamp", "max"), Hours=("Timestamp", "size"),
        Disruption=("Event_Type", "first"),
        Max_Confidence=("Prediction_Confidence", "max"),
        System_Severity=("System_Severity", lambda s: max(s, key=LEVEL_ORDER.index)),
        Affected_Facilities=("Affected_Facilities", lambda s: s.mode().iloc[0]),
    ).reset_index(drop=True)
    summary["Disruption"] = summary["Disruption"].map(DISRUPTION_LABELS)
    summary["Affected_Facilities"] = summary["Affected_Facilities"].str.replace("_", " ")
    return summary


# Required columns: (unit, description); "typical" values come from the normal hours of the original test period
FORMAT_SPEC = {
    "Timestamp": ("date-time", "Hour of the reading, e.g. 2025-01-01 00:00:00"),
    "Upstream_Flow": ("MMSCFD", "Gas flow leaving the upstream field"),
    "Upstream_Pressure": ("bar", "Upstream gas pressure"),
    "Midstream_Pipeline_Flow": ("MMSCFD", "Gas flow in the transmission pipeline"),
    "Midstream_Compressor_Load": ("%", "Compressor station load"),
    **{f"{p}_Gas_Supply_Received": ("MMSCFD", f"Gas delivered to the {PLANT_LABELS[p]} plant") for p in PLANTS},
    **{f"{p}_Plant_Efficiency": ("%", f"{PLANT_LABELS[p]} plant efficiency") for p in PLANTS},
    **{f"{p}_Plant_Output": ("supply × efficiency", f"{PLANT_LABELS[p]} plant output") for p in PLANTS},
}
FORMAT_ORDER = ["Timestamp"] + MODEL_FEATURES


def page_upload():
    page_header("Upload & Score", "Run a new hourly dataset through the trained models, then explore it in "
                                  "Supply Chain Overview, Event Inspector and Disruption Logs")

    missing_models = [f for f in MODEL_FILES if not (RESULT_DIR / f).exists()]
    if missing_models:
        st.error(f"Missing trained model files in {RESULT_DIR}: {', '.join(missing_models)}. "
                 "Run Models/Train_Anomaly.py and Models/Train_Classifier.py first.")
        return

    with st.expander("**1. File format**: required columns and rules", icon=":material/description:",
                     expanded=False):
        st.markdown("Upload a **CSV file with one row per hour** and the **14 columns below**, using exactly "
                    "these column names. This is the same layout as the project's master dataset "
                    "(`master_gas_supply_chain_dataset.csv`), so that file can be uploaded as it is.")
        normal = df[df["System_Status"] == "Normal"]
        spec = pd.DataFrame([{
            "Column": col, "Unit": FORMAT_SPEC[col][0], "Description": FORMAT_SPEC[col][1],
            "Typical normal value": "—" if col == "Timestamp" else f"{normal[col].median():.1f}",
        } for col in FORMAT_ORDER])
        st.dataframe(spec, hide_index=True, width="stretch", height=(len(spec) + 1) * 35 + 3)  # all rows
        st.markdown(
            "- **Order of rows and columns does not matter.** Rows are sorted by time; extra columns are ignored.\n"
            "- **Each hour must appear once.** Duplicate or sub-hourly timestamps are rejected. Missing hours "
            "are allowed, but disruptions are never joined across a gap.\n"
            "- **Label columns are optional.** If the file also has `System_Status` (the true condition of each "
            "hour), the page also checks the models' predictions against it.\n"
            "- **Use data with similar normal levels.** The models learned normal operation from this "
            "project's synthetic dataset (see *Typical normal value*). Data from a plant that normally runs at "
            "different levels will give unreliable results.")
        template = df[FORMAT_ORDER].head(24).copy()
        template["Timestamp"] = template["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Download example file (24 hours, CSV)", template.to_csv(index=False).encode("utf-8"),
                           file_name="aegis_upload_example.csv", mime="text/csv", icon=":material/download:")

    with st.container(border=True):
        st.markdown("#### 2. Upload your dataset")
        st.file_uploader("Upload dataset (CSV)", type=["csv"], key="upload_csv", on_change=on_upload,
                         help="The file is scored as soon as it is uploaded (a full year takes a few seconds).")
        failed = st.session_state.get("upload_errors")
        if failed and failed[1]:
            st.error(f"**{failed[0]} cannot be scored:**\n\n" + "\n".join(f"- {e}" for e in failed[1]))
        if upload is None:
            st.caption("After a successful upload, the results appear below and in Supply Chain Overview, "
                       "Event Inspector and Disruption Logs. Your file is only used in this browser session.")
            return
        c1, c2 = st.columns([3, 1])
        c1.success(f"Loaded **{upload['name']}**: {len(upload['df']):,} hours, {upload['period']}.")
        c2.button("Remove uploaded dataset", icon=":material/delete:", on_click=clear_upload, width="stretch")
        st.caption("Explore it in **Supply Chain Overview**, **Event Inspector** and **Disruption Logs**. Use the selector in the "
                   "sidebar to switch between your file and the original test period.")

    scored, up_hours = upload["df"], upload["hours"]
    for w in upload["warnings"]:
        st.warning(w)
    # Data unlike the training data gets flagged far more often than the training rate
    flag_rate = scored["Predicted_Anomaly"].mean()
    train_rate = float(load_models()["iso_forest"].contamination)
    if flag_rate > 3 * train_rate:
        st.warning(f"The Isolation Forest flagged {flag_rate:.0%} of hours as unusual, versus about "
                   f"{train_rate:.0%} in the training data. This data may differ from what the models were "
                   "trained on, so treat the results with caution.")

    summary = event_summary(up_hours) if len(up_hours) else pd.DataFrame()
    n_high = int((summary["System_Severity"] == "High").sum()) if len(summary) else 0
    cols = st.columns(4)
    cols[0].markdown(kpi_card("Hours Scored", f"{len(scored):,}", upload["period"], NAVY), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Confirmed Disruption Hours", f"{len(up_hours):,}",
                              f"{int(scored['Predicted_Anomaly'].sum() - len(up_hours))} single-hour flags "
                              "filtered out", CONFIRMED_RED), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Confirmed Disruption Events", f"{len(summary):,}",
                              "back-to-back confirmed hours = 1 event", "#D9773A"), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("High-Severity Events", f"{n_high:,}", "events reaching High system severity",
                              "#D04848"), unsafe_allow_html=True)
    st.write("")

    if "System_Status" in scored.columns:
        with st.container(border=True):
            st.markdown("#### Check against the labels in your file")
            actual = scored["System_Status"].astype(str)
            disrupted = actual != "Normal"
            confirmed = scored[CONFIRMED_COL] == 1
            c1, c2, c3 = st.columns(3)
            c1.metric("Disruption hours caught (recall)",
                      f"{(confirmed & disrupted).sum() / max(disrupted.sum(), 1):.1%}",
                      help="Share of the truly disrupted hours that were confirmed (≥2h).")
            c2.metric("Normal hours wrongly confirmed", f"{int((confirmed & ~disrupted).sum())}",
                      help="Hours labelled Normal in your file that were confirmed as a disruption.")
            c3.metric("Disruption type accuracy", f"{(scored['Predicted_Anomaly_Type'] == actual).mean():.1%}",
                      help="Share of all hours where XGBoost's predicted type matches System_Status.")

    with st.container(border=True):
        if summary.empty:
            st.markdown("#### No confirmed disruptions")
            st.info("No unusual readings lasted 2 or more consecutive hours in this file.")
        else:
            st.markdown(f"#### {len(summary)} confirmed disruption events ({len(up_hours)} hours)")
            st.dataframe(
                summary.style.map(color_severity, subset=["System_Severity"])
                .format({"Max_Confidence": "{:.1%}", "Start": "{:%Y-%m-%d %H:%M}", "End": "{:%Y-%m-%d %H:%M}"}),
                hide_index=True, width="stretch")
        out = scored.copy()
        out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Download all scored hours (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"scored_{Path(upload['name']).stem}.csv", mime="text/csv",
                           icon=":material/download:")


# =========================
# Page 4: Disruption Logs
# =========================

def page_logs():
    page_header("Disruption Logs", f"Every confirmed disruption, {view_period}")
    upload_banner()

    if view_hours.empty:
        st.info("No confirmed disruptions in this dataset.")
        return

    summary = event_summary(view_hours)

    with st.container(border=True):
        st.markdown(f"#### {len(summary)} confirmed disruption events ({len(view_hours)} hours)")
        st.dataframe(
            summary.style.map(color_severity, subset=["System_Severity"])
            .format({"Max_Confidence": "{:.1%}", "Start": "{:%Y-%m-%d %H:%M}", "End": "{:%Y-%m-%d %H:%M}"}),
            hide_index=True, width="stretch")

        hourly_cols = ["Timestamp", "Disruption", "Anomaly_Score", "Prediction_Confidence", "System_Severity",
                       "Affected_Facilities", "Recommendation", "SHAP_Top_Contributors"]
        hourly = view_hours[hourly_cols + (["System_Status"] if "System_Status" in view_hours else [])].rename(
            columns={"Disruption": "Predicted_Disruption", "Recommendation": "Monitoring_Advisory",
                     "System_Status": "Recorded_Status"})
        hourly["Predicted_Disruption"] = hourly["Predicted_Disruption"].map(DISRUPTION_LABELS)

        with st.expander("Hourly detail", expanded=False):
            st.dataframe(
                hourly.style.map(color_severity, subset=["System_Severity"])
                .format({"Anomaly_Score": "{:+.3f}", "Prediction_Confidence": "{:.1%}",
                         "Timestamp": "{:%Y-%m-%d %H:%M}"}),
                hide_index=True, width="stretch")

        c1, c2, _ = st.columns([1, 1, 1.2])
        c1.download_button("Download event summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           file_name="confirmed_disruption_events.csv", mime="text/csv", width="stretch")
        c2.download_button("Download hourly log (CSV)", hourly.to_csv(index=False).encode("utf-8"),
                           file_name="confirmed_disruption_hourly_log.csv", mime="text/csv", width="stretch")

    # Robustness check of the severity rules (from ImpactAssessment.py): result only, no cut-off values
    sens = load_result_csv("impact_threshold_sensitivity.csv").set_index("Scenario")
    level_cols = [c for c in sens.columns if c.endswith("_Records")]
    total = int(sens.loc["Baseline", level_cols].sum())
    with st.expander("Severity rule robustness (±15% threshold check)"):
        low_n, high_n = int(sens.loc["-15%", "System_Severity_Changed"]), int(sens.loc["+15%", "System_Severity_Changed"])
        st.markdown(f"Moving every severity cut-off **down 15%** changes the severity tier of **{low_n} of {total}** "
                    f"anomalous test hours ({low_n / total:.0%}); moving them **up 15%** changes **{high_n}** "
                    f"({high_n / total:.0%}).")
        robustness = sens[level_cols + ["System_Severity_Changed"]].rename(
            columns=lambda c: c.replace("_Records", "").replace("System_Severity_Changed", "Tier changed"))
        st.dataframe(robustness, width="stretch")
        st.caption(f"Computed over the {total} hours of the original test period ({PERIOD}) that the Isolation "
                   "Forest confirmed (≥2h) or XGBoost classified as a disruption, a slightly wider set than its "
                   "confirmed events. It describes the severity rules themselves, so it does not change with an "
                   "uploaded dataset.")


# =========================
# Page 5: Model Performance
# =========================

def page_performance():
    page_header("Model Performance", "Test-period performance of the detection, classification and impact modules")

    anomaly = load_result_csv("anomaly_detection_comparison.csv").copy()
    anomaly["Model"] = anomaly["Model"].str.replace("+ 2h rule", "+ 2h persistence filter", regex=False)
    anomaly = anomaly.rename(columns={"False_Alarm_Hours_Per_Day": "False_Positive_Hours_Per_Day"})
    anomaly["Base_Model"] = anomaly["Model"].str.replace(" + 2h persistence filter", "", regex=False)
    anomaly["Variant"] = np.where(anomaly["Model"].str.contains("persistence"), "+ 2h persistence filter", "Raw")

    iso = anomaly[anomaly["Base_Model"] == "Isolation Forest"].set_index("Variant")
    if {"Raw", "+ 2h persistence filter"} <= set(iso.index):
        raw, filt = iso.loc["Raw"], iso.loc["+ 2h persistence filter"]
        cols = st.columns(4)
        cols[0].markdown(kpi_card("Isolation Forest False Positives", f"{int(raw['FP'])} &#8594; {int(filt['FP'])}",
                                  "raw vs + 2h persistence filter", CONFIRMED_RED), unsafe_allow_html=True)
        cols[1].markdown(kpi_card("Anomaly F1-Score", f"{raw['Anomaly_F1']:.2f} &#8594; {filt['Anomaly_F1']:.2f}",
                                  "raw vs + 2h persistence filter", PRIMARY_LINE), unsafe_allow_html=True)
        cols[2].markdown(kpi_card("Events Detected", f"{int(filt['Events_Detected'])}/{int(filt['Total_Events'])}",
                                  "with the 2h persistence filter", "#3FA27F"), unsafe_allow_html=True)
        cols[3].markdown(kpi_card("ROC-AUC", f"{raw['ROC_AUC']:.3f}", "Isolation Forest anomaly score", NAVY),
                         unsafe_allow_html=True)
        st.write("")

    with st.container(border=True):
        st.markdown("#### Anomaly Detection: Raw vs. 2h Persistence Filter")
        c1, c2 = st.columns(2)
        for container, metric, title in [(c1, "FP", "False positives (hours)"), (c2, "Anomaly_F1", "Anomaly F1-score")]:
            fig = go.Figure()
            for variant, color in [("Raw", "#C3CDD9"), ("+ 2h persistence filter", NAVY)]:
                part = anomaly[anomaly["Variant"] == variant]
                fig.add_trace(go.Bar(x=part["Base_Model"], y=part[metric], name=variant, marker_color=color,
                                     text=part[metric].round(2), textposition="outside", cliponaxis=False))
            fig.update_layout(title=title, barmode="group")
            style_fig(fig, height=360)
            # Legend below the bars so it cannot overlap the chart title
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
                              margin=dict(t=50, b=10))
            container.plotly_chart(fig, width="stretch")

        st.caption("**How to read:** each pair of bars compares a model before (grey) and after (navy) the 2-hour "
                   "persistence filter, which ignores unusual readings that last only one hour. "
                   "**False positives** are normal hours wrongly flagged as disruptions, so *lower is better*. "
                   "**F1-score** combines catching real disruptions with avoiding false flags (0 to 1), so "
                   "*higher is better*. Isolation Forest is the model used by this dashboard.")

        with st.expander("Full metrics"):
            st.dataframe(
                anomaly.drop(columns=["Base_Model", "Variant"]).style.format({
                    "Accuracy": "{:.3f}", "Anomaly_Precision": "{:.3f}", "Anomaly_Recall": "{:.3f}",
                    "Anomaly_F1": "{:.3f}", "ROC_AUC": "{:.3f}", "False_Positive_Hours_Per_Day": "{:.2f}",
                    "Inference_Time_ms": "{:.1f}"}),
                hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown("#### Disruption Classification: XGBoost (6 classes)")
        # True label vs XGBoost prediction for every test hour (same rows as Train_Classifier.py's test split)
        order = [c for c in CLASS_ORDER if c in set(df["System_Status"]) | set(df["Predicted_Anomaly_Type"])]
        short = [CLASS_SHORT[c] for c in order]
        cm = confusion_matrix(df["System_Status"], df["Predicted_Anomaly_Type"], labels=order)
        row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig = go.Figure(go.Heatmap(
                z=row_pct, x=short, y=short, text=cm, texttemplate="%{text}", showscale=False,
                colorscale=[[0, "#F5F8FC"], [1, NAVY]], zmin=0, zmax=1, xgap=2, ygap=2,
                hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{text} hours (%{z:.1%} of row)<extra></extra>"))
            fig.update_layout(title="Confusion matrix (hours; shading = share of actual class)")
            fig.update_xaxes(title_text="Predicted", side="bottom", showline=False)
            fig.update_yaxes(title_text="Actual", autorange="reversed", showgrid=False, showline=False)
            st.plotly_chart(style_fig(fig, height=400), width="stretch")
            st.caption("**How to read:** each row is what actually happened and each column is what XGBoost "
                       "predicted. Numbers on the **diagonal** (top-left to bottom-right) are correct predictions; "
                       "any number **off the diagonal** is a mistake, e.g. the row *PCG Under.* / column "
                       "*Normal* counts PCG faults the model missed.")
        with c2:
            precision, recall, f1, support = precision_recall_fscore_support(
                df["System_Status"], df["Predicted_Anomaly_Type"], labels=order, zero_division=0)
            per_class = pd.DataFrame({"Class": short, "Precision": precision, "Recall": recall, "F1": f1,
                                      "Hours": support})
            st.markdown("**Per-class scores**")
            st.dataframe(per_class.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                         hide_index=True, width="stretch")
            st.caption("Recall = share of each actual class the model caught; precision = share of its "
                       "predictions for that class that were correct. Hours = test hours of that class.")

        st.markdown("**Model comparison**")
        classification = load_result_csv("classification_comparison.csv")
        st.dataframe(classification.style.format({c: "{:.4f}" for c in classification.columns
                                                  if c not in ("Model", "Inference_Time_ms")}
                                                 | {"Inference_Time_ms": "{:.1f}"}),
                     hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown("#### Global SHAP Feature Importance (XGBoost)")
        importance = load_result_csv("shap_global_importance.csv")
        class_choice = st.selectbox("Class", ["Mean_Abs_SHAP_All_Classes"] + CLASSES,
                                    format_func=lambda c: "All classes (mean)" if c.startswith("Mean_")
                                    else DISRUPTION_LABELS.get(c, c), filter_mode=None)  # select only, no typing
        ranked = importance.sort_values(class_choice)
        fig = go.Figure(go.Bar(x=ranked[class_choice], y=ranked["Feature"].str.replace("_", " "), orientation="h",
                               marker_color=PRIMARY_LINE, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Average influence of each sensor on the decision",
                          xaxis_title="Mean |SHAP value| (longer bar = sensor matters more)")
        st.plotly_chart(style_fig(fig, height=420), width="stretch")

        # Plain-language takeaway for the selected class, generated from the chart values
        top = ranked.iloc[::-1]
        names = [f.replace("_", " ") for f in top["Feature"]]
        values = top[class_choice].to_numpy()
        key = [n for n, v in zip(names, values) if v >= 0.05 * values[0]][:3]  # sensors that clearly matter
        negligible = [n for n, v in zip(names, values) if v < 0.05 * values[0]]
        if class_choice.startswith("Mean_"):
            focus = "deciding the disruption type, averaged over all six classes"
        else:
            focus = f"deciding whether an hour is <b>{DISRUPTION_LABELS.get(class_choice, class_choice)}</b>"
        listed = ", ".join(f"<b>{n}</b>" for n in key[:-1]) + (" and " if len(key) > 1 else "") + f"<b>{key[-1]}</b>"
        takeaway = f"When {focus}, XGBoost relies mainly on {listed}."
        if negligible:
            takeaway += (f" The {'other ' if len(negligible) > 3 else ''}{len(negligible)} sensor"
                         f"{'s' if len(negligible) > 1 else ''}"
                         f"{' (' + ', '.join(negligible) + ')' if len(negligible) <= 3 else ''} "
                         f"barely affect{'s' if len(negligible) == 1 else ''} this decision.")
        st.markdown(
            f'<div class="summary"><b>What this chart shows:</b> SHAP measures how much each sensor reading '
            f'pushed XGBoost towards or away from its decision, averaged over every hour in the test period. '
            f'A <b>longer bar</b> means the model leans on that sensor more; a bar near zero means the sensor '
            f'hardly matters. Use the <b>Class</b> dropdown to see what the model looks at for one disruption type. '
            f'It shows <i>which</i> sensors matter, not whether their readings were high or low; the Event '
            f'Inspector shows that for a single hour.<br><br><b>Takeaway:</b> {takeaway}</div>',
            unsafe_allow_html=True)


{
    "Supply Chain Overview": page_overview,
    "Event Inspector": page_inspector,
    "Disruption Logs": page_logs,
    "Upload & Score": page_upload,
    "Model Performance": page_performance,
}[st.session_state["page"]]()
