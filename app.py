import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Type 2 Diabetes Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Design tokens
# ---------------------------------------------------------
INK = "#0C1A2B"        # page background
PANEL = "#12263B"      # cards
EDGE = "#23415E"       # borders / grid
SAND = "#EDE3D0"       # primary text
MUTED = "#8FA3B8"      # secondary text
CORAL = "#FF6B57"      # high risk
SEA = "#6FD3B5"        # low risk
AMBER = "#F4B860"      # glucose line
SKY = "#6EA8FE"        # protective / negative SHAP


# ---------------------------------------------------------
# Global styling
# ---------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', sans-serif;
    color: {SAND};
}}
.stApp {{
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(255,107,87,0.10), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(110,168,254,0.10), transparent 60%),
        {INK};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}

section[data-testid="stSidebar"] {{
    background: #0A1524;
    border-right: 1px solid {EDGE};
}}

h1, h2, h3 {{ font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }}

.hero-title {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 3rem;
    line-height: 1.05;
    margin: 0;
    color: {SAND};
}}
.hero-sub {{
    color: {MUTED};
    font-size: 1.05rem;
    max-width: 680px;
    margin-top: 0.7rem;
}}
.chip {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid {EDGE};
    background: rgba(18,38,59,0.8);
    color: {MUTED};
    font-size: 0.8rem;
    margin-right: 8px;
}}
.chip.warn {{ border-color: rgba(244,184,96,0.5); color: {AMBER}; }}

.card {{
    background: {PANEL};
    border: 1px solid {EDGE};
    border-radius: 18px;
    padding: 22px 24px;
    height: 100%;
}}
.card-label {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 6px; }}
.card-value {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2rem;
    line-height: 1.1;
    color: {SAND};
}}
.card-unit {{ display: block; font-size: 0.85rem; color: {MUTED}; margin-top: 2px; font-family: 'DM Sans', sans-serif; }}
.card-value {{ white-space: nowrap; }}
.card-note {{ color: {MUTED}; font-size: 0.8rem; margin-top: 8px; }}

.verdict {{
    border-radius: 18px;
    padding: 22px 26px;
    border: 1px solid;
}}
.verdict.high {{ background: rgba(255,107,87,0.10); border-color: rgba(255,107,87,0.55); }}
.verdict.low  {{ background: rgba(111,211,181,0.10); border-color: rgba(111,211,181,0.55); }}
.verdict-head {{
    font-family: 'Fraunces', serif;
    font-size: 1.7rem;
    font-weight: 600;
    margin-bottom: 4px;
}}
.verdict.high .verdict-head {{ color: {CORAL}; }}
.verdict.low .verdict-head {{ color: {SEA}; }}
.verdict-body {{ color: {SAND}; opacity: 0.85; font-size: 0.95rem; }}

.section-title {{
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 600;
    margin: 2.4rem 0 0.2rem 0;
}}
.section-sub {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 1rem; max-width: 760px; }}

.driver {{
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 12px 0;
    border-bottom: 1px solid {EDGE};
}}
.driver:last-child {{ border-bottom: none; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; margin-top: 7px; flex-shrink: 0; }}
.driver-name {{ font-weight: 600; }}
.driver-text {{ color: {MUTED}; font-size: 0.88rem; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {PANEL};
    border: 1px solid {EDGE} !important;
    border-radius: 18px;
    padding: 6px 12px;
}}

.foot {{ color: {MUTED}; font-size: 0.82rem; margin-top: 3rem; text-align: center; }}
</style>
""",
    unsafe_allow_html=True,
)


def show(fig):
    """Render a plotly figure full-width (works on old and new Streamlit)."""
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    shap_df = pd.read_csv("data/processed/demo_shap_explanations.csv")
    ml_df = pd.read_csv("data/processed/ml_ready.csv")
    return shap_df, ml_df


shap_df, ml_df = load_data()


# ---------------------------------------------------------
# Sidebar: patient selection
# ---------------------------------------------------------
patients = shap_df["patient_id"].tolist()

st.sidebar.markdown("### Patients")
st.sidebar.caption("Pick a synthetic patient to open their digital twin.")

selected_patient = st.sidebar.selectbox(
    "Select patient",
    patients,
    format_func=lambda pid: f"Patient {patients.index(pid) + 1:02d} · {str(pid)[:8]}",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Everything shown here is synthetic. Built for the "
    "Digital Twin Challenge 2026."
)


# ---------------------------------------------------------
# Selected patient
# ---------------------------------------------------------
patient_prediction = shap_df[shap_df["patient_id"] == selected_patient].iloc[0]
patient_data = ml_df[ml_df["patient_id"] == selected_patient].copy()

current_glucose = patient_prediction["current_glucose"]
risk = float(patient_prediction["predicted_probability"])
is_high = patient_prediction["predicted_spike"] == 1
accent = CORAL if is_high else SEA

age = patient_data["age"].iloc[0]
hba1c = patient_data["hba1c"].iloc[0]
bmi = patient_data["bmi"].iloc[0]

# Actual outcome at the prediction timestamp (ground truth from the data)
pred_ts = pd.to_datetime(patient_prediction["timestamp"])
actual_spike = None
actual_glucose = None
if {"glucose_spike_2h", "future_glucose_2h"}.issubset(patient_data.columns):
    match = patient_data[pd.to_datetime(patient_data["timestamp"]) == pred_ts]
    if not match.empty:
        actual_spike = int(match.iloc[0]["glucose_spike_2h"])
        actual_glucose = float(match.iloc[0]["future_glucose_2h"])


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
<div class="hero-title">Type 2 Diabetes<br>Digital Twin</div>
<div class="hero-sub">
A virtual copy of a patient that watches their glucose and forecasts
whether the modeled glucose target will be exceeded two hours from now.
</div>
<div style="margin-top:16px">
<span class="chip">Simulated EHR + synthetic CGM</span>
<span class="chip warn">Research PoC · not for clinical use</span>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------------------------------------
# Digital Twin State
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title" style="margin-top:1.2rem">'
    'Digital Twin State'
    '</div>',
    unsafe_allow_html=True,
)

state_col1, state_col2, state_col3, state_col4 = st.columns(4)

state_cards = [
    (
        state_col1,
        "Status",
        "● Monitoring",
        "Patient state loaded",
        SEA,
    ),
    (
        state_col2,
        "Signals",
        "CGM · Activity",
        "Synthetic wearable stream",
        SKY,
    ),
    (
        state_col3,
        "EHR context",
        "Active",
        "Demographics + laboratory features",
        AMBER,
    ),
    (
        state_col4,
        "Forecast horizon",
        "2 hours",
        "Exact 2-hour prediction target",
        CORAL,
    ),
]

for col, label, value, note, color in state_cards:
    col.markdown(
        f"""
<div class="card">
<div class="card-label">{label}</div>
<div class="card-value" style="font-size:1.45rem;color:{color}">
{value}
</div>
<div class="card-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Scorecard: how did the model do across all demo patients?
# ---------------------------------------------------------
@st.cache_data
def build_scorecard(shap_df, ml_df):
    needed = {"glucose_spike_2h", "future_glucose_2h"}
    if not needed.issubset(ml_df.columns):
        return None
    s = shap_df.copy()
    s["_ts"] = pd.to_datetime(s["timestamp"])
    m = ml_df[["patient_id", "timestamp", "glucose_spike_2h", "future_glucose_2h"]].copy()
    m["_ts"] = pd.to_datetime(m["timestamp"])
    joined = s.merge(
        m[["patient_id", "_ts", "glucose_spike_2h", "future_glucose_2h"]],
        on=["patient_id", "_ts"],
        how="inner",
    )
    if joined.empty:
        return None
    joined["correct"] = (
        joined["predicted_spike"].astype(int) == joined["glucose_spike_2h"].astype(int)
    )
    return joined


score = build_scorecard(shap_df, ml_df)

if score is not None:
    n = len(score)
    n_correct = int(score["correct"].sum())
    actual_spikes = int((score["glucose_spike_2h"] == 1).sum())
    false_alarms = int(
        ((score["predicted_spike"] == 1) & (score["glucose_spike_2h"] == 0)).sum()
    )
    missed = int(
        ((score["predicted_spike"] == 0) & (score["glucose_spike_2h"] == 1)).sum()
    )

    st.markdown(
        '<div class="section-title" style="margin-top:0.5rem">How the model did on these patients</div>'
        '<div class="section-sub">Each demo patient has one highlighted moment. '
        "Here is every prediction checked against what really happened 2 hours later.</div>",
        unsafe_allow_html=True,
    )

    sc = st.columns(4)
    score_cards = [
        ("Correct calls", f"{n_correct}/{n}", f"{n_correct / n * 100:.0f}% of demo patients"),
        ("Real spikes", f"{actual_spikes}", "crossed the synthetic target"),
        ("False alarms", f"{false_alarms}", "predicted spike, none happened"),
        ("Missed spikes", f"{missed}", "spike happened, not predicted"),
    ]
    for col, (label, value, note) in zip(sc, score_cards):
        col.markdown(
            f"""
<div class="card">
<div class="card-label">{label}</div>
<div class="card-value">{value}</div>
<div class="card-note">{note}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # One dot per patient: green = right, amber = missed. Ring = selected.
    status = dict(zip(score["patient_id"], score["correct"]))
    dots = ""
    for pid in patients:
        if pid not in status:
            continue
        color = SEA if status[pid] else AMBER
        ring = f"box-shadow:0 0 0 3px {INK}, 0 0 0 5px {SAND};" if pid == selected_patient else ""
        dots += (
            f'<span title="Patient {patients.index(pid) + 1:02d}" '
            f'style="display:inline-block;width:14px;height:14px;border-radius:50%;'
            f'background:{color};margin:0 8px 8px 0;{ring}"></span>'
        )
    st.markdown(
        f'<div style="margin-top:14px">{dots}'
        f'<span class="card-note" style="margin-left:6px">green = correct, amber = missed, ring = patient you are viewing</span></div>',
        unsafe_allow_html=True,
    )

    st.write("")


# ---------------------------------------------------------
# Held-out model performance
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">'
    'Model validation'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-sub">'
    'Performance on the untouched patient-level test set. '
    'Patients were kept separate between training and testing '
    'to reduce patient-level leakage.'
    '</div>',
    unsafe_allow_html=True,
)

metrics = st.columns(4)

model_metrics = [
    ("ROC-AUC", "0.972", "Ranking performance"),
    ("F1-score", "0.645", "Balance of precision and recall"),
    ("Precision", "0.540", "Positive prediction precision"),
    ("Recall", "0.800", "Spike detection rate"),
]

for col, (label, value, note) in zip(metrics, model_metrics):
    col.markdown(
        f"""
<div class="card">
<div class="card-label">{label}</div>
<div class="card-value">{value}</div>
<div class="card-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Row 1: risk ring + verdict + patient profile
# ---------------------------------------------------------
left, right = st.columns([1.05, 1.6], gap="large")

with left:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={
                "suffix": "%",
                "valueformat": ".1f",
                "font": {"size": 54, "family": "Fraunces", "color": SAND},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 50, 100],
                    "tickcolor": MUTED,
                    "tickfont": {"color": MUTED, "size": 11},
                },
                "bar": {"color": accent, "thickness": 0.28},
                "bgcolor": PANEL,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(111,211,181,0.18)"},
                    {"range": [40, 70], "color": "rgba(244,184,96,0.18)"},
                    {"range": [70, 100], "color": "rgba(255,107,87,0.20)"},
                ],
            },
        )
    )
    gauge.update_layout(
        height=270,
        margin=dict(l=30, r=30, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans", "color": SAND},
    )

    with st.container(border=True):
        st.markdown(
            '<div class="card-label">Chance of a spike in 2 hours</div>',
            unsafe_allow_html=True,
        )
        show(gauge)

with right:
    if is_high:
        st.markdown(
            f"""
<div class="verdict high">
<div class="verdict-head">High spike risk · {risk * 100:.1f}%</div>
<div class="verdict-body">
The model expects this patient's glucose to exceed the synthetic modeling target
two hours after the marked moment.
</div></div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="verdict low">
<div class="verdict-head">Low spike risk · {risk * 100:.1f}%</div>
<div class="verdict-body">
The model expects this patient's glucose to stay below the synthetic modeling target
two hours after the marked moment.
</div></div>
""",
            unsafe_allow_html=True,
        )

    st.write("")

    if actual_spike is not None:
        predicted_spike = int(patient_prediction["predicted_spike"])
        correct = predicted_spike == actual_spike
        badge_color = SEA if correct else AMBER
        badge_text = "Model was right" if correct else "Model missed this one"
        outcome_text = (
            "exceeded the synthetic modeling target"
            if actual_spike == 1
            else "stayed below the synthetic modeling target"
        )
        st.markdown(
            f"""
<div class="card" style="padding:18px 24px">
<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
<div>
<div class="card-label">What actually happened 2 hours later</div>
<div class="card-value" style="font-size:1.8rem">{actual_glucose:.0f}<span class="card-unit">mg/dL</span></div>
<div class="card-note">Glucose {outcome_text}.</div>
</div>
<div class="chip" style="border-color:{badge_color};color:{badge_color};font-size:0.9rem">{badge_text}</div>
</div></div>
""",
            unsafe_allow_html=True,
        )
        st.write("")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Glucose now", f"{current_glucose:.0f}", "mg/dL"),
        (c2, "HbA1c", f"{hba1c:.1f}", "%"),
        (c3, "BMI", f"{bmi:.1f}", ""),
        (c4, "Age", f"{age:.0f}", "yrs"),
    ]
    for col, label, value, unit in cards:
        col.markdown(
            f"""
<div class="card">
<div class="card-label">{label}</div>
<div class="card-value">{value}<span class="card-unit">{unit}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="card-note" style="margin-top:12px">'
        "Prediction target: glucose above the synthetic 160 mg/dL modeling target exactly 2 hours ahead "
        "(synthetic modeling target)."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# CGM timeline with forecast window
# ---------------------------------------------------------
st.markdown('<div class="section-title">24 hours of glucose</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">The amber line is the synthetic CGM trace. '
    "The shaded band after the marker is the 2-hour window the model is "
    "forecasting. Values above the synthetic modeling target are labeled as spikes.</div>",
    unsafe_allow_html=True,
)

timeline = patient_data.copy()
timeline["timestamp"] = pd.to_datetime(timeline["timestamp"])
timeline = timeline.sort_values("timestamp")

smooth = (
    timeline.set_index("timestamp")["glucose_mg_dl"].rolling("30min").mean()
)

prediction_time = pd.to_datetime(patient_prediction["timestamp"])
prediction_glucose = patient_prediction["current_glucose"]
horizon_end = prediction_time + pd.Timedelta(hours=2)

has_steps = "steps" in timeline.columns
rows = 2 if has_steps else 1

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.8, 0.2] if has_steps else [1],
)

y_min = min(timeline["glucose_mg_dl"].min() - 10, 90)
y_max = max(timeline["glucose_mg_dl"].max() + 15, 190)

# Raw trace (faint) + smooth trace
fig.add_trace(
    go.Scatter(
        x=timeline["timestamp"],
        y=timeline["glucose_mg_dl"],
        mode="lines",
        name="CGM reading",
        line=dict(color=AMBER, width=1),
        opacity=0.35,
        hovertemplate="%{x|%H:%M}<br>%{y:.1f} mg/dL<extra>CGM</extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=smooth.index,
        y=smooth.values,
        mode="lines",
        name="30-min average",
        line=dict(color=AMBER, width=3, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(244,184,96,0.08)",
        hovertemplate="%{x|%H:%M}<br>%{y:.1f} mg/dL<extra>30-min avg</extra>",
    ),
    row=1,
    col=1,
)

# Prediction marker with a soft halo
fig.add_trace(
    go.Scatter(
        x=[prediction_time],
        y=[prediction_glucose],
        mode="markers",
        name="Prediction moment",
        marker=dict(
            size=26,
            color=accent,
            opacity=0.25,
            line=dict(width=0),
        ),
        hoverinfo="skip",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[prediction_time],
        y=[prediction_glucose],
        mode="markers",
        name="Prediction moment",
        marker=dict(size=12, color=accent, line=dict(color=SAND, width=2)),
        hovertemplate=(
            "%{x|%H:%M}<br>%{y:.1f} mg/dL<br>"
            f"Spike risk: {risk * 100:.1f}%<extra>Prediction</extra>"
        ),
    ),
    row=1,
    col=1,
)

# Actual glucose 2 hours later, joined to the prediction point
if actual_glucose is not None:
    fig.add_trace(
        go.Scatter(
            x=[prediction_time, horizon_end],
            y=[prediction_glucose, actual_glucose],
            mode="lines",
            line=dict(color=SKY, width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[horizon_end],
            y=[actual_glucose],
            mode="markers",
            name="Actual glucose at +2h",
            marker=dict(
                size=13,
                symbol="diamond",
                color=SKY,
                line=dict(color=SAND, width=2),
            ),
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} mg/dL<extra>Actual +2h</extra>",
        ),
        row=1,
        col=1,
    )

# Shapes go AFTER the traces: plotly silently drops subplot shapes
# added before the subplot has any data.
# Danger zone
fig.add_hrect(
    y0=160, y1=y_max, fillcolor=CORAL, opacity=0.10, line_width=0, layer="below", row=1, col=1
)
fig.add_hline(
    y=160,
    line_dash="dash",
    line_color=CORAL,
    line_width=1,
    annotation_text="Synthetic modeling target · 160 mg/dL",
    annotation_position="top left",
    annotation_font_color=CORAL,
    row=1,
    col=1,
)

# Forecast window
fig.add_vrect(
    x0=prediction_time,
    x1=horizon_end,
    fillcolor=SKY,
    opacity=0.12,
    line_width=0,
    layer="below",
    annotation_text="Next 2 hours",
    annotation_position="bottom left",
    annotation_font_color=SKY,
    row=1,
    col=1,
)

# Activity strip
if has_steps:
    fig.add_trace(
        go.Bar(
            x=timeline["timestamp"],
            y=timeline["steps"],
            name="Steps",
            marker_color=SEA,
            opacity=0.7,
            hovertemplate="%{x|%H:%M}<br>%{y} steps<extra>Activity</extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Steps", row=2, col=1, showgrid=False)

fig.update_yaxes(
    title_text="mg/dL", range=[y_min, y_max], row=1, col=1
)
fig.update_layout(
    height=560 if has_steps else 480,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=SAND),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
        font=dict(color=MUTED),
    ),
    bargap=0.1,
)
fig.update_xaxes(showgrid=False, linecolor=EDGE, tickformat="%H:%M")
fig.update_yaxes(gridcolor=EDGE, zeroline=False)

show(fig)


# ---------------------------------------------------------
# Explainability
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">Why did the model say this?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">The three features that pushed the prediction '
    "the most at the marked moment (SHAP values). They describe how the "
    "model behaves, not proven biological causes.</div>",
    unsafe_allow_html=True,
)

feature_names = {
    "hour": "Time of day",
    "minute": "Minute of hour",
    "glucose_mg_dl": "Current glucose",
    "glucose_change_15min": "15-minute glucose change",
    "glucose_rolling_mean_30min": "30-minute glucose pattern",
    "glucose_rolling_std_30min": "30-minute glucose variability",
    "steps": "Current activity",
    "steps_30min": "Recent activity (30 min)",
    "sleep_stage_encoded": "Sleep stage",
    "gender_encoded": "Gender",
    "age": "Age",
    "hba1c": "HbA1c",
    "bmi": "BMI",
    "baseline_glucose": "Baseline glucose",
    "cholesterol": "Cholesterol",
    "triglycerides": "Triglycerides",
    "medication_count": "Medication count",
}

top_features = [
    (patient_prediction["top_feature_1"], float(patient_prediction["top_feature_1_shap"])),
    (patient_prediction["top_feature_2"], float(patient_prediction["top_feature_2_shap"])),
    (patient_prediction["top_feature_3"], float(patient_prediction["top_feature_3_shap"])),
]

labels = [feature_names.get(f, f) for f, _ in top_features]
values = [v for _, v in top_features]

chart_col, text_col = st.columns([1.3, 1], gap="large")

with chart_col:
    # reversed so the strongest driver sits on top
    bar = go.Figure(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=[CORAL if v >= 0 else SKY for v in values[::-1]],
            text=[f"{v:+.2f}" for v in values[::-1]],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=SAND),
            hovertemplate="%{y}<br>SHAP %{x:+.2f}<extra></extra>",
        )
    )
    bar.add_vline(x=0, line_color=MUTED, line_width=1)
    bar.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=SAND),
        margin=dict(l=10, r=60, t=10, b=60),
        xaxis=dict(
            range=[min(0, min(values)) * 1.3, max(0, max(values)) * 1.3],
            title="Pushes risk down  ←   →  Pushes risk up",
            gridcolor=EDGE,
            zeroline=False,
            title_font=dict(color=MUTED, size=12),
        ),
        yaxis=dict(showgrid=False, automargin=True),
    )
    show(bar)

with text_col:
    items = ""
    for name, v in zip(labels, values):
        color = CORAL if v >= 0 else SKY
        verb = "Raised" if v >= 0 else "Lowered"
        items += (
            f'<div class="driver"><div class="dot" style="background:{color}"></div>'
            f'<div><div class="driver-name">{name}</div>'
            f'<div class="driver-text">{verb} the predicted spike risk '
            f"(SHAP {v:+.2f})</div></div></div>"
        )
    st.markdown(f'<div class="card">{items}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    '<div class="foot">Digital Twin Challenge 2026 · '
    "Type 2 Diabetes Glucose Spike Prediction PoC · synthetic data only</div>",
    unsafe_allow_html=True,
)