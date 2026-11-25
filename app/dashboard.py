"""Plant Performance Lab dashboard with a clear glass interface."""
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote, urlparse

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, FEATURES, LABELS, load_data
from src.chatbot import context, answer, predict, offline, ollama_answer, ollama_models

st.set_page_config(
    page_title="Plant Performance Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEXT = "#16324A"
MUTED = "#5F788B"
TEAL = "#47C5BE"
CYAN = "#7FD7F2"
BLUE = "#8AA9D6"
AMBER = "#E9B96E"
WHITE = "#FFFFFF"
CARD = "rgba(255,255,255,0.56)"
GRID = "rgba(89, 111, 131, 0.16)"
SLIDES_DEFAULT = "https://docs.google.com/presentation/d/1Uj1B1aXG04VDL4O6AU2daHEBnA8wsqHOsEDvCKaP1GY/edit"
GITHUB_DEFAULT = "https://github.com/RichWOkello/plant-performance-lab"

st.markdown(
    """
<style>
:root {
    --ink: #16324A;
    --muted: #5F788B;
    --glass: rgba(255,255,255,0.56);
    --glass-strong: rgba(255,255,255,0.78);
    --edge: rgba(255,255,255,0.66);
    --teal: #47C5BE;
    --blue: #8AA9D6;
    --cyan: #7FD7F2;
    --amber: #E9B96E;
    --shadow: 0 20px 50px rgba(130, 160, 185, 0.18);
}
html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% 0%, rgba(177, 219, 235, 0.42), transparent 25%),
        radial-gradient(circle at 90% 12%, rgba(203, 221, 246, 0.42), transparent 25%),
        linear-gradient(180deg, #F8FCFF 0%, #EEF5FB 46%, #F8FBFF 100%);
    color: var(--ink);
}
[data-testid="stHeader"] {
    background: rgba(248, 252, 255, 0.60);
    backdrop-filter: blur(16px);
}
.block-container {
    max-width: 1580px;
    padding-top: 2rem;
    padding-bottom: 2.2rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.80), rgba(248,252,255,0.72));
    border-right: 1px solid rgba(255,255,255,0.72);
    backdrop-filter: blur(24px);
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.4rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.58rem 0.8rem;
    border: 1px solid transparent;
    border-radius: 16px;
    background: rgba(255,255,255,0.36);
    transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    transform: translateY(-1px);
    background: rgba(255,255,255,0.62);
    box-shadow: 0 10px 18px rgba(130, 160, 185, 0.12);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(235,247,252,0.86));
    border-color: rgba(142, 202, 198, 0.34);
    box-shadow: 0 12px 24px rgba(127, 172, 199, 0.15);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.84), rgba(255,255,255,0.50));
    border: 1px solid rgba(255,255,255,0.72);
    border-radius: 22px;
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.80);
    padding: 0.75rem 0.95rem;
    min-height: 115px;
    animation: rise 0.42s ease;
}
[data-testid="stMetricLabel"] {
    color: var(--muted);
}
[data-testid="stMetricValue"] {
    color: var(--ink);
    font-size: clamp(1.2rem, 2vw, 1.95rem);
    font-weight: 700;
    letter-spacing: -0.04em;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.72), rgba(255,255,255,0.40));
    border: 1px solid rgba(255,255,255,0.72);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.84);
    border-radius: 28px;
    backdrop-filter: blur(20px);
    animation: rise 0.45s ease;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 28px;
}
.stButton button,
.stDownloadButton button,
[data-testid="stLinkButton"] a {
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.85);
    background: linear-gradient(145deg, rgba(255,255,255,0.90), rgba(245,250,253,0.75));
    color: var(--ink);
    box-shadow: 0 10px 20px rgba(130, 160, 185, 0.10);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton button:hover,
.stDownloadButton button:hover,
[data-testid="stLinkButton"] a:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(124, 170, 198, 0.15);
    border-color: rgba(147, 216, 211, 0.50);
}
.stSlider [data-baseweb="slider"] > div div {
    background: linear-gradient(90deg, #7FD7F2, #47C5BE) !important;
}
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
textarea,
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.66) !important;
    border: 1px solid rgba(255,255,255,0.78) !important;
    border-radius: 15px !important;
}
.stRadio [role="radiogroup"] {
    gap: 0.45rem;
}
.stRadio [role="radiogroup"] label {
    background: rgba(255,255,255,0.42);
    border: 1px solid rgba(255,255,255,0.64);
    border-radius: 14px;
    padding: 0.5rem 0.75rem;
}
[data-testid="stChatMessage"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.82), rgba(255,255,255,0.56));
    border: 1px solid rgba(255,255,255,0.84);
    border-radius: 22px;
    backdrop-filter: blur(18px);
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 1.45rem 1.55rem 1.35rem;
    border-radius: 30px;
    border: 1px solid rgba(255,255,255,0.78);
    background: linear-gradient(145deg, rgba(255,255,255,0.84), rgba(255,255,255,0.48));
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.85);
    backdrop-filter: blur(18px);
    animation: rise 0.45s ease;
}
.hero::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    right: -60px;
    top: -60px;
    background: radial-gradient(circle, rgba(127,215,242,0.26), transparent 68%);
}
.hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    left: -70px;
    bottom: -90px;
    background: radial-gradient(circle, rgba(138,169,214,0.20), transparent 68%);
}
.kicker {
    color: #55829E;
    font: 600 0.72rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.hero h1 {
    margin: 0.45rem 0 0.15rem;
    font-size: clamp(2.05rem, 4vw, 3.45rem);
    line-height: 1.02;
    letter-spacing: -0.05em;
    color: var(--ink);
}
.hero p {
    max-width: 760px;
    margin: 0.4rem 0 0;
    color: #5B7588;
    font-size: 1rem;
}
.hero-pills {
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
    margin-top: 0.95rem;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    padding: 0.42rem 0.82rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.54);
    border: 1px solid rgba(255,255,255,0.74);
    color: #47677E;
    font-size: 0.78rem;
}
.hero-pill b {
    color: var(--ink);
}
.brand-block {
    margin: 0.2rem 0 0.5rem;
    font-size: 1.44rem;
    font-weight: 760;
    line-height: 1.06;
    letter-spacing: -0.04em;
    color: var(--ink);
}
.brand-block span {
    color: #4D8FA8;
}
.soft-note {
    background: linear-gradient(145deg, rgba(255,255,255,0.72), rgba(241,248,252,0.58));
    border: 1px solid rgba(255,255,255,0.84);
    border-left: 3px solid rgba(71,197,190,0.68);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    color: #486779;
}
.sensor-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 0.75rem;
}
.sensor-card {
    padding: 0.9rem 1rem;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(255,255,255,0.82), rgba(255,255,255,0.46));
    border: 1px solid rgba(255,255,255,0.84);
    box-shadow: 0 10px 24px rgba(130, 160, 185, 0.10);
}
.sensor-card b {
    color: #6C92A9;
    font: 600 0.72rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: 0.08em;
}
.sensor-card small {
    display: block;
    color: #6B8597;
    margin-top: 0.2rem;
}
.sensor-card strong {
    display: block;
    color: var(--ink);
    margin-top: 0.35rem;
    font-size: 1.2rem;
    font-variant-numeric: tabular-nums;
}
.mini-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 0.85rem;
}
.mini-card {
    padding: 0.9rem 1rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.50);
    border: 1px solid rgba(255,255,255,0.74);
}
.mini-card small { color: #6A8497; display:block; margin-bottom: 0.25rem; }
.mini-card strong { font-size: 1.05rem; color: var(--ink); }
.watermark {
    position: fixed;
    right: 1.2rem;
    top: 8.1rem;
    width: 300px;
    height: 300px;
    opacity: 0.12;
    pointer-events: none;
    z-index: 0;
}
.watermark svg {
    width: 100%;
    height: 100%;
}
.fab-chat {
    position: fixed;
    right: 1.55rem;
    bottom: 1.55rem;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    font-size: 1.55rem;
    color: #33536A;
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(243,249,253,0.84));
    border: 1px solid rgba(255,255,255,0.88);
    box-shadow: 0 22px 40px rgba(130, 160, 185, 0.22), inset 0 1px 0 rgba(255,255,255,0.85);
    backdrop-filter: blur(16px);
    z-index: 1000;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.fab-chat:hover {
    transform: translateY(-2px);
    box-shadow: 0 28px 44px rgba(130, 160, 185, 0.28);
}
.fab-chat span {
    position: absolute;
    right: 76px;
    width: max-content;
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    color: #466277;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(255,255,255,0.84);
    opacity: 0;
    transform: translateX(8px);
    transition: 0.18s ease;
}
.fab-chat:hover span {
    opacity: 1;
    transform: translateX(0px);
}
@keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 1100px) {
    .sensor-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .watermark { width: 220px; height: 220px; }
}
@media (max-width: 700px) {
    .block-container { padding-top: 1.2rem; padding-left: 0.8rem; padding-right: 0.8rem; }
    .sensor-grid, .mini-grid { grid-template-columns: 1fr; }
    .fab-chat { width: 58px; height: 58px; right: 1rem; bottom: 1rem; }
    .fab-chat span { display:none; }
    .watermark { opacity: 0.08; right: 0.5rem; top: 7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def setting(name, default=""):
    """Prefer explicit environment variables, then local .env."""
    local = dotenv_values(ROOT / ".env").get(name)
    if name == "GOOGLE_SLIDES_URL" and local:
        return str(local).strip()
    return str(os.getenv(name) or local or default).strip()


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return joblib.load(ROOT / "models/best_model.joblib")


@st.cache_data
def get_results():
    p = ROOT / "models"
    return (
        json.loads((p / "metrics.json").read_text()),
        pd.read_csv(p / "comparison.csv"),
        pd.read_csv(p / "importance.csv"),
        pd.read_csv(p / "predictions.csv"),
    )


@st.cache_data(ttl=15, show_spinner=False)
def available_ollama():
    try:
        return ollama_models()
    except Exception:
        return []


def get_query_page(default):
    try:
        query = st.query_params
        return query.get("page", default)
    except Exception:
        query = st.experimental_get_query_params()
        value = query.get("page", [default])
        return value[0] if isinstance(value, list) else value


def set_query_page(page):
    try:
        st.query_params["page"] = page
    except Exception:
        st.experimental_set_query_params(page=page)


PAGES = ["Overview", "Explore", "Predict", "Model Performance", "Ask the Data"]
NAV = {
    "Overview": "01   Overview",
    "Explore": "02   Explore",
    "Predict": "03   Predict",
    "Model Performance": "04   Model performance",
    "Ask the Data": "05   Ask the data",
}

default_page = get_query_page("Overview")
if default_page not in PAGES:
    default_page = "Overview"

if "workspace" not in st.session_state:
    st.session_state.workspace = default_page
elif st.session_state.workspace not in PAGES:
    st.session_state.workspace = "Overview"

st.sidebar.markdown(
    '<div class="kicker">COMBINED CYCLE POWER PLANT</div>'
    '<div class="brand-block"><span>Plant</span><br>Performance Lab</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Prediction, exploration and AI-assisted explanation")

page = st.sidebar.radio(
    "Workspace",
    PAGES,
    index=PAGES.index(st.session_state.workspace),
    format_func=NAV.get,
    key="workspace",
)
set_query_page(page)


def project_files():
    st.sidebar.divider()
    st.sidebar.subheader("Project links")
    notebook_path = ROOT / "notebooks/capstone.ipynb"
    presentation_path = ROOT / "reports/Plant_Performance_Lab_Presentation.pptx"
    slides_url = setting("GOOGLE_SLIDES_URL", SLIDES_DEFAULT)
    github_url = setting("GITHUB_REPO_URL", GITHUB_DEFAULT)

    if notebook_path.is_file():
        st.sidebar.download_button(
            "Download notebook",
            data=notebook_path.read_bytes(),
            file_name=notebook_path.name,
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    if presentation_path.is_file():
        st.sidebar.download_button(
            "Download presentation",
            data=presentation_path.read_bytes(),
            file_name=presentation_path.name,
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )

    parsed = urlparse(slides_url)
    if parsed.scheme == "https" and parsed.hostname == "docs.google.com" and parsed.path.startswith("/presentation/"):
        st.sidebar.link_button("Open Google Slides", slides_url, use_container_width=True)
    else:
        st.sidebar.link_button("Open Google Slides", "https://slides.google.com", use_container_width=True)
    if github_url.startswith("http"):
        st.sidebar.link_button("Open GitHub", github_url, use_container_width=True)
    st.sidebar.caption("UCI CCPP dataset · Full-load observations from one plant. This dashboard is an academic explainer, not a live control system.")


project_files()

try:
    df = get_data()
    model = get_model()
    metrics, comparison, importance, pairs = get_results()
except Exception as exc:
    st.error("The data or saved model could not be loaded. Check the project files or rerun the notebook.")
    st.caption(str(exc))
    st.stop()


def chart(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Inter, Arial, sans-serif", color=TEXT, size=12),
        margin=dict(l=14, r=14, t=50, b=34),
        title_font=dict(size=15, color=TEXT),
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, title_font=dict(color=TEXT))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, title_font=dict(color=TEXT))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def scatter_with_trend(frame, feature):
    fig = px.scatter(frame, x=feature, y="PE", labels=LABELS, opacity=0.40, color_discrete_sequence=[TEAL])
    if frame[feature].nunique() > 1:
        x_vals = np.array([frame[feature].min(), frame[feature].max()])
        trend = np.polyval(np.polyfit(frame[feature], frame.PE, 1), x_vals)
        fig.add_scatter(x=x_vals, y=trend, mode="lines", name="Trend", line=dict(color=AMBER, width=3))
    return fig


def gauge(output, height=320):
    mean = float(df.PE.mean())
    low = float(df.PE.min())
    high = float(df.PE.max())
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=output,
            number={"suffix": " MW", "valueformat": ".2f", "font": {"size": 40, "color": TEXT}},
            delta={
                "reference": mean,
                "valueformat": ".2f",
                "suffix": " MW",
                "increasing": {"color": TEAL},
                "decreasing": {"color": AMBER},
            },
            gauge={
                "axis": {"range": [min(low, output), max(high, output)], "tickfont": {"color": MUTED}, "nticks": 5},
                "bar": {"color": TEAL, "thickness": 0.26},
                "bgcolor": "rgba(255,255,255,0.46)",
                "borderwidth": 0,
                "threshold": {"value": mean, "line": {"color": AMBER, "width": 4}},
                "steps": [
                    {"range": [low, mean], "color": "rgba(127,215,242,0.18)"},
                    {"range": [mean, high], "color": "rgba(71,197,190,0.14)"},
                ],
            },
        )
    )
    chart(fig, height)
    st.caption(f"Historical mean: {mean:.2f} MW. Observed range: {low:.2f} to {high:.2f} MW.")


def sensor_tiles(values):
    units = {"AT": "°C", "V": "cm Hg", "AP": "mbar", "RH": "%"}
    names = {"AT": "Temperature", "V": "Exhaust vacuum", "AP": "Ambient pressure", "RH": "Relative humidity"}
    cards = []
    for feature in FEATURES:
        cards.append(
            f'<div class="sensor-card"><b>{feature}</b><small>{names[feature]}</small>'
            f'<strong>{values[feature]:.2f}</strong><small>{units[feature]}</small></div>'
        )
    st.markdown(f'<div class="sensor-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="soft-note">{text}</div>', unsafe_allow_html=True)


def hero(section, title, subtitle):
    st.markdown(
        f"""
<div class="hero">
    <div class="kicker">ENERGY SYSTEMS / {section.upper()}</div>
    <h1>{title}</h1>
    <p>{subtitle}</p>
    <div class="hero-pills">
        <span class="hero-pill"><b>{len(df):,}</b> rows</span>
        <span class="hero-pill"><b>{metrics['model']}</b> best model</span>
        <span class="hero-pill"><b>{metrics['R2']:.4f}</b> test R²</span>
        <span class="hero-pill"><b>{metrics['RMSE']:.2f} MW</b> RMSE</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def plant_svg():
    return """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 760 290'>
    <defs>
      <linearGradient id='g' x1='0' x2='1'><stop offset='0%' stop-color='#DDECF5'/><stop offset='100%' stop-color='#F7FBFF'/></linearGradient>
      <marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0 0L10 5 0 10Z' fill='#59C1BC'/></marker>
      <marker id='h' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0 0L10 5 0 10Z' fill='#E9B96E'/></marker>
    </defs>
    <rect width='760' height='290' rx='26' fill='rgba(255,255,255,0.18)'/>
    <g fill='url(#g)' stroke='#BFD5E4' stroke-width='1.3'>
      <rect x='145' y='88' width='122' height='54' rx='18'/><rect x='322' y='80' width='108' height='66' rx='18'/>
      <rect x='488' y='88' width='112' height='54' rx='18'/><rect x='631' y='88' width='92' height='54' rx='18'/>
      <rect x='180' y='192' width='96' height='44' rx='15'/><rect x='545' y='192' width='106' height='44' rx='15'/>
    </g>
    <g stroke='#59C1BC' stroke-width='2.5' fill='none' marker-end='url(#a)'>
      <path d='M34 115H145'/><path d='M267 115H322'/><path d='M430 115H488'/><path d='M600 115H631'/><path d='M544 142V192'/><path d='M228 142V192'/><path d='M598 236H410V142'/><path d='M228 236V265H718V142'/>
    </g>
    <path d='M267 96H322' stroke='#E9B96E' stroke-width='2.4' stroke-dasharray='8 5' fill='none' marker-end='url(#h)'/>
    <g fill='#35546A' font-family='Arial' text-anchor='middle'>
      <text x='206' y='111' font-size='14' font-weight='bold'>Gas turbine</text><text x='206' y='127' font-size='10'>Air and fuel</text>
      <text x='376' y='110' font-size='14' font-weight='bold'>Heat recovery</text><text x='376' y='126' font-size='10'>Uses hot exhaust</text>
      <text x='544' y='111' font-size='14' font-weight='bold'>Steam turbine</text><text x='544' y='127' font-size='10'>Extra generation</text>
      <text x='677' y='111' font-size='14' font-weight='bold'>Net output</text><text x='677' y='127' font-size='10'>Electricity</text>
      <text x='228' y='219' font-size='13' font-weight='bold'>Generator</text><text x='598' y='219' font-size='13' font-weight='bold'>Condenser</text>
    </g>
    <text x='22' y='25' fill='#6F8A9C' font-family='Arial' font-size='12' letter-spacing='2'>HOW THE PLANT WORKS</text>
    </svg>"""


def watermark_svg(kind):
    icons = {
        "plant": "<path d='M55 215h190v18H55z'/><path d='M82 210c-8-43 5-72 22-105 12 31 24 63 18 105z'/><path d='M134 210c-8-43 5-72 22-105 12 31 24 63 18 105z'/><rect x='184' y='96' width='24' height='114' rx='9'/><path d='M74 112c8-6 20-6 28 0M126 92c8-6 20-6 28 0'/>",
        "temperature": "<path d='M128 64a18 18 0 0 1 36 0v92a36 36 0 1 1-36 0z'/><line x1='146' y1='78' x2='146' y2='165'/><circle cx='146' cy='192' r='18'/>",
        "vacuum": "<circle cx='150' cy='148' r='68'/><path d='M150 148l46-24'/><path d='M92 204a82 82 0 1 0 116 0'/><circle cx='150' cy='148' r='6'/>",
        "pressure": "<circle cx='150' cy='150' r='70'/><path d='M150 150l52-10'/><path d='M104 164h92'/><path d='M150 98v18'/>",
        "humidity": "<path d='M150 76c22 34 52 62 52 96a52 52 0 0 1-104 0c0-34 30-62 52-96z'/><path d='M129 164c6 14 18 22 35 25' />",
        "performance": "<polyline points='72,182 118,148 150,164 198,110 228,124'/><line x1='70' y1='202' x2='236' y2='202'/><line x1='70' y1='92' x2='70' y2='202'/>",
        "chat": "<path d='M78 98h144a22 22 0 0 1 22 22v56a22 22 0 0 1-22 22h-88l-38 28 10-28H78a22 22 0 0 1-22-22v-56a22 22 0 0 1 22-22z'/><circle cx='116' cy='148' r='8'/><circle cx='150' cy='148' r='8'/><circle cx='184' cy='148' r='8'/>",
    }
    path = icons[kind]
    return f"""<div class='watermark'><svg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'>
    <g fill='none' stroke='#8FB0C4' stroke-width='10' stroke-linecap='round' stroke-linejoin='round'>{path}</g>
    </svg></div>"""


def floating_chat_icon():
    ask_url = f"?page={quote('Ask the Data')}"
    st.markdown(
        f"<a class='fab-chat' href='{ask_url}' title='Open assistant'>💬<span>Open assistant</span></a>",
        unsafe_allow_html=True,
    )


page_icons = {
    "Overview": "plant",
    "Predict": "plant",
    "Model Performance": "performance",
    "Ask the Data": "chat",
}

try:
    explore_default = st.session_state.get("explore_feature", "AT")
except Exception:
    explore_default = "AT"
explore_icons = {"AT": "temperature", "V": "vacuum", "AP": "pressure", "RH": "humidity"}
if page == "Explore":
    current_feature = st.session_state.get("explore_feature", explore_default)
    st.markdown(watermark_svg(explore_icons.get(current_feature, "temperature")), unsafe_allow_html=True)
else:
    st.markdown(watermark_svg(page_icons.get(page, "plant")), unsafe_allow_html=True)

floating_chat_icon()

if page == "Overview":
    hero("Overview", "Plant Performance Lab", "Predicting hourly electrical power output from ambient conditions in a combined cycle power plant.")
    for col, (label, value) in zip(
        st.columns(5),
        [
            ("Historical rows", f"{len(df):,}"),
            ("Average output", f"{df.PE.mean():.2f} MW"),
            ("Best model", metrics["model"]),
            ("Test R²", f"{metrics['R2']:.4f}"),
            ("Test RMSE", f"{metrics['RMSE']:.2f} MW"),
        ],
    ):
        col.metric(label, value)

    left, right = st.columns([1.45, 1])
    with left:
        with st.container(border=True):
            st.markdown("### How to read the project")
            st.caption("This view is designed for a non-technical audience. It explains what goes in and what comes out.")
            st.markdown(plant_svg(), unsafe_allow_html=True)
            medians = {feature: float(df[feature].median()) for feature in FEATURES}
            sensor_tiles(medians)
            st.caption("Median readings from the dataset. These are reference values, not live plant readings.")
    with right:
        with st.container(border=True):
            st.markdown("### Output estimate at median conditions")
            gauge(predict({feature: float(df[feature].median()) for feature in FEATURES}, df, model), 280)
            if st.button("Open simulator", use_container_width=True):
                st.session_state.workspace = "Predict"
                set_query_page("Predict")
                st.rerun()
            st.markdown(
                f"""
<div class='mini-grid'>
  <div class='mini-card'><small>Average absolute error</small><strong>{metrics['MAE']:.2f} MW</strong></div>
  <div class='mini-card'><small>Observed output range</small><strong>{df.PE.min():.2f} to {df.PE.max():.2f}</strong></div>
  <div class='mini-card'><small>Top driver</small><strong>Temperature</strong></div>
</div>
""",
                unsafe_allow_html=True,
            )

    low_left, low_right = st.columns([1.3, 1])
    with low_left:
        with st.container(border=True):
            st.markdown("### Temperature and output")
            chart(scatter_with_trend(df, "AT"), 335)
    with low_right:
        with st.container(border=True):
            st.markdown("### Output distribution")
            hist = px.histogram(df, x="PE", nbins=45, labels=LABELS, color_discrete_sequence=[BLUE])
            hist.add_vline(x=df.PE.mean(), line_dash="dash", line_color=AMBER)
            chart(hist, 335)
    note("<strong>What this means:</strong> the model helps compare expected and observed generation under certain weather conditions. It supports explanation, not control.")

elif page == "Explore":
    hero("Explore", "Explore the plant data", "Filter the historical records, compare patterns and see how each input relates to hourly output.")
    with st.container(border=True):
        options = {"AT": "Temperature", "V": "Exhaust vacuum", "AP": "Ambient pressure", "RH": "Relative humidity"}
        selected = st.selectbox("Choose a feature", FEATURES, index=FEATURES.index(st.session_state.get("explore_feature", "AT")), format_func=lambda feature: LABELS[feature], key="explore_feature")
        filtered = df.copy()
        for col, feature in zip(st.columns(4), FEATURES):
            minimum = float(df[feature].min())
            maximum = float(df[feature].max())
            low, high = col.slider(LABELS[feature], minimum, maximum, (minimum, maximum), key=f"filter_{feature}")
            filtered = filtered[filtered[feature].between(low, high)]
        st.metric("Matching rows", f"{len(filtered):,}")
        st.caption(f"Currently focused on {options[selected].lower()}. The faint background icon changes to match it.")
    if filtered.empty:
        st.info("No observations match the current filters. Widen the filters and try again.")
    else:
        a, b = st.columns([1.35, 1])
        with a:
            with st.container(border=True):
                st.markdown(f"### {LABELS[selected]} vs output")
                chart(scatter_with_trend(filtered, selected), 340)
        with b:
            with st.container(border=True):
                st.markdown(f"### Distribution of {LABELS[selected]}")
                dist = px.histogram(filtered, x=selected, labels=LABELS, nbins=35, color_discrete_sequence=[CYAN])
                chart(dist, 340)
        with st.container(border=True):
            st.markdown("### Correlation matrix")
            corr = px.imshow(
                filtered.corr(numeric_only=True),
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                color_continuous_scale=["#DDEAF3", "#F9FCFE", "#95D9EA", "#6FCBC4"],
                aspect="auto",
            )
            chart(corr, 430)
            st.caption("Correlation describes association, not cause and effect.")

elif page == "Predict":
    hero("Predict", "Power output simulator", "Move the sliders to estimate the plant's hourly electrical output under different ambient conditions.")
    controls, display = st.columns([1, 1.4])
    with controls:
        with st.container(border=True):
            st.markdown("### Scenario inputs")
            st.caption("The sliders stay within the real ranges observed in the dataset.")
            values = {}
            for feature in FEATURES:
                values[feature] = st.slider(
                    LABELS[feature],
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].median()),
                    step=0.01,
                    key=f"predict_{feature}",
                )
    output = predict(values, df, model)
    with display:
        with st.container(border=True):
            st.markdown("### Estimated electrical output")
            gauge(output, 330)
            sensor_tiles(values)

    with st.container(border=True):
        st.markdown("### Quick comparison: if temperature rises by 5°C")
        if values["AT"] + 5 <= df.AT.max():
            warmer = predict({**values, "AT": values["AT"] + 5}, df, model)
            c1, c2, c3 = st.columns(3)
            c1.metric("Current estimate", f"{output:.2f} MW")
            c2.metric(f"At {values['AT'] + 5:.2f}°C", f"{warmer:.2f} MW")
            c3.metric("Estimated change", f"{warmer - output:+.2f} MW")
            bar = px.bar(
                x=["Current", "+5°C"],
                y=[output, warmer],
                labels={"x": "Scenario", "y": "Estimated output (MW)"},
                color=["Current", "Warmer"],
                color_discrete_sequence=[TEAL, AMBER],
            )
            chart(bar, 250)
        else:
            st.info("A 5°C increase would go beyond the observed temperature range, so no estimate is shown.")
        st.caption("Only temperature changes in this comparison. The other three readings stay fixed.")

    with st.container(border=True):
        st.markdown("### Temperature response profile")
        temperatures = np.linspace(float(df.AT.min()), float(df.AT.max()), 75)
        scenarios = pd.DataFrame([{**values, "AT": value} for value in temperatures], columns=FEATURES)
        estimates = model.predict(scenarios)
        fig = go.Figure()
        fig.add_scatter(x=temperatures, y=estimates, mode="lines", name="Estimated output", line={"color": TEAL, "width": 3}, fill="tozeroy", fillcolor="rgba(71,197,190,0.10)")
        fig.add_scatter(x=[values["AT"]], y=[output], mode="markers", name="Selected scenario", marker={"size": 12, "color": AMBER, "line": {"width": 2, "color": WHITE}})
        fig.update_layout(xaxis_title="Ambient temperature (°C)", yaxis_title="Estimated output (MW)")
        chart(fig, 340)
        note("This is a model sensitivity view. It is not a full thermodynamic simulation of the plant.")

elif page == "Model Performance":
    hero("Model performance", "How well the model works", "Compare the four candidate models and inspect how close the final predictions are to the measured output.")
    for col, (label, value) in zip(
        st.columns(4),
        [
            ("Final model", metrics["model"]),
            ("Test R²", f"{metrics['R2']:.4f}"),
            ("Test RMSE", f"{metrics['RMSE']:.2f} MW"),
            ("Average absolute error", f"{metrics['MAE']:.2f} MW"),
        ],
    ):
        col.metric(label, value)
    with st.container(border=True):
        st.markdown("### Model comparison")
        st.caption("The winning model was selected by five-fold cross-validation on the training data.")
        compare = px.bar(comparison, x="Model", y=["CV RMSE", "RMSE"], barmode="group", color_discrete_sequence=[CYAN, BLUE], labels={"value": "Error (MW)", "variable": "Metric"})
        chart(compare)
        st.dataframe(comparison.round(4), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Actual vs predicted")
            actual = px.scatter(pairs, x="Actual", y="Predicted", opacity=0.42, color_discrete_sequence=[TEAL])
            low = pairs.Actual.min()
            high = pairs.Actual.max()
            actual.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction", line=dict(color=AMBER))
            chart(actual, 340)
    with right:
        with st.container(border=True):
            st.markdown("### Residuals")
            residuals = px.scatter(pairs, x="Predicted", y="Residual", opacity=0.42, color_discrete_sequence=[BLUE])
            residuals.add_hline(y=0, line_dash="dash", line_color=AMBER)
            chart(residuals, 340)
    with st.container(border=True):
        st.markdown("### Which input matters most?")
        imp = importance.assign(Variable=importance.Feature.map(LABELS))
        perm = px.bar(imp.sort_values("Permutation RMSE increase"), x="Permutation RMSE increase", y="Variable", orientation="h", color_discrete_sequence=[TEAL])
        chart(perm, 340)
        st.caption("Permutation importance shows how much the error rises when one variable is shuffled. It does not prove cause and effect.")

else:
    hero("Ask the data", "Project assistant", "Ask short questions about the dataset, model results or a specific prediction scenario.")
    installed = available_ollama()
    cloud_key = setting("ANTHROPIC_API_KEY")
    engines = ["Offline reference", "Local Ollama", "Anthropic Claude"]
    default_engine = 1 if installed else (2 if cloud_key else 0)
    provider = st.radio("Assistant engine", engines, index=default_engine, horizontal=True)
    local_model = setting("OLLAMA_MODEL", "qwen2.5:3b")
    with st.container(border=True):
        if provider == "Local Ollama":
            local_model = st.text_input("Ollama model", local_model)
            if local_model in installed:
                st.success(f"Connected to Ollama · {local_model}")
            else:
                st.info("Ollama is not available on this machine or server right now. The offline reference still works.")
        elif provider == "Anthropic Claude" and not cloud_key:
            st.info("Add ANTHROPIC_API_KEY to .env or Streamlit Cloud secrets if you want to use Claude online.")
        else:
            st.caption("Offline mode works without any API key and is safe for Streamlit Cloud.")

        if "history" not in st.session_state:
            st.session_state.history = []
        if st.button("Clear chat"):
            st.session_state.history = []

        suggestions = [
            "Which weather variable matters most?",
            "What is the best model?",
            "How does temperature affect output?",
            "Predict AT=20, V=50, AP=1013, RH=70",
        ]
        prompt = None
        for i in range(0, len(suggestions), 2):
            for col, text in zip(st.columns(2), suggestions[i:i + 2]):
                if col.button(text, use_container_width=True):
                    prompt = text

        if not st.session_state.history:
            with st.chat_message("assistant"):
                st.write("Hi. I can explain the plant data, compare the models and estimate output from the four input readings.")

        for item in st.session_state.history:
            with st.chat_message(item["role"]):
                st.write(item["content"])

        typed = st.chat_input("Ask about the plant data...")
        prompt = typed or prompt
        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Checking the project data..."):
                    ctx = context(df, metrics, importance, comparison)
                    try:
                        if provider == "Local Ollama" and local_model in installed:
                            reply = ollama_answer(st.session_state.history, ctx, df, model, local_model)
                        elif provider == "Anthropic Claude" and cloud_key:
                            reply = answer(st.session_state.history, ctx, df, model)
                        else:
                            reply = offline(prompt, ctx, df, model)
                    except Exception:
                        reply = "The selected AI service is unavailable right now. " + offline(prompt, ctx, df, model)
                st.write(reply)
            st.session_state.history.append({"role": "assistant", "content": reply})

st.divider()
st.caption("Plant Performance Lab · UCI Combined Cycle Power Plant dataset · Academic dashboard, not a live plant control interface.")
