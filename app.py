"""
Student Dropout Prediction System
Run with: streamlit run app.py

Expects in the same folder:
  random_forest_model.pkl
  decision_tree_model.pkl
  logistic_regression_model.pkl
  svm_model.pkl
  knn_model.pkl
  naive_bayes_model.pkl
  model_metrics.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict · Dropout Risk",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --ink:     #291C0E;
    --paper:   #E1D4C2;
    --card:    #EDE4D6;
    --stroke:  #BEB5A9;
    --accent:  #6E473B;
    --success: #6E473B;
    --olive:   #6E473B;
    --warning: #291C0E;
    --muted:   #A78D78;
    --sidebar: #BEB5A9;
    --serif:   'DM Serif Display', Georgia, serif;
    --sans:    'DM Sans', system-ui, sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); background: #E1D4C2 !important; color: var(--ink); }

/* Force background on all Streamlit wrappers */
.stApp, .stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], [data-testid="block-container"],
section.main, section.main > div { background: #E1D4C2 !important; }

[data-testid="block-container"] { padding-top: 1rem !important; }

/* Force all text dark unless overridden */
.stApp p, .stApp span, .stApp div, .stApp label { color: #291C0E; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
    box-shadow: none !important;
}

[data-testid="collapsedControl"] {
    background: var(--sidebar) !important;
    border-radius: 0 8px 8px 0 !important;
    border: 1px solid #BEB5A9 !important;
    border-left: none !important;
    top: 3rem !important;
}
[data-testid="collapsedControl"] svg { fill: #6E473B !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid #A78D78 !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #A78D78 !important;
    border: 1px solid #6E473B !important;
    border-radius: 6px !important;
    color: #E1D4C2 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6E473B !important;
    font-weight: 500 !important;
}
.sb-sep   { border: none; border-top: 1px solid #A78D78; margin: 1.2rem 0 0.8rem 0; }
.sb-label {
    font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #6E473B; font-weight: 600; margin-bottom: 0.6rem; display: block;
}

/* Algorithm selector highlight */
.algo-select-wrap {
    background: #c4b49e;
    border: 1.5px solid #6E473B;
    border-radius: 8px;
    padding: 0.7rem 0.8rem 0.4rem;
    margin-bottom: 0.5rem;
}

/* Predict button */
.stButton > button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important; width: 100% !important;
    letter-spacing: 0.02em !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── Main ── */
.page-header { padding: 2rem 0 1.5rem 0; border-bottom: 1.5px solid var(--stroke); margin-bottom: 2.5rem; }
.eyebrow { font-size: 0.68rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }
.page-title { font-family: var(--serif); font-size: 2.8rem; line-height: 1.05; color: var(--ink); margin: 0 0 0.8rem 0; font-weight: 400; }
.page-title em { color: #291C0E; font-style: italic; }
.meta-row { display: flex; gap: 1.8rem; font-size: 0.8rem; color: var(--muted); align-items: center; flex-wrap: wrap; }
.meta-dot { width: 5px; height: 5px; border-radius: 50%; background: #6E473B; display: inline-block; margin-right: 5px; vertical-align: middle; }
.algo-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d4c5b0; border: 1px solid #A78D78;
    color: var(--accent) !important; font-weight: 600;
    font-size: 0.75rem; padding: 3px 10px; border-radius: 20px;
}

.sec-title { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; color: var(--muted); border-bottom: 1px solid var(--stroke); padding-bottom: 0.5rem; margin-bottom: 1.2rem; }

/* Model comparison table */
.model-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.model-table thead tr { border-bottom: 2px solid var(--stroke); }
.model-table thead th {
    padding: 0.5rem 0.8rem; text-align: left; font-size: 0.65rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 600;
}
.model-table thead th:not(:first-child) { text-align: center; }
.model-table tbody tr { border-bottom: 1px solid var(--stroke); transition: background 0.1s; }
.model-table tbody tr:last-child { border-bottom: none; }
.model-table tbody tr:hover { background: #A78D78; color: #E1D4C2; }
.model-table tbody td { padding: 0.6rem 0.8rem; }
.model-table tbody td:not(:first-child) { text-align: center; }

.row-best    { background: #6E473B !important; color: #E1D4C2 !important; }
.row-active  { background: #291C0E !important; color: #E1D4C2 !important; }
.row-both    { background: #291C0E !important; color: #E1D4C2 !important; }

.row-best td, .row-active td, .row-both td { color: #E1D4C2 !important; }
.row-best td *, .row-active td *, .row-both td * { color: #E1D4C2 !important; }
.row-best td span, .row-active td span, .row-both td span { color: #E1D4C2 !important; }

.badge-best   { background: #A78D78; color: #E1D4C2; font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700; padding: 2px 7px; border-radius: 10px; margin-left: 6px; }
.badge-active { background: #6E473B; color: #E1D4C2; font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700; padding: 2px 7px; border-radius: 10px; margin-left: 6px; }

.bar-mini { display: inline-block; background: var(--accent); border-radius: 2px; height: 6px; vertical-align: middle; margin-right: 5px; opacity: 0.7; }

/* Risk factor cards */
.risk-grid  { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-top: 0.5rem; }
.risk-card  { background: var(--card); border: 1px solid var(--stroke); border-radius: 10px; padding: 1.3rem 1.5rem; }
.risk-num   { font-family: var(--serif); font-size: 1.8rem; color: #291C0E; line-height: 1; margin-bottom: 0.4rem; }
.risk-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem; color: #6E473B; }
.risk-desc  { font-size: 0.78rem; color: #291C0E; line-height: 1.55; }

/* Result card */
.result-wrap     { border-radius: 12px; padding: 2rem 2.5rem; margin-bottom: 2rem; }
.result-complete { background: #EDE4D6; border: 1.5px solid #A78D78; }
.result-dropout  { background: #e8d8c8; border: 1.5px solid #6E473B; }

.result-badge   { display: inline-block; font-size: 0.62rem; letter-spacing: 0.16em; text-transform: uppercase; font-weight: 600; padding: 3px 11px; border-radius: 20px; margin-bottom: 0.7rem; }
.badge-complete { background: #BEB5A9; color: #291C0E; }
.badge-dropout  { background: #6E473B; color: #E1D4C2; }

.result-title { font-family: var(--serif); font-size: 2rem; font-weight: 400; line-height: 1.1; margin-bottom: 0.25rem; }
.result-complete .result-title { color: #6E473B; }
.result-dropout  .result-title { color: #291C0E; }
.result-sub   { font-size: 0.84rem; color: var(--muted); margin-bottom: 1.2rem; }

.prob-row { display: flex; align-items: center; gap: 1.8rem; }
.prob-num       { font-family: var(--serif); font-size: 3.2rem; line-height: 1; }
.prob-num-green { color: #6E473B; }
.prob-num-warm  { color: #291C0E; }
.prob-lbl       { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-top: 4px; }

.bar-track { flex: 1; }
.bar-bg    { background: #BEB5A9; border-radius: 4px; height: 7px; overflow: hidden; }
.bar-fill-green { height: 100%; border-radius: 4px; background: #6E473B; }
.bar-fill-warm  { height: 100%; border-radius: 4px; background: #291C0E; }
.bar-labels { display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 5px; }

.algo-note {
    background: #d4c5b0; border: 1px solid #A78D78; border-radius: 6px;
    padding: 0.45rem 0.8rem; font-size: 0.76rem; color: var(--accent);
    margin-bottom: 1rem; display: inline-block;
}

/* Profile table */
.ptable { width: 100%; border-collapse: collapse; }
.ptable tr { border-bottom: 1px solid var(--stroke); }
.ptable tr:last-child { border-bottom: none; }
.ptable td { padding: 0.55rem 0; font-size: 0.84rem; vertical-align: middle; text-align: center; }
.ptable td:first-child { color: var(--muted); width: 50%; }

/* Chart legend */
.chart-legend { background: var(--card); border: 1px solid var(--stroke); border-radius: 8px; padding: 1rem 1.2rem; margin-top: 1rem; font-size: 0.82rem; line-height: 1.6; }
.legend-row   { display: flex; align-items: flex-start; gap: 0.7rem; margin-bottom: 0.6rem; }
.legend-row:last-child { margin-bottom: 0; }
.legend-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; margin-top: 3px; }
.legend-text strong { font-weight: 600; display: block; font-size: 0.8rem; }
.legend-text span   { color: var(--muted); font-size: 0.76rem; }

/* Insight boxes */
.insight-box {
    background: #EDE4D6; border: 1px solid #BEB5A9; border-left: 3px solid #6E473B;
    border-radius: 0 8px 8px 0; padding: 1.1rem 1.4rem; font-size: 0.84rem;
    line-height: 1.65; color: var(--ink); margin-top: 1.5rem;
}
.insight-box strong { color: #6E473B; }
.insight-warn {
    background: #e8d8c8; border: 1px solid #A78D78; border-left: 3px solid #291C0E;
    border-radius: 0 8px 8px 0; padding: 1.1rem 1.4rem; font-size: 0.84rem;
    line-height: 1.65; color: var(--ink); margin-top: 1.5rem;
}
.insight-warn strong { color: #291C0E; }

.idle-prompt { font-size: 0.88rem; color: var(--muted); margin-bottom: 2rem; padding: 0.9rem 1.2rem; background: var(--card); border: 1px solid var(--stroke); border-radius: 8px; display: inline-block; }
</style>
""", unsafe_allow_html=True)


# ── Model & metrics loading ───────────────────────────────────────────────────
MODEL_FILES = {
    'Random Forest':       'random_forest_model.pkl',
    'Decision Tree':       'decision_tree_model.pkl',
    'Logistic Regression': 'logistic_regression_model.pkl',
    'SVM':                 'svm_model.pkl',
    'KNN':                 'knn_model.pkl',
    'Naive Bayes':         'naive_bayes_model.pkl',
}

@st.cache_resource
def load_all_models():
    loaded = {}
    for name, fname in MODEL_FILES.items():
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                loaded[name] = pickle.load(f)
    return loaded

@st.cache_data
def load_metrics():
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json') as f:
            return json.load(f)
    # Fallback — old single-model results if json not yet generated
    return {
        'Random Forest':       {'accuracy':0.8727,'precision':0.8401,'recall':0.9018,'f1':0.8699},
        'Decision Tree':       {'accuracy':0.8567,'precision':0.8240,'recall':0.8950,'f1':0.8563},
        'Logistic Regression': {'accuracy':0.8490,'precision':0.8150,'recall':0.8900,'f1':0.8474},
        'SVM':                 {'accuracy':0.8473,'precision':0.8130,'recall':0.8880,'f1':0.8465},
        'KNN':                 {'accuracy':0.8117,'precision':0.7830,'recall':0.8650,'f1':0.8149},
        'Naive Bayes':         {'accuracy':0.7800,'precision':0.7550,'recall':0.8500,'f1':0.7959},
        '_best': 'Random Forest',
    }

all_models = load_all_models()
metrics    = load_metrics()
best_model_name = metrics.get('_best', 'Random Forest')

# Warn if models aren't retrained yet (old single best_model.pkl setup)
if not all_models:
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            all_models['Random Forest'] = pickle.load(f)
        best_model_name = 'Random Forest'
    else:
        st.error("No model files found. Run the retrain cell in your notebook first.")
        st.stop()


# ── Encoding maps ─────────────────────────────────────────────────────────────
GENDER_MAP = {"Female": 0, "Male": 1}
REGION_MAP = {
    "East Anglian Region": 0,  "East Midlands Region": 1, "Ireland": 2,
    "London Region": 3,        "North Region": 4,          "North Western Region": 5,
    "Scotland": 6,             "South East Region": 7,     "South Region": 8,
    "South West Region": 9,    "Wales": 10,                "West Midlands Region": 11,
    "Yorkshire Region": 12,
}
EDUCATION_MAP = {
    "A Level or Equivalent": 0,  "HE Qualification": 1,
    "Lower Than A Level": 2,     "No Formal Quals": 3,
    "Post Graduate Qualification": 4,
}
IMD_MAP = {
    "0-10%": 0,"10-20%": 1,"20-30%": 2,"30-40%": 3,"40-50%": 4,
    "50-60%": 5,"60-70%": 6,"70-80%": 7,"80-90%": 8,"90-100%": 9,
}
AGE_MAP        = {"0-35": 0, "35-55": 1, "55<=": 2}
DISABILITY_MAP = {"No": 0, "Yes": 1}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem">
        <div style="font-family:'DM Serif Display',serif;font-size:1.25rem;color:#291C0E">EduPredict</div>
        <div style="font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;color:#A78D78;margin-top:3px">Student Risk Analyser</div>
    </div>
    <hr class="sb-sep">
    <span class="sb-label">Algorithm</span>
    """, unsafe_allow_html=True)

    available_models = list(all_models.keys())
    default_idx = available_models.index(best_model_name) if best_model_name in available_models else 0

    selected_algo = st.selectbox(
        "Prediction Model",
        available_models,
        index=default_idx,
        help=(
            "Choose which trained algorithm to use for prediction. "
            f"'{best_model_name}' is the best-performing model based on F1 score. "
            "You can compare all models on the home screen."
        ),
    )

    # Show quick metric for chosen model
    if selected_algo in metrics and selected_algo != '_best':
        m = metrics[selected_algo]
        is_best = (selected_algo == best_model_name)
        tag = " Best" if is_best else ""
        st.markdown(
            f'<div style="font-size:0.72rem;color:#E1D4C2;margin:-4px 0 8px;padding:6px 8px;'
            f'background:#6E473B;border-radius:6px;">'
            f'Accuracy <b>{m["accuracy"]*100:.1f}%</b> &nbsp;·&nbsp; '
            f'F1 <b>{m["f1"]*100:.1f}%</b>{tag}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sb-sep"><span class="sb-label">Demographics</span>', unsafe_allow_html=True)

    gender     = st.selectbox("Gender",     list(GENDER_MAP.keys()))
    age_band   = st.selectbox("Age Group",  list(AGE_MAP.keys()))
    region     = st.selectbox("Region",     list(REGION_MAP.keys()))
    disability = st.selectbox("Disability", list(DISABILITY_MAP.keys()),
                               help="Whether the student has declared a disability.")

    st.markdown('<hr class="sb-sep"><span class="sb-label">Background</span>', unsafe_allow_html=True)
    highest_education = st.selectbox(
        "Highest Education", list(EDUCATION_MAP.keys()),
        help="Highest qualification held before enrolling.",
    )
    imd_band = st.selectbox(
        "Deprivation Band (IMD)", list(IMD_MAP.keys()),
        help="0–10% = most deprived neighbourhood. 90–100% = most affluent.",
    )

    st.markdown('<hr class="sb-sep"><span class="sb-label">Academic History</span>', unsafe_allow_html=True)
    num_of_prev_attempts = st.slider(
        "Previous Course Attempts", 0, 5, 0,
        help="How many times the student previously enrolled in and didn't complete this course. 0 = first attempt.",
    )

    st.markdown('<hr class="sb-sep"><span class="sb-label">Assessment Performance</span>', unsafe_allow_html=True)
    avg_score = st.slider("Average Score (%)", 0, 100, 65,
                           help="Overall average mark across all submitted assessments.")
    submission_count = st.slider(
        "Assessments Submitted", 0, 15, 5,
        help="How many assessments the student handed in. The single most important predictor in the model.",
    )

    st.markdown('<hr class="sb-sep"><span class="sb-label">Registration</span>', unsafe_allow_html=True)
    days_to_start = st.slider(
        "Days Before Course Start", -120, 60, -30,
        help="Negative = registered early. Positive = registered after course started. Earlier is better.",
    )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse Risk Now →")


# ── Build input dataframe (10 features) ──────────────────────────────────────
def build_input_df():
    return pd.DataFrame([{
        "gender":               GENDER_MAP[gender],
        "region":               REGION_MAP[region],
        "highest_education":    EDUCATION_MAP[highest_education],
        "imd_band":             IMD_MAP[imd_band],
        "age_band":             AGE_MAP[age_band],
        "num_of_prev_attempts": num_of_prev_attempts,
        "disability":           DISABILITY_MAP[disability],
        "avg_score":            float(avg_score),
        "submission_count":     float(submission_count),
        "days_to_start":        float(days_to_start),
    }])


# ── Score chart ───────────────────────────────────────────────────────────────
def score_chart():
    cats  = ["Avg\nScore", "Submit\nRate"]
    vals  = [avg_score, min(submission_count / 15 * 100, 100)]
    bench = [60, 60]
    x, w  = np.arange(len(cats)), 0.32

    fig, ax = plt.subplots(figsize=(4.5, 2.4))
    fig.patch.set_facecolor("#EDE4D6"); ax.set_facecolor("#EDE4D6")
    ax.bar(x - w/2, vals,  w, color="#6E473B", alpha=0.9, label="Student",   zorder=3)
    ax.bar(x + w/2, bench, w, color="#BEB5A9", alpha=1.0,  label="Benchmark", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9, color="#291C0E")
    ax.set_ylim(0, 115); ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"], fontsize=8, color="#291C0E")
    ax.yaxis.grid(True, color="#BEB5A9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BEB5A9"); ax.spines["bottom"].set_color("#BEB5A9")
    ax.tick_params(colors="#291C0E", length=0)
    ax.legend(fontsize=8, framealpha=0, labelcolor="#291C0E")
    plt.tight_layout(pad=0.3)
    return fig


# ── Model comparison table HTML ───────────────────────────────────────────────
def model_comparison_html(active_algo, best_algo):
    # Sort by F1 descending
    model_names = [k for k in MODEL_FILES.keys() if k in metrics]
    model_names.sort(key=lambda n: metrics[n]['f1'], reverse=True)

    rows = ""
    for name in model_names:
        m = metrics[name]
        is_best   = (name == best_algo)
        is_active = (name == active_algo)

        if is_best and is_active:
            row_cls = "row-both"
        elif is_best:
            row_cls = "row-best"
        elif is_active:
            row_cls = "row-active"
        else:
            row_cls = ""

        badges = ""
        if is_best:
            badges += '<span class="badge-best">Best</span>'
        if is_active and not is_best:
            badges += '<span class="badge-active">Active</span>'
        if is_best and is_active:
            badges = '<span class="badge-best">Best · Active</span>'

        # Mini bar proportional to accuracy
        bar_w = int(m['accuracy'] * 60)

        def fmt(v):
            return f"{v*100:.1f}%"

        name_color = "color:#E1D4C2" if (is_best or is_active) else ""
        rows += f"""
        <tr class="{row_cls}">
            <td><span style="font-weight:500;{name_color}">{name}</span>{badges}</td>
            <td>
                <span class="bar-mini" style="width:{bar_w}px"></span><span style="{name_color}">{fmt(m['accuracy'])}</span>
            </td>
            <td><span style="{name_color}">{fmt(m['precision'])}</span></td>
            <td><span style="{name_color}">{fmt(m['recall'])}</span></td>
            <td><b><span style="{name_color}">{fmt(m['f1'])}</span></b></td>
        </tr>"""

    return f"""
    <table class="model-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:0.7rem;line-height:1.5">
        <b>Accuracy</b> — overall % correct &nbsp;·&nbsp;
        <b>Precision</b> — of those flagged at-risk, how many truly were &nbsp;·&nbsp;
        <b>Recall</b> — of all at-risk students, how many were caught &nbsp;·&nbsp;
        <b>F1</b> — harmonic mean of precision & recall (best overall balance)
    </div>"""


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="eyebrow">ML-Powered · OULAD Dataset · 10 Features</div>
    <h1 class="page-title">Student <em>Dropout</em><br>Risk Predictor</h1>
    <div class="meta-row">
        <span><span class="meta-dot"></span>Trained on 15000 students</span>
        <span><span class="meta-dot"></span>6 Algorithms Compared</span>
        <span class="algo-badge">{selected_algo}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── IDLE ──────────────────────────────────────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div style="max-width:720px;margin-bottom:2rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#291C0E;margin-bottom:0.8rem;font-weight:400;">
            About this tool
        </div>
        <div style="font-size:0.88rem;color:#291C0E;line-height:1.75;">
            This app uses machine learning to predict whether a student is likely to complete or drop out of their course,
            based on data from the Open University Learning Analytics Dataset (OULAD).
            Six classical ML algorithms were trained on 15,000 student records across 10 features.
        </div>
        <div style="margin-top:1.2rem;font-size:0.88rem;color:#291C0E;line-height:1.75;">
            <strong style="color:#6E473B;">How to use it</strong><br>
            Fill in the student profile fields in the sidebar on the left — demographics, academic background,
            assessment performance, and registration timing. Once you click <strong>Analyse Risk Now</strong>,
            the model will output a completion or dropout prediction along with a probability score and a breakdown of key risk factors.
        </div>
        <div style="margin-top:1.2rem;font-size:0.88rem;color:#291C0E;line-height:1.75;">
            <strong style="color:#6E473B;">Interpreting the result</strong><br>
            A <strong>Likely to Complete</strong> result means the student's profile closely matches students who finished their course.
            A <strong>Dropout Risk Detected</strong> result flags the student for early intervention — the app will highlight the
            most significant risk factor and suggest a course of action.
            You can also switch between the six trained algorithms using the selector at the top of the sidebar.
        </div>
    </div>
    <div class="idle-prompt">
        ← Fill in the student profile on the left, then click <strong>Analyse Risk Now</strong>
    </div>""", unsafe_allow_html=True)

    
    st.markdown('<div class="sec-title">All Model Performance</div>', unsafe_allow_html=True)
    st.markdown(model_comparison_html(selected_algo, best_model_name), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Top risk factors
    st.markdown('<div class="sec-title">Top Dropout Risk Factors</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="risk-grid">
        <div class="risk-card">
            <div class="risk-num">#1</div>
            <div class="risk-title">Assessment Submissions</div>
            <div class="risk-desc">Submitting consistently — even with modest scores — is the single strongest predictor of course completion.</div>
        </div>
        <div class="risk-card">
            <div class="risk-num">#2</div>
            <div class="risk-title">Average Score</div>
            <div class="risk-desc">Overall performance is the second strongest signal. Students scoring below 50% face significantly higher risk.</div>
        </div>
        <div class="risk-card">
            <div class="risk-num">#3</div>
            <div class="risk-title">Early Registration</div>
            <div class="risk-desc">Students who enrol well before the course start date show higher engagement and are more likely to complete.</div>
        </div>
    </div>""", unsafe_allow_html=True)



else:
    model = all_models.get(selected_algo)
    if model is None:
        st.error(f"Model '{selected_algo}' not loaded. Run the retrain cell in your notebook.")
        st.stop()

    input_df      = build_input_df()
    prediction    = model.predict(input_df)[0]
    prob_complete = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else float(prediction)
    prob_dropout  = 1.0 - prob_complete
    is_complete   = prediction == 1

    # Active model note
    is_best_active = (selected_algo == best_model_name)
    note = "Best-performing model" if is_best_active else f"Switch to <b>{best_model_name}</b> in the sidebar for the highest-accuracy prediction"
    st.markdown(f'<div class="algo-note">Using: <b>{selected_algo}</b> &nbsp;·&nbsp; {note}</div>', unsafe_allow_html=True)

    # Result banner
    if is_complete:
        st.markdown(f"""
        <div class="result-wrap result-complete">
            <div class="result-badge badge-complete">Low Risk</div>
            <div class="result-title">Likely to Complete</div>
            <div class="result-sub">This student's profile closely aligns with successful course completers in the training data.</div>
            <div class="prob-row">
                <div>
                    <div class="prob-num prob-num-green">{prob_complete*100:.0f}%</div>
                    <div class="prob-lbl">Completion probability</div>
                </div>
                <div class="bar-track">
                    <div class="bar-bg"><div class="bar-fill-green" style="width:{prob_complete*100:.1f}%"></div></div>
                    <div class="bar-labels"><span>0%</span><span>100%</span></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-wrap result-dropout">
            <div class="result-badge badge-dropout">At Risk</div>
            <div class="result-title">Dropout Risk Detected</div>
            <div class="result-sub">This student may benefit from early academic support and engagement check-ins.</div>
            <div class="prob-row">
                <div>
                    <div class="prob-num prob-num-warm">{prob_dropout*100:.0f}%</div>
                    <div class="prob-lbl">Dropout probability</div>
                </div>
                <div class="bar-track">
                    <div class="bar-bg"><div class="bar-fill-warm" style="width:{prob_dropout*100:.1f}%"></div></div>
                    <div class="bar-labels"><span>0%</span><span>100%</span></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Two-column detail
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="sec-title">Score vs Benchmark</div>', unsafe_allow_html=True)
        st.pyplot(score_chart(), use_container_width=True); plt.close()

        submit_rate = min(int(submission_count / 15 * 100), 100)
        st.markdown(f"""
        <div class="chart-legend">
            <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.7rem">How to read this chart</div>
            <div class="legend-row">
                <div class="legend-swatch" style="background:#6E473B"></div>
                <div class="legend-text">
                    <strong>Brown — This student</strong>
                    <span>Avg Score: {avg_score}% &nbsp;·&nbsp; Submit Rate: {submission_count} submissions → {submit_rate}% of max</span>
                </div>
            </div>
            <div class="legend-row">
                <div class="legend-swatch" style="background:#BEB5A9"></div>
                <div class="legend-text">
                    <strong>Stone — Benchmark (60)</strong>
                    <span>A 60% reference line representing an average passing student.</span>
                </div>
            </div>
            <div style="border-top:1px solid var(--stroke);margin:0.6rem 0 0.4rem"></div>
            <div style="font-size:0.74rem;color:var(--muted)">
                Bars above 60 are above average. Both metrics matter: consistently submitting work — even at modest scores — is a stronger predictor than score alone.
            </div>
        </div>""", unsafe_allow_html=True)

        # Model comparison mini-table on results page too
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">All Model Performance</div>', unsafe_allow_html=True)
        st.markdown(model_comparison_html(selected_algo, best_model_name), unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-title">Student Profile Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="ptable">
            <tr><td>Gender</td>               <td>{gender}</td></tr>
            <tr><td>Age Group</td>            <td>{age_band}</td></tr>
            <tr><td>Region</td>               <td>{region}</td></tr>
            <tr><td>Education</td>            <td>{highest_education}</td></tr>
            <tr><td>IMD Band</td>             <td>{imd_band}</td></tr>
            <tr><td>Disability</td>           <td>{disability}</td></tr>
            <tr><td>Previous Attempts</td>    <td>{num_of_prev_attempts}</td></tr>
            <tr><td>Average Score</td>        <td>{avg_score}%</td></tr>
            <tr><td>Assessments Submitted</td><td>{submission_count}</td></tr>
            <tr><td>Days to Start</td>        <td>{days_to_start:+d}</td></tr>
        </table>""", unsafe_allow_html=True)

        if is_complete:
            strength = "submission count" if submission_count >= 8 else "assessment performance"
            st.markdown(f"""
            <div class="insight-box">
                <strong>Key Insight —</strong>
                This student's <strong>{strength}</strong> is the strongest positive signal.
                With {submission_count} submissions and an average of {avg_score}%, this profile
                closely matches students who completed their courses.
            </div>""", unsafe_allow_html=True)
        else:
            weakest = (
                "low submission rate" if submission_count < 5
                else "average score" if avg_score < 50
                else "history of previous attempts"
            )
            st.markdown(f"""
            <div class="insight-warn">
                <strong>Suggested Action —</strong>
                The most significant risk factor is this student's <strong>{weakest}</strong>.
                Early outreach, assignment reminders, and peer mentoring have been shown to
                improve retention for this profile. Consider flagging for pastoral support.
            </div>""", unsafe_allow_html=True)
