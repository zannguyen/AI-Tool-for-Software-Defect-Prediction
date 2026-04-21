"""DefectSight — Streamlit frontend (2025 best-practices edition).

Tabs
────
  🏠 Workspace   – import files, code viewer, file explorer
  📊 Dashboard   – workspace-level KPIs + risk charts
  🤖 Training    – load NASA dataset → preprocess → train → leaderboard
  🔍 Evaluation  – confusion matrix, ROC/PR curves, model comparison radar
  🎯 Prediction  – SHAP waterfall + LIME for any selected module
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "backend"))

from backend.api import (           # noqa: E402
    build_entries_from_files,
    build_entries_from_zip,
    get_risk_label,
    render_file_code,
    run_analysis,
)
from ui.state import ensure_state   # noqa: E402
from ui.styles import apply_global_styles  # noqa: E402
from ui.tree import build_tree, render_tree  # noqa: E402

# ── page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="DefectSight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── design tokens ──────────────────────────────────────────────────────────
BRAND   = "#2f9d8a"
DANGER  = "#b8323d"
WARN    = "#e2a44c"
SAFE    = "#2e6f44"
PANEL   = "#171d24"
BG      = "#0f1318"
INK     = "#e6edf6"
MUTED   = "#9aabbe"
LINE    = "#2c3643"

# ─────────────────────────────────────────────────────────────────────────────
# CACHING — heavy operations never run twice for the same input
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_run_analysis(entries_key: str, entries: list[dict]):
    """Cache full analysis results keyed by a hash of file names."""
    return run_analysis(entries)


@st.cache_data(show_spinner=False)
def _cached_load_dataset(filepath: str):
    """Cache raw dataset load from CSV/ARFF."""
    import preprocessing as pp_mod
    return pp_mod.load_data(filepath)


@st.cache_resource(show_spinner=False)
def _cached_preprocessor(filepath: str, strategy: str):
    """Cache fitted SDPPreprocessor — expensive; only refit on param change."""
    import preprocessing as pp_mod
    df = pp_mod.load_data(filepath)
    pp = pp_mod.SDPPreprocessor(imbalance_strategy=strategy)
    X_tr, X_te, y_tr, y_te = pp.fit_transform(df)
    return pp, X_tr, X_te, y_tr, y_te


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA CSS — professional polish on top of styles.py base
# ─────────────────────────────────────────────────────────────────────────────

_EXTRA_CSS = f"""
<style>
/* ── tab strip ──────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {{
    font-family: 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 8px 20px;
    color: {MUTED};
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {BRAND} !important;
    border-bottom: 2px solid {BRAND} !important;
}}
[data-testid="stTabs"] > div:first-child {{
    border-bottom: 1px solid {LINE};
    margin-bottom: 0.6rem;
}}

/* ── metric cards ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 12px;
    padding: 14px 18px;
    transition: box-shadow 0.2s;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 0 0 1px {BRAND}44;
}}
[data-testid="stMetricLabel"] > div {{
    color: {MUTED} !important;
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] > div {{
    color: {INK} !important;
    font-size: 1.45rem !important;
    font-weight: 700;
}}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.78rem !important;
}}

/* ── file uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {PANEL};
    border: 1.5px dashed {LINE};
    border-radius: 10px;
    padding: 8px;
    transition: border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {BRAND};
}}

/* ── primary buttons ────────────────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, {BRAND}, #23806f);
    border: none;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.03em;
    transition: opacity 0.18s, box-shadow 0.18s;
}}
[data-testid="stButton"] > button[kind="primary"]:hover {{
    opacity: 0.88;
    box-shadow: 0 4px 18px {BRAND}55;
}}

/* ── secondary buttons ──────────────────────────────────────────────────── */
[data-testid="stButton"] > button:not([kind="primary"]) {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 8px;
    color: {INK};
    transition: border-color 0.18s;
}}
[data-testid="stButton"] > button:not([kind="primary"]):hover {{
    border-color: {BRAND};
    color: {BRAND};
}}

/* ── spinner / status ───────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 10px;
}}

/* ── dataframe / leaderboard ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {LINE};
    border-radius: 8px;
    overflow: hidden;
}}

/* ── selectbox ──────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {{
    background: {PANEL};
    border-color: {LINE};
    color: {INK};
}}

/* ── expander ───────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] > summary {{
    font-size: 0.84rem;
    font-weight: 600;
    color: {MUTED};
}}

/* ── risk badge pill ────────────────────────────────────────────────────── */
.risk-badge-high {{ background:{DANGER}22; color:{DANGER}; border:1px solid {DANGER}66;
    border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:700; }}
.risk-badge-med  {{ background:{WARN}22; color:{WARN}; border:1px solid {WARN}66;
    border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:700; }}
.risk-badge-low  {{ background:{SAFE}22; color:{SAFE}; border:1px solid {SAFE}66;
    border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:700; }}

/* ── section header ─────────────────────────────────────────────────────── */
.ds-section-header {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {MUTED};
    font-weight: 700;
    margin-bottom: 10px;
    border-bottom: 1px solid {LINE};
    padding-bottom: 5px;
}}
</style>
"""


def _apply_extra_css():
    st.markdown(_EXTRA_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_risk(score: float) -> str:
    label, _ = get_risk_label(score)
    return label


def _risk_badge(score: float) -> str:
    label = _fmt_risk(score).lower()
    cls = f"risk-badge-{label}"
    return f"<span class='{cls}'>{label.upper()} {score*100:.0f}%</span>"


def _plotly_base() -> dict:
    """Shared Plotly layout kwargs so all charts match the dark theme.

    Note: `margin` is intentionally excluded — each call-site passes its own
    margin so there is no duplicate-keyword conflict.
    """
    return dict(
        plot_bgcolor=PANEL,
        paper_bgcolor=BG,
        font=dict(color=INK, family="Manrope, sans-serif", size=12),
    )


def _model_color(name: str) -> str:
    palette = {
        "Random Forest":        "#2f9d8a",
        "XGBoost":              "#e2a44c",
        "LightGBM":             "#79c0ff",
        "Logistic Regression":  "#9aabbe",
        "Neural Network (MLP)": "#c586c0",
        "Stacking Ensemble":    "#b8323d",
        "Voting Ensemble (Soft)": "#7eddaf",
        "Gradient Boosting":    "#f4a261",
    }
    return palette.get(name, "#7ec6ff")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — WORKSPACE
# ─────────────────────────────────────────────────────────────────────────────

def _tab_workspace() -> None:
    left, center, right = st.columns([1.25, 3.0, 1.55], gap="small")

    # ── LEFT: import / explorer ────────────────────────────────────────────
    with left:
        with st.container(height=820, border=False):
            files   = st.session_state.files
            analyzed = st.session_state.analyzed

            if not analyzed:
                # ── import panel ───────────────────────────────────────────
                st.markdown('<p class="ds-section-header">Import Workspace</p>',
                            unsafe_allow_html=True)

                st.file_uploader(
                    "Source files",
                    accept_multiple_files=True,
                    key="files_input",
                    help="Supported: .py .java .js .ts .c .cpp .cs .go .rs .rb .php .swift .kt .html .css .sql",
                    type=["py","java","js","jsx","ts","tsx","c","cpp","cs","h","hpp",
                          "go","rs","rb","php","swift","kt","html","css","sql"],
                )
                st.file_uploader("ZIP project", type=["zip"], key="zip_input")
                st.caption("💡 ZIP upload preserves folder structure.")

                st.divider()
                if st.button("▶ Run Analysis", type="primary", use_container_width=True):
                    _do_analysis()

            else:
                # ── explorer panel ─────────────────────────────────────────
                st.markdown(f'<p class="ds-section-header">Explorer — {len(files)} files</p>',
                            unsafe_allow_html=True)

                c1, c2 = st.columns([2, 1])
                with c1:
                    st.text_input("🔍 Search", placeholder="filename…", key="tree_query",
                                  label_visibility="collapsed")
                with c2:
                    st.selectbox("Risk", ["All","High","Medium","Low"],
                                 key="tree_risk_filter", label_visibility="collapsed")

                tree_zone = st.container(height=640, border=False)
                with tree_zone:
                    render_tree(
                        build_tree(files),
                        active_file=st.session_state.active_file,
                        analyzed=analyzed,
                        query=st.session_state.tree_query,
                        risk_filter=st.session_state.tree_risk_filter,
                    )
                st.divider()
                if st.button("↺ New Import", use_container_width=True):
                    from ui.state import reset_analysis_state
                    reset_analysis_state()
                    st.rerun()

    # ── CENTER: code viewer ────────────────────────────────────────────────
    with center:
        files    = st.session_state.files
        analyzed = st.session_state.analyzed

        if not files:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:60vh;gap:12px;opacity:.55">
              <p style="font-size:3rem;margin:0">🔍</p>
              <p style="font-size:1.1rem;font-weight:700;margin:0">DefectSight Workbench</p>
              <p style="font-size:.86rem;margin:0">Upload source files and run analysis to get started.</p>
            </div>""", unsafe_allow_html=True)
            return

        # resolve active file
        names = [e["name"] for e in files]
        if st.session_state.active_file not in names:
            st.session_state.active_file = files[0]["name"]
        active = next(e for e in files if e["name"] == st.session_state.active_file)

        # ── file header ────────────────────────────────────────────────────
        risk_html = _risk_badge(active["risk_score"]) if analyzed else ""
        st.markdown(
            f"""<div style="background:{PANEL};border-radius:10px;padding:12px 16px;
                           border:1px solid {LINE};margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:700;font-size:.92rem">{active['lang'][0]} {active['name']}</span>
                    {risk_html}
                  </div>
                  <span style="font-size:.78rem;color:{MUTED}">{active['lang'][2]}</span>
                </div>""",
            unsafe_allow_html=True,
        )

        # ── metrics row ────────────────────────────────────────────────────
        if analyzed:
            m = active.get("metrics", {})
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("LOC", m.get("loc", 0))
            k2.metric("Cyclomatic", m.get("cc", 0))
            k3.metric("Functions", m.get("funcs", 0))
            k4.metric("Risk Score", f"{active['risk_score']*100:.1f}%")

        # ── code viewer ────────────────────────────────────────────────────
        code_html = render_file_code(active, analyzed)
        st.markdown(f'<div class="ds-code-box">{code_html}</div>',
                    unsafe_allow_html=True)

    # ── RIGHT: insights panel ──────────────────────────────────────────────
    with right:
        with st.container(height=820, border=False):
            analysis = st.session_state.analysis
            analyzed = st.session_state.analyzed
            files    = st.session_state.files

            st.markdown('<p class="ds-section-header">Insights</p>',
                        unsafe_allow_html=True)

            if not analyzed or not analysis:
                st.caption("Run analysis to see model metrics & risk insights.")
                return

            # KPIs
            avg_risk = analysis["avg_risk"] * 100
            st.metric("Workspace Risk", f"{avg_risk:.1f}%",
                      delta=f"{'⚠ Above' if avg_risk > 30 else '✓ Below'} 30% threshold",
                      delta_color="inverse")
            cols = st.columns(3)
            cols[0].metric("🔴 High",  analysis["high"])
            cols[1].metric("🟡 Med",   analysis["med"])
            cols[2].metric("🟢 Low",   analysis["low"])

            # mini risk donut
            fig = go.Figure(go.Pie(
                labels=["High","Medium","Low"],
                values=[analysis["high"], analysis["med"], analysis["low"]],
                hole=0.62,
                marker_colors=[DANGER, WARN, SAFE],
                textinfo="none",
                hovertemplate="%{label}: %{value} files<extra></extra>",
            ))
            fig.update_layout(
                **_plotly_base(),
                height=160,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # model progress bars
            st.markdown('<p class="ds-section-header">Model Performance</p>',
                        unsafe_allow_html=True)
            models = analysis.get("models", {})
            sel = st.selectbox("Model", ["All Models"] + list(models.keys()),
                               key="model_choice_workspace")
            show = models if sel == "All Models" else {sel: models.get(sel, {})}

            for mname, mmet in show.items():
                if not mmet:
                    continue
                with st.expander(mname, expanded=(sel != "All Models")):
                    st.progress(float(mmet["accuracy"]),
                                text=f"Accuracy {mmet['accuracy']*100:.1f}%")
                    st.progress(float(mmet["auc"]),
                                text=f"ROC-AUC  {mmet['auc']*100:.1f}%")
                    st.progress(float(mmet["f1"]),
                                text=f"F1-Score {mmet['f1']*100:.1f}%")

            st.divider()

            # top risky files
            st.markdown('<p class="ds-section-header">Top Risky Files</p>',
                        unsafe_allow_html=True)
            risky = sorted(files, key=lambda e: e.get("risk_score", 0), reverse=True)[:8]
            for entry in risky:
                badge = _fmt_risk(entry["risk_score"]).upper()
                label = f"{entry['lang'][0]} {entry.get('fname', entry['name'])} · {entry['risk_score']*100:.0f}%"
                if st.button(label, key=f"rz_{entry['name']}", use_container_width=True):
                    st.session_state.active_file = entry["name"]
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD (workspace-level analytics)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_dashboard() -> None:
    analysis = st.session_state.analysis
    files    = st.session_state.files
    analyzed = st.session_state.analyzed

    if not analyzed or not analysis or not files:
        st.info("Run analysis in the **Workspace** tab first.")
        return

    # ── KPI row ────────────────────────────────────────────────────────────
    st.markdown('<p class="ds-section-header">Workspace Overview</p>',
                unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Files",  len(files))
    k2.metric("Total LOC",    analysis["total_loc"])
    k3.metric("🔴 High Risk", analysis["high"],
              delta=f"{analysis['high']/max(len(files),1)*100:.0f}% of files",
              delta_color="inverse")
    k4.metric("🟡 Medium",    analysis["med"])
    k5.metric("🟢 Low Risk",  analysis["low"])

    st.divider()

    col_a, col_b = st.columns([1, 1.2])

    # ── Risk distribution bar chart ─────────────────────────────────────────
    with col_a:
        st.markdown('<p class="ds-section-header">Risk Distribution</p>',
                    unsafe_allow_html=True)
        scores  = [e["risk_score"] for e in files]
        buckets = pd.cut(scores,
                         bins=[0, 0.33, 0.67, 1.0],
                         labels=["Low","Medium","High"])
        counts  = buckets.value_counts().reindex(["High","Medium","Low"], fill_value=0)

        fig = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[DANGER, WARN, SAFE],
            text=counts.values,
            textposition="outside",
            textfont=dict(color=INK, size=13),
        ))
        fig.update_layout(**_plotly_base(), height=280,
                          xaxis=dict(color=INK),
                          yaxis=dict(color=INK, showgrid=True,
                                     gridcolor=LINE, title="Files"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Risk score histogram ────────────────────────────────────────────────
    with col_b:
        st.markdown('<p class="ds-section-header">Risk Score Distribution</p>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Histogram(
            x=scores, nbinsx=30,
            marker=dict(
                color=scores,
                colorscale=[[0, SAFE], [0.5, WARN], [1, DANGER]],
                showscale=False,
            ),
        ))
        fig2.update_layout(**_plotly_base(), height=280,
                           xaxis=dict(color=INK, title="Risk Score"),
                           yaxis=dict(color=INK, title="Count",
                                      showgrid=True, gridcolor=LINE))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top 15 risky files bar chart ────────────────────────────────────────
    st.markdown('<p class="ds-section-header">Top 15 Highest-Risk Files</p>',
                unsafe_allow_html=True)
    top15 = sorted(files, key=lambda e: e.get("risk_score", 0), reverse=True)[:15]
    df_top = pd.DataFrame({
        "File":  [e.get("fname", e["name"]) for e in top15],
        "Risk":  [round(e["risk_score"] * 100, 1) for e in top15],
        "Lang":  [e["lang"][2] for e in top15],
    })
    fig3 = px.bar(df_top, x="Risk", y="File", orientation="h",
                  color="Risk",
                  color_continuous_scale=[[0,SAFE],[0.5,WARN],[1,DANGER]],
                  text="Risk", labels={"Risk": "Risk Score (%)"})
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig3.update_layout(**_plotly_base(), height=420,
                       coloraxis_showscale=False,
                       yaxis=dict(color=INK, autorange="reversed"),
                       xaxis=dict(color=INK, title="Risk (%)"))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Per-language breakdown ──────────────────────────────────────────────
    st.markdown('<p class="ds-section-header">Risk by Language</p>',
                unsafe_allow_html=True)
    df_lang = pd.DataFrame({
        "Lang":  [e["lang"][2] for e in files],
        "Risk":  [e["risk_score"] for e in files],
    })
    lang_stats = df_lang.groupby("Lang")["Risk"].agg(["mean","count"]).reset_index()
    lang_stats.columns = ["Language","Avg Risk","Files"]
    lang_stats["Avg Risk %"] = (lang_stats["Avg Risk"] * 100).round(1)

    fig4 = px.scatter(lang_stats, x="Files", y="Avg Risk %", size="Files",
                      color="Avg Risk %", text="Language",
                      color_continuous_scale=[[0,SAFE],[0.5,WARN],[1,DANGER]],
                      size_max=40)
    fig4.update_traces(textposition="top center", textfont=dict(color=INK, size=11))
    fig4.update_layout(**_plotly_base(), height=320, coloraxis_showscale=False,
                       xaxis=dict(color=INK, title="File count"),
                       yaxis=dict(color=INK, title="Avg Risk (%)"))
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _tab_training() -> None:
    st.markdown('<p class="ds-section-header">Train on NASA SDP Datasets</p>',
                unsafe_allow_html=True)

    DATA_DIR = os.path.join(ROOT, "backend", "data")
    datasets = []
    for fname in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []:
        if fname.endswith((".csv", ".arff")):
            datasets.append(os.path.join(DATA_DIR, fname))

    if not datasets:
        st.warning(f"No CSV/ARFF files found in `backend/data/`. "
                   f"Place KC1, KC2, or kc1_kc2.csv there.")
        return

    # ── config sidebar ────────────────────────────────────────────────────
    c1, c2 = st.columns([1.5, 2.5])

    with c1:
        st.markdown('<p class="ds-section-header">Configuration</p>',
                    unsafe_allow_html=True)

        ds_label = st.selectbox("Dataset", [os.path.basename(d) for d in datasets])
        ds_path  = next(d for d in datasets if os.path.basename(d) == ds_label)

        imbalance = st.selectbox(
            "Imbalance strategy",
            ["none", "smote", "adasyn"],
            help="`smote` requires imbalanced-learn (`pip install imbalanced-learn`)",
        )
        tune = st.checkbox("Hyperparameter tuning (Optuna)", value=False,
                           help="Adds ~3-8 min. Disable for quick preview.")
        cv_folds = st.slider("CV folds", 3, 10, 5)
        n_select = st.slider("Features to select (per stage)", 8, 20, 12)
        run_btn  = st.button("🚀 Train All Models", type="primary",
                             use_container_width=True)

    # ── dataset preview ───────────────────────────────────────────────────
    with c2:
        st.markdown('<p class="ds-section-header">Dataset Preview</p>',
                    unsafe_allow_html=True)
        try:
            df_prev = _cached_load_dataset(ds_path)
            defect_rate = df_prev["LABEL"].mean() * 100
            p1, p2, p3 = st.columns(3)
            p1.metric("Rows", len(df_prev))
            p2.metric("Columns", len(df_prev.columns))
            p3.metric("Defect Rate", f"{defect_rate:.1f}%",
                      delta="imbalanced" if defect_rate < 25 else "balanced",
                      delta_color="inverse" if defect_rate < 25 else "normal")

            # class count bar
            counts = df_prev["LABEL"].value_counts().rename({0:"Clean",1:"Defective"})
            fig = go.Figure(go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[SAFE, DANGER],
                text=counts.values, textposition="outside",
                textfont=dict(color=INK),
            ))
            fig.update_layout(**_plotly_base(), height=180,
                              showlegend=False,
                              xaxis=dict(color=INK),
                              yaxis=dict(color=INK, showgrid=True, gridcolor=LINE))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # ── train ─────────────────────────────────────────────────────────────
    if run_btn:
        import pathlib
        import preprocessing as pp_mod
        import models as ml_mod
        progress = st.progress(0, "Initializing pipeline...")
        log_box  = st.empty()

        with st.status("Training models...", expanded=True) as status_widget:
            try:
                # load + preprocess
                st.write("⚙️ Preprocessing dataset...")
                progress.progress(10, "Preprocessing...")

                df   = pp_mod.load_data(ds_path)
                pp   = pp_mod.SDPPreprocessor(imbalance_strategy=imbalance,
                                              n_select=n_select)
                X_tr, X_te, y_tr, y_te = pp.fit_transform(df)

                progress.progress(30, "Preprocessing done. Loading models...")
                st.write(f"✅ Features selected: {len(pp.selected_features_)}")
                st.write(f"✅ Train: {X_tr.shape} | Test: {X_te.shape}")

                # train
                st.write("🤖 Training classifiers...")

                trainer = ml_mod.ModelTrainer(
                    class_weight_dict=pp.class_weight_dict_,
                    include_ensembles=True,
                )
                results = trainer.train_all_models(
                    X_tr, X_te, y_tr, y_te,
                    tune=tune, cv_folds=cv_folds,
                )
                progress.progress(90, "Training done. Saving models...")

                # save
                save_dir = str(pathlib.Path(ROOT) / "backend" / "models")
                trainer.save_all_models(save_dir)

                # persist to session state
                st.session_state.trainer        = trainer
                st.session_state.pp             = pp
                st.session_state.X_train        = X_tr
                st.session_state.X_test         = X_te
                st.session_state.y_train        = y_tr
                st.session_state.y_test         = y_te
                st.session_state.feature_names  = pp.selected_features_
                st.session_state.train_logs     = trainer.get_logs()
                st.session_state.trained        = True

                progress.progress(100, "Complete!")
                status_widget.update(label="✅ Training complete!", state="complete")

            except Exception as exc:
                status_widget.update(label=f"❌ Error: {exc}", state="error")
                st.exception(exc)
                return

        # ── leaderboard ────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="ds-section-header">Model Leaderboard</p>',
                    unsafe_allow_html=True)

        lb = trainer.get_leaderboard()
        metric_cols = ["name","roc_auc","recall","f1","precision","accuracy",
                       "cv_roc_auc_mean","train_time_s"]
        lb_show = lb[[c for c in metric_cols if c in lb.columns]].rename(columns={
            "name":"Model","roc_auc":"AUC","recall":"Recall","f1":"F1",
            "precision":"Precision","accuracy":"Accuracy",
            "cv_roc_auc_mean":"CV-AUC","train_time_s":"Time(s)",
        })
        st.dataframe(
            lb_show.style.format({
                "AUC":":.3f","Recall":":.3f","F1":":.3f",
                "Precision":":.3f","Accuracy":":.3f","CV-AUC":":.3f","Time(s)":":.1f",
            }).background_gradient(subset=["AUC","Recall"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True,
        )

        # training log
        with st.expander("📋 Training Logs"):
            st.code("\n".join(trainer.get_logs()), language=None)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _tab_evaluation() -> None:
    if not st.session_state.get("trained"):
        st.info("Train models in the **Training** tab first.")
        return

    trainer      = st.session_state.trainer
    X_te         = st.session_state.X_test
    y_te         = st.session_state.y_test
    feat_names   = st.session_state.feature_names
    results      = trainer.results_

    if not results:
        st.warning("No trained models found.")
        return

    model_names  = list(results.keys())
    sel_model    = st.selectbox("Select model to inspect", model_names,
                                key="eval_model_select")
    res          = results[sel_model]

    # ── top KPIs ──────────────────────────────────────────────────────────
    st.markdown('<p class="ds-section-header">Performance Metrics</p>',
                unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("ROC-AUC",   f"{res.roc_auc:.3f}")
    k2.metric("Recall",    f"{res.recall:.3f}",
              delta="↑ good for SDP", delta_color="normal")
    k3.metric("F1",        f"{res.f1:.3f}")
    k4.metric("Precision", f"{res.precision:.3f}")
    k5.metric("CV-AUC",    f"{res.cv_roc_auc_mean:.3f} ± {res.cv_roc_auc_std:.3f}")

    st.divider()
    left, right = st.columns(2)

    # ── confusion matrix ──────────────────────────────────────────────────
    with left:
        st.markdown('<p class="ds-section-header">Confusion Matrix</p>',
                    unsafe_allow_html=True)
        cm = confusion_matrix(y_te, res.y_pred)
        fig = go.Figure(go.Heatmap(
            z=cm[::-1],
            x=["Predicted Clean", "Predicted Defective"],
            y=["Actual Defective", "Actual Clean"],
            colorscale=[[0, PANEL],[0.5, f"{BRAND}88"],[1, BRAND]],
            text=cm[::-1],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=20, color=INK),
            showscale=False,
        ))
        tn, fp, fn, tp = cm.ravel()
        fig.update_layout(
            **_plotly_base(), height=300,
            annotations=[
                dict(x=0, y=1, text=f"TN={tn}", showarrow=False,
                     font=dict(color=MUTED, size=11)),
                dict(x=1, y=1, text=f"FP={fp}", showarrow=False,
                     font=dict(color=DANGER, size=11)),
                dict(x=0, y=0, text=f"FN={fn}", showarrow=False,
                     font=dict(color=DANGER, size=11)),
                dict(x=1, y=0, text=f"TP={tp}", showarrow=False,
                     font=dict(color=BRAND, size=11)),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── ROC curve ────────────────────────────────────────────────────────
    with right:
        st.markdown('<p class="ds-section-header">ROC Curve — All Models</p>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dash", color=MUTED, width=1),
            name="Random", showlegend=False,
        ))
        for mname, mres in results.items():
            fpr, tpr, _ = roc_curve(y_te, mres.y_proba)
            fig2.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=mname,
                line=dict(color=_model_color(mname), width=2),
                hovertemplate=f"{mname}<br>FPR=%{{x:.3f}} TPR=%{{y:.3f}}<extra></extra>",
            ))
        fig2.update_layout(
            **_plotly_base(), height=300,
            xaxis=dict(color=INK, title="False Positive Rate",
                       showgrid=True, gridcolor=LINE),
            yaxis=dict(color=INK, title="True Positive Rate",
                       showgrid=True, gridcolor=LINE),
            legend=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Radar chart — model comparison ─────────────────────────────────────
    st.markdown('<p class="ds-section-header">Multi-Metric Radar Comparison</p>',
                unsafe_allow_html=True)
    cats = ["AUC","Recall","F1","Precision","Accuracy"]
    fig3 = go.Figure()
    for mname, mres in results.items():
        vals = [mres.roc_auc, mres.recall, mres.f1, mres.precision, mres.accuracy]
        fig3.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself", fillcolor=_model_color(mname) + "22",
            line=dict(color=_model_color(mname), width=2),
            name=mname,
        ))
    fig3.update_layout(
        **_plotly_base(), height=420,
        polar=dict(
            bgcolor=PANEL,
            radialaxis=dict(visible=True, range=[0, 1],
                            color=MUTED, gridcolor=LINE),
            angularaxis=dict(color=INK),
        ),
        legend=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(size=10)),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Feature importance table ───────────────────────────────────────────
    if hasattr(res.model, "feature_importances_") and feat_names:
        st.markdown('<p class="ds-section-header">Feature Importance</p>',
                    unsafe_allow_html=True)
        imp = pd.Series(res.model.feature_importances_, index=feat_names)
        imp = imp.sort_values(ascending=False).head(15)

        fig4 = go.Figure(go.Bar(
            x=imp.values[::-1], y=imp.index[::-1],
            orientation="h",
            marker=dict(
                color=imp.values[::-1],
                colorscale=[[0, SAFE],[0.5, WARN],[1, DANGER]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in imp.values[::-1]],
            textposition="outside",
            textfont=dict(color=INK, size=10),
        ))
        fig4.update_layout(**_plotly_base(), height=400,
                           xaxis=dict(color=INK, title="Importance"),
                           yaxis=dict(color=INK))
        st.plotly_chart(fig4, use_container_width=True)



# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — AI RISK ASSESSMENT (Groq-powered)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ai_reviewer():
    """Import ai_reviewer module."""
    import ai_reviewer as mod
    return mod


def _risk_gauge_html(score: float) -> str:
    """Compact risk score gauge in HTML/CSS."""
    pct   = score * 100
    color = DANGER if score >= 0.67 else (WARN if score >= 0.33 else SAFE)
    label = "HIGH RISK" if score >= 0.67 else ("MEDIUM RISK" if score >= 0.33 else "LOW RISK")
    return (
        f'<div style="text-align:center;background:{PANEL};border:1px solid {LINE};'

        f'border-radius:14px;padding:20px 10px;margin-bottom:8px">'

        f'<div style="font-size:2.6rem;font-weight:800;color:{color};line-height:1">{pct:.0f}%</div>'

        f'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;'

        f'color:{color};margin-top:4px;font-weight:700">{label}</div>'

        f'<div style="background:{LINE};border-radius:4px;height:6px;margin:10px 0 4px">'

        f'<div style="background:{color};height:6px;border-radius:4px;width:{pct:.0f}%"></div>'

        f'</div></div>'
    )


def _tab_prediction() -> None:
    files    = st.session_state.files
    analyzed = st.session_state.analyzed

    if not analyzed or not files:
        st.markdown(
            f'''<div style="background:{PANEL};border:1px dashed {LINE};border-radius:14px;
                    padding:48px;text-align:center">
                <p style="font-size:2rem;margin:0">&#x1f916;</p>
                <p style="font-size:1.05rem;font-weight:700;margin:8px 0 4px">AI Risk Assessment</p>
                <p style="font-size:.86rem;color:{MUTED};margin:0">
                    Upload &amp; analyze source files in the <b>Workspace</b> tab first.</p>
            </div>''', unsafe_allow_html=True)
        return

    try:
        ai_mod = _load_ai_reviewer()
    except Exception as e:
        st.error(f"Cannot load ai_reviewer module: {e}")
        return

    left, right = st.columns([1.1, 2.9], gap="medium")

    with left:
        st.markdown('<p class="ds-section-header">Groq AI Configuration</p>',
                    unsafe_allow_html=True)

        api_key = st.text_input(
            "Groq API Key", type="password", placeholder="gsk_...",
            key="groq_api_key", help="Get your free key at console.groq.com",
        )

        model_id = st.selectbox(
            "Model", options=list(ai_mod.GROQ_MODELS.keys()),
            format_func=lambda m: ai_mod.GROQ_MODELS[m], key="groq_model",
        )

        lang = st.radio(
            "Language", ["vi", "en"],
            format_func=lambda x: "Tieng Viet" if x == "vi" else "English",
            horizontal=True, key="groq_lang",
        )

        if st.button("Validate API Key", use_container_width=True):
            if not api_key:
                st.warning("Nhap API key truoc.")
            else:
                with st.spinner("Kiem tra..."):
                    ok, msg = ai_mod.validate_api_key(api_key)
                if ok:
                    st.success("API Key hop le")
                    st.session_state["groq_key_valid"] = True
                else:
                    st.error(msg)
                    st.session_state["groq_key_valid"] = False

        if st.session_state.get("groq_key_valid"):
            st.markdown(
                f'<div style="background:{SAFE}22;border:1px solid {SAFE}55;'

                f'border-radius:6px;padding:4px 10px;font-size:.78rem;color:{SAFE};'

                f'text-align:center">API Key Valid</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown('<p class="ds-section-header">Analysis Mode</p>', unsafe_allow_html=True)
        mode = st.radio(
            "Mode", ["single_file", "workspace"],
            format_func=lambda x: (
                "Single File Deep Dive" if x == "single_file" else "Workspace Overview"
            ),
            label_visibility="collapsed", key="ai_mode",
        )

        if mode == "single_file":
            st.divider()
            st.markdown('<p class="ds-section-header">Select File</p>', unsafe_allow_html=True)
            sorted_files = sorted(files, key=lambda f: f.get("risk_score", 0), reverse=True)
            file_labels  = [
                f"{f['lang'][0]} {f.get('fname', f['name'])} -- {f['risk_score']*100:.0f}%"
                for f in sorted_files
            ]
            sel_idx = st.selectbox(
                "File", range(len(sorted_files)),
                format_func=lambda i: file_labels[i], key="ai_file_select",
            )
            selected_file = sorted_files[sel_idx]
            st.markdown(_risk_gauge_html(selected_file["risk_score"]), unsafe_allow_html=True)

            m = selected_file.get("metrics", {})
            c1, c2 = st.columns(2)
            c1.metric("LOC",       m.get("loc",   "--"))
            c2.metric("CC",        m.get("cc",    "--"))
            c1.metric("Functions", m.get("funcs", "--"))
            c2.metric("Comment%",
                      f"{m.get('comment_ratio', 0)*100:.1f}%"
                      if m.get("comment_ratio") else "--")

            ctx_preview = ai_mod.build_file_context(selected_file)
            n_v, n_h = len(ctx_preview.violations), len(ctx_preview.hotspots)
            if n_v or n_h:
                st.markdown(
                    f'<div style="background:{DANGER}18;border:1px solid {DANGER}44;'

                    f'border-radius:8px;padding:8px 12px;font-size:.8rem;margin-top:4px">'

                    f'{n_v} metric violations &middot; {n_h} hotspots</div>',
                    unsafe_allow_html=True,
                )
            run_label = "Analyze This File"
        else:
            selected_file = None
            high = sum(1 for f in files if f.get("risk_score", 0) >= 0.67)
            med  = sum(1 for f in files if 0.33 <= f.get("risk_score", 0) < 0.67)
            st.metric("Files to review", f"{high} HIGH + {med} MED")
            run_label = "Analyze Workspace"

        st.divider()
        run_btn = st.button(run_label, type="primary",
                            use_container_width=True, disabled=not api_key)
        st.caption("Powered by Groq -- ultra-fast inference")

    with right:
        st.markdown('<p class="ds-section-header">AI Analysis Report</p>',
                    unsafe_allow_html=True)

        cache_key = (
            f"ai_report_{selected_file['name']}" if selected_file
            else "ai_report_workspace"
        )

        if not run_btn:
            cached = st.session_state.get(cache_key)
            if cached:
                st.markdown(cached)
                if st.button("Re-analyze", key="reanalyze_cached"):
                    del st.session_state[cache_key]
                    st.rerun()
            else:
                st.markdown(
                    f'<div style="background:{PANEL};border:1px dashed {LINE};'

                    f'border-radius:12px;padding:48px;text-align:center">'

                    f'<p style="font-size:.9rem;color:{MUTED};margin:0">'

                    f'Configure your API key and click <b>{run_label}</b>.'

                    f'</p></div>', unsafe_allow_html=True,
                )
            return

        if not api_key:
            st.error("Enter your Groq API key first.")
            return

        try:
            reviewer = ai_mod.GroqReviewer(api_key=api_key.strip(), model=model_id)
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            return

        report_box  = st.empty()
        full_report = ""
        token_box   = st.empty()
        model_label = ai_mod.GROQ_MODELS.get(model_id, model_id)

        with st.status(f"Analyzing with {model_label}...", expanded=True) as status:
            try:
                if mode == "single_file" and selected_file:
                    ctx    = ai_mod.build_file_context(selected_file)
                    stream = reviewer.review_stream(ctx, language=lang)
                    st.write(f"Analyzing: {selected_file['name']}")
                    st.write(f"Risk: {selected_file['risk_score']:.1%} | "
                             f"Violations: {len(ctx.violations)} | "
                             f"Hotspots: {len(ctx.hotspots)}")
                else:
                    stream = reviewer.review_workspace(files, language=lang)
                    st.write(f"Analyzing {len(files)} files...")

                token_count = 0
                for chunk in stream:
                    full_report += chunk
                    token_count += len(chunk.split())
                    report_box.markdown(full_report + " |")
                    token_box.caption(f"{token_count} tokens streamed...")

                report_box.markdown(full_report)
                token_box.empty()
                status.update(label="Analysis complete!", state="complete")
                st.session_state[cache_key] = full_report

            except Exception as e:
                status.update(label="Error", state="error")
                err = str(e)
                if "401" in err:
                    st.error("Invalid API key — check at console.groq.com")
                elif "429" in err:
                    st.error("Rate limit hit. Wait a moment and retry.")
                else:
                    st.error(f"Groq API error: {err}")
                return

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            fname = (
                selected_file["fname"].replace(".", "_") + ".md"
                if selected_file else "workspace_risk_report.md"
            )
            st.download_button(
                "Download Report (.md)", data=full_report,
                file_name=fname, mime="text/markdown", use_container_width=True,
            )
        with cb:
            if st.button("Re-analyze", use_container_width=True, key="reanalyze_after"):
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _do_analysis() -> None:
    """Validate, extract, and run defect analysis on uploaded files."""
    # validation
    has_files = bool(st.session_state.get("files_input"))
    has_zip   = bool(st.session_state.get("zip_input"))

    if not has_files and not has_zip:
        st.error("⚠️ Please upload source files or a ZIP archive first.")
        return

    entries = []
    if has_zip:
        with st.spinner("Extracting ZIP..."):
            raw = st.session_state["zip_input"].getvalue()
            entries = build_entries_from_zip(raw)
        if not entries:
            st.error("No supported source files found in ZIP. "
                     "Ensure it contains .py/.java/.js/... files "
                     "outside library directories (venv/, node_modules/, etc.).")
            return
    elif has_files:
        entries = build_entries_from_files(st.session_state["files_input"])
        if not entries:
            st.error("No valid source files uploaded.")
            return

    st.toast(f"Analysing {len(entries)} files…", icon="⚙️")
    with st.spinner(f"Running defect analysis on {len(entries)} files…"):
        results, logs, summary = run_analysis(entries)

    st.session_state.files       = results
    st.session_state.logs        = logs
    st.session_state.analysis    = summary
    st.session_state.analyzed    = True
    st.session_state.active_file = results[0]["name"] if results else None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────

def _render_topbar() -> None:
    files    = st.session_state.files
    analysis = st.session_state.analysis
    analyzed = st.session_state.analyzed

    total_loc = analysis["total_loc"] if analysis else "—"
    avg_risk  = f"{analysis['avg_risk']*100:.1f}%" if analysis else "—"
    n_files   = len(files) if files else 0
    trained   = "✓ Models trained" if st.session_state.get("trained") else "No models trained"

    st.markdown(
        f"""
        <div class="ds-topbar">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <p class="ds-topbar-title">🔍 DefectSight Workbench</p>
                    <p class="ds-topbar-sub">AI-driven defect screening · {n_files} files
                    · LOC: {total_loc} · Avg risk: {avg_risk} · {trained}</p>
                </div>
                <div style="font-size:.8rem;opacity:.75">{trained}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_state()
    apply_global_styles()
    _apply_extra_css()

    # init extra state keys
    for key in ("trainer", "pp", "X_train", "X_test", "y_train", "y_test",
                "feature_names", "train_logs", "trained"):
        if key not in st.session_state:
            st.session_state[key] = None if key != "trained" else False

    _render_topbar()

    # ── main tab bar ───────────────────────────────────────────────────────
    tab_workspace, tab_dashboard, tab_training, tab_evaluation, tab_prediction = st.tabs([
        "🏠 Workspace",
        "📊 Dashboard",
        "🤖 Training",
        "🔍 Evaluation",
        "🎯 Prediction",
    ])

    with tab_workspace:
        _tab_workspace()

    with tab_dashboard:
        _tab_dashboard()

    with tab_training:
        _tab_training()

    with tab_evaluation:
        _tab_evaluation()

    with tab_prediction:
        _tab_prediction()

    # ── console log (always visible at bottom) ─────────────────────────────
    logs = st.session_state.logs
    if logs:
        with st.expander("📟 Console Log", expanded=False):
            html_lines = []
            for line in logs[-120:]:
                esc = (line.replace("&","&amp;")
                           .replace("<","&lt;")
                           .replace(">","&gt;"))
                color = (
                    "#7eddaf" if any(k in line for k in ("[OK]","[SUCCESS]")) else
                    "#ff8f99" if "HIGH" in line else
                    "#ffd58a" if "MED"  in line else
                    "#7ec6ff" if any(k in line for k in ("[RUNNING]","[LOADING]","[METRICS]")) else
                    "#c5d0da"
                )
                html_lines.append(f"<span style='color:{color}'>{esc}</span>")
            st.markdown(
                f"<div class='ds-log'>{'<br>'.join(html_lines)}</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
