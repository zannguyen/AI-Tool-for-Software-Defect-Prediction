"""
DefectSight — AI Software Defect Prediction Tool
Frontend: Streamlit UI — VS Code Style Single Viewport
"""

import os, sys
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, 'backend'))

st.set_page_config(
    page_title="DefectSight",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛡️"
)

from backend.api import (
    build_entries_from_files, build_entries_from_zip,
    run_analysis, get_risk_label, render_file_code,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Lock viewport, no body scroll, VS Code look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

:root {
  --bg0: #1e1e1e;
  --bg1: #252526;
  --bg2: #2d2d2d;
  --bg3: #3c3c3c;
  --accent: #007acc;
  --green: #4ec9b0;
  --red: #f85149;
  --yellow: #e2c08d;
  --txt: #cccccc;
  --txt2: #969696;
  --txt3: #6e6e6e;
  --border: #3c3c3c;
}

/* ── Kill all page scroll ── */
html, body {
    height: 100% !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
.stApp {
    background: var(--bg0) !important;
    height: 100vh !important;
    overflow: hidden !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
}
.block-container {
    height: 100vh !important;
    padding: 0 !important;
    max-width: 100% !important;
    overflow: hidden !important;
}
header, footer, [data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stMainMenu"] {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
}

/* ── Kill ALL Streamlit vertical spacing between blocks ── */
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stVerticalBlock"] > div,
[data-testid="element-container"] {
    margin: 0 !important;
    padding: 0 !important;
}
/* Kill st.container border/padding that adds space */
[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Columns: position:fixed at top:30px (below title bar), like terminal & status bar ── */
/* ── OUTER 3-column layout — position:fixed below title bar ── */
[data-testid="stHorizontalBlock"] {
    position: fixed !important;
    top: 30px !important;
    left: 0 !important;
    right: 0 !important;
    height: calc(100vh - 202px) !important;
    min-height: 0 !important;
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    align-items: stretch !important;
    overflow: hidden !important;
    display: flex !important;
}
/* ── NESTED horizontal blocks (inside columns) — restore normal flow ── */
[data-testid="column"] [data-testid="stHorizontalBlock"] {
    position: static !important;
    top: auto !important;
    left: auto !important;
    right: auto !important;
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
    display: flex !important;
}
[data-testid="column"] {
    overflow: hidden !important;
    height: 100% !important;
    padding: 0 !important;
    min-height: 0 !important;
}
/* Streamlit's inner vertical block inside columns — no extra padding */
[data-testid="column"] > div > [data-testid="stVerticalBlock"] {
    height: 100% !important;
    gap: 0 !important;
    overflow: hidden !important;
}

/* ── Terminal Panel: fixed bottom, above status bar ──
   terminal top = 100vh - 22(status) - 150(terminal) = 100vh - 172px
   column bottom = 30(title) + (100vh - 202) = 100vh - 172px  ✔ no gap ── */
.ds-terminal-fixed {
    position: fixed !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 22px !important;
    height: 150px !important;
    background: var(--bg1) !important;
    border-top: 1px solid var(--border) !important;
    z-index: 8888 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
.ds-terminal-hdr {
    height: 30px;
    background: var(--bg2);
    display: flex;
    align-items: center;
    padding: 0 14px;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.ds-terminal-tab {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--txt);
    border-bottom: 2px solid var(--accent);
    padding: 0 4px;
    height: 30px;
    display: flex;
    align-items: center;
}
.ds-terminal-scroll {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 12px;
    line-height: 18px;
    white-space: pre-wrap;
    word-break: break-all;
}
.ds-terminal-scroll::-webkit-scrollbar { width: 6px; }
.ds-terminal-scroll::-webkit-scrollbar-thumb { background: #424242; border-radius: 3px; }

/* ── Status bar: fixed to bottom ── */
.ds-statusbar {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 9999 !important;
}

/* ── Title bar: position:fixed so it never gets overlapped ── */
.ds-titlebar {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 9990 !important;
    height: 30px !important;
    background: #323233;
    border-bottom: 1px solid var(--border);
    display: flex !important;
    align-items: center;
    padding: 0 12px;
    font-size: 13px;
    color: var(--txt);
    user-select: none;
    flex-shrink: 0;
}
/* Spacer div that replaces title bar in DOM flow */
.ds-titlebar-spacer {
    height: 30px !important;
    flex-shrink: 0;
    display: block;
}
.ds-tb-logo { margin-right: 8px; font-size: 15px; }
.ds-tb-title { font-weight: 600; color: #fff; }
.ds-tb-sub { color: var(--txt2); margin-left: 8px; font-size: 11px; }

/* ── LEFT PANEL ── */
.ds-left {
    height: calc(100vh - 52px); /* titlebar + statusbar */
    background: var(--bg1);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* ── Panel Header ── */
.ds-pane-header {
    height: 35px;
    padding: 0 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    background: var(--bg1);
}
.ds-pane-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--txt);
}
.ds-pane-action {
    font-size: 11px;
    color: var(--accent);
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    user-select: none;
}

/* ── File Tree scrollable zone ── */
.ds-tree-wrap {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 4px 0;
}
.ds-tree-wrap::-webkit-scrollbar { width: 6px; }
.ds-tree-wrap::-webkit-scrollbar-thumb { background: #424242; border-radius: 3px; }

/* ── Tree buttons styling ── */
[data-testid="column"]:first-child .stButton > button {
    border: none !important;
    background: transparent !important;
    padding: 2px 8px !important;
    min-height: 22px !important;
    height: 22px !important;
    line-height: 1 !important;
    color: var(--txt) !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    transition: background 0.08s !important;
    transform: none !important;
    box-shadow: none !important;
}
[data-testid="column"]:first-child .stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="column"]:first-child .stButton > button[data-active="true"],
[data-testid="column"]:first-child .stButton[data-selected] > button {
    background: rgba(0,122,204,0.25) !important;
    color: #fff !important;
}
[data-testid="column"]:first-child .stButton > button p {
    text-align: left !important;
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 13px !important;
}

/* ── Import form styling ── */
.ds-import-wrap {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex: 1;
    overflow-y: auto;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed var(--txt3) !important;
    border-radius: 6px !important;
    margin-top: 14px !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ── Status bar ── */
.ds-statusbar {
    height: 22px;
    background: var(--accent);
    display: flex;
    align-items: center;
    padding: 0 6px;
    font-size: 12px;
    color: #fff;
    flex-shrink: 0;
    gap: 0;
}
.ds-sb-item {
    padding: 0 8px;
    height: 22px;
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: default;
    white-space: nowrap;
    transition: background 0.1s;
}
.ds-sb-item:hover { background: rgba(255,255,255,.15); }
.ds-sb-sep { width: 1px; height: 16px; background: rgba(255,255,255,.2); }
.ds-sb-right { margin-left: auto; display: flex; }

/* ── Center panel ── */
.ds-center {
    height: calc(100vh - 52px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg0);
    border-right: 1px solid var(--border);
}

/* ── Tab bar ── */
.ds-tabs {
    height: 35px;
    background: var(--bg1);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-end;
    overflow-x: auto;
    overflow-y: hidden;
    flex-shrink: 0;
}
.ds-tabs::-webkit-scrollbar { height: 3px; }
.ds-tab {
    height: 35px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    border-right: 1px solid var(--border);
    border-top: 1px solid transparent;
    background: var(--bg1);
    color: var(--txt2);
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
    flex-shrink: 0;
}
.ds-tab.active {
    background: var(--bg0);
    color: #fff;
    border-top-color: var(--accent);
}

/* ── Breadcrumb ── */
.ds-breadcrumb {
    height: 24px;
    background: var(--bg0);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 12px;
    gap: 6px;
    font-size: 12px;
    color: var(--txt3);
    flex-shrink: 0;
}
.ds-bc-sep { color: var(--txt3); }
.ds-bc-name { color: var(--txt); }
.ds-bc-risk-hi { color: var(--red); margin-left: auto; font-weight: 600; font-size: 11px; }
.ds-bc-risk-me { color: var(--yellow); margin-left: auto; font-weight: 600; font-size: 11px; }
.ds-bc-risk-lo { color: var(--green); margin-left: auto; font-weight: 600; font-size: 11px; }

/* ── Code area: col_height(100vh-202) - tabs(35) - breadcrumb(24) - 2px ── */
.ds-code-area {
    height: calc(100vh - 263px) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    background: var(--bg0);
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 13px;
    line-height: 19px;
    display: block !important;
}
.ds-code-area::-webkit-scrollbar { width: 8px; }
.ds-code-area::-webkit-scrollbar-thumb { background: #424242; border-radius: 4px; }
.vs-cl { display:flex; align-items:flex-start; min-height:19px; padding-right:8px; }
.vs-cl:hover { background: rgba(255,255,255,.03); }
.vs-cl.hi { background: rgba(248,81,73,.08); border-left: 3px solid rgba(248,81,73,.5); }
.vs-cl.me { background: rgba(226,192,141,.06); border-left: 3px solid rgba(226,192,141,.3); }
.vs-ln { width:48px; flex-shrink:0; text-align:right; padding:0 14px 0 8px; color:var(--txt3); font-size:12px; user-select:none; }
.vs-cl.hi .vs-ln { color:rgba(248,81,73,.5); }
.vs-cl.me .vs-ln { color:rgba(226,192,141,.4); }
.vs-code-txt { color:var(--txt); white-space:pre; flex:1; overflow:hidden; }
.vs-rb { font-size:9px; padding:1px 5px; border-radius:3px; font-weight:600; margin-top:4px; flex-shrink:0; font-family:'Inter',sans-serif; }
.vs-rb.hi { background:rgba(248,81,73,.2); color:#ff6b6b; border:1px solid rgba(248,81,73,.3); }
.vs-rb.me { background:rgba(226,192,141,.15); color:#e2c08d; border:1px solid rgba(226,192,141,.25); }
.kw{color:#569cd6}.fn{color:#dcdcaa}.cls{color:#4ec9b0}.str{color:#ce9178}.num{color:#b5cea8}.dec{color:#9cdcfe}

/* ── Welcome screen ── */
.ds-welcome {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--txt2);
}
.ds-welcome-icon { font-size: 56px; opacity: .15; }
.ds-welcome-title { font-size: 22px; font-weight: 400; color: var(--txt); }
.ds-welcome-sub { font-size: 13px; color: var(--txt3); text-align: center; line-height: 1.7; }

/* ── Right panel: analysis body h = col_height(100vh-202) - pane_header(35) ── */
.ds-right-analysis {
    height: calc(100vh - 237px) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 0 0 8px 0;
}
.ds-right-analysis::-webkit-scrollbar { width: 6px; }
.ds-right-analysis::-webkit-scrollbar-thumb { background: #424242; border-radius: 3px; }

/* ── Risk summary cards ── */
.ds-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-bottom: 12px;
}
.ds-stat {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
}
.ds-stat-n { font-size: 1.6rem; font-weight: 600; }
.ds-stat-l { font-size: 10px; text-transform: uppercase; color: var(--txt3); letter-spacing: .05em; }
.ds-mc {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.ds-mc-name { font-size: 12px; font-weight: 600; margin-bottom: 6px; }
.ds-mc-row { display:flex; justify-content:space-between; font-size:11px; color:var(--txt2); margin-bottom:3px; }
.ds-mc-bar { height:3px; background:var(--bg3); border-radius:2px; margin-bottom:6px; overflow:hidden; }
.ds-mc-fill { height:100%; border-radius:2px; }
.ds-log-line { font-family:'JetBrains Mono',monospace; font-size:11px; line-height:17px; padding:0 4px; }

/* ── Terminal Panel (bottom, full-width) ── */
.ds-terminal-panel {
    height: 160px;
    background: var(--bg1);
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
}
.ds-terminal-header {
    height: 30px;
    background: var(--bg2);
    display: flex;
    align-items: center;
    padding: 0 12px;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.ds-terminal-tab {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--txt);
    border-bottom: 2px solid var(--accent);
    padding: 0 4px;
    height: 30px;
    display: flex;
    align-items: center;
    gap: 5px;
}
.ds-terminal-body {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 12px;
    line-height: 18px;
}
.ds-terminal-body::-webkit-scrollbar { width: 6px; }
.ds-terminal-body::-webkit-scrollbar-thumb { background: #424242; border-radius: 3px; }

/* Run button override → primary VS Code blue */
[data-testid="column"]:first-child .stButton > button[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border-radius: 4px !important;
    height: 30px !important;
    font-size: 12px !important;
    padding: 0 16px !important;
    border: none !important;
    width: 100% !important;
}
[data-testid="column"]:first-child .stButton > button[data-testid="baseButton-primary"]:hover {
    background: #1a8ad4 !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    'files': [], 'analysis': None, 'analyzed': False,
    'model_choice': 'All Models', 'logs': [], 'active_file': None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

files    = st.session_state.files
analysis = st.session_state.analysis
analyzed = st.session_state.analyzed
logs     = st.session_state.logs

if files and st.session_state.active_file not in [f['name'] for f in files]:
    st.session_state.active_file = files[0]['name']

active_f = next((f for f in files if f['name'] == st.session_state.active_file), files[0] if files else None)


# ─────────────────────────────────────────────────────────────────────────────
# TITLE BAR (HTML only — no widget)
# ─────────────────────────────────────────────────────────────────────────────
nf = len(files)
tl = analysis['total_loc'] if analysis else 0
pct = int(analysis['avg_risk'] * 100) if analysis else 0
status_txt = f"{analysis['high']} HIGH · {analysis['med']} MED" if analyzed else "Ready"
risk_col = "#ff8080" if pct >= 60 else "#ffd580" if pct >= 35 else "#80d080"
lang_name = active_f['lang'][2] if active_f else "—"

st.markdown(
    f'<div class="ds-titlebar">'
    f'<span class="ds-tb-logo">🛡️</span>'
    f'<span class="ds-tb-title">DefectSight</span>'
    f'<span class="ds-tb-sub">— defect-prediction-tool</span>'
    f'</div>',
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# TREE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_tree(files_list):
    tree = {}
    for f in files_list:
        parts = [p for p in f['name'].replace('\\', '/').split('/') if p]
        curr = tree
        for part in parts[:-1]:
            if part not in curr or not isinstance(curr[part], dict) or 'content' in curr[part]:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = f
    return tree


def render_tree(node, depth=0, parent_path=""):
    def sort_key(item):
        k, v = item
        is_file = isinstance(v, dict) and 'content' in v
        return (is_file, k.lower())

    for k, v in sorted(node.items(), key=sort_key):
        is_file = isinstance(v, dict) and 'content' in v
        indent = "\u00a0\u00a0\u00a0" * depth

        if not is_file:
            fk = f"dir_{parent_path}_{k}"
            is_open = st.session_state.get(fk, depth < 1)
            d_icon = "▾" if is_open else "▸"
            label = f"{indent}{d_icon} 📁 {k}"
            if st.button(label, key=f"btn_{fk}", use_container_width=True):
                st.session_state[fk] = not is_open
                st.rerun()
            if is_open:
                render_tree(v, depth + 1, fk)
        else:
            fe = v
            lang = fe.get('lang', ('📄', '#9d9d9d', 'Text'))
            ficon = lang[0]
            badge = ""
            if analyzed and 'risk_score' in fe:
                rs = fe['risk_score']
                badge = " ●" if rs >= 0.5 else " ◑" if rs >= 0.3 else ""
            is_active = fe['name'] == st.session_state.active_file
            label = f"{indent}  {ficon} {k}{badge}"
            # Use button; highlight active via CSS class trick via markdown
            if is_active:
                st.markdown(
                    f'<div style="background:rgba(0,122,204,.25);border-left:2px solid #007acc;'
                    f'padding:1px 8px 1px {8 + depth*18}px;cursor:default;font-size:13px;'
                    f'color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'{ficon} {k}{badge}</div>',
                    unsafe_allow_html=True
                )
                # invisible button to allow clicking
                st.button(label, key=f"fbtn_{fe['name']}", use_container_width=True)
            else:
                if st.button(label, key=f"fbtn_{fe['name']}", use_container_width=True):
                    st.session_state.active_file = fe['name']
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# THREE-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1.4, 3.5, 1.4], gap="small")

# ══════════════════════════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════════════════════════
with col_left:
    if not analyzed:
        # ── IMPORT MODE ──
        st.markdown(
            '<div class="ds-pane-header">'
            '<span class="ds-pane-title">📁 Import Workspace</span>'
            '</div>',
            unsafe_allow_html=True
        )
        ufiles = st.file_uploader(
            "code_files", accept_multiple_files=True, label_visibility="collapsed",
            key="uf",
            type=['py','java','js','jsx','ts','tsx','c','cpp','cs','h','hpp','go','rs','rb','php','swift','kt','html','css','sql']
        )
        uzip = st.file_uploader(
            "zip_file", type=['zip'], label_visibility="collapsed", key="uz"
        )
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
        run_clicked = st.button("▶  Run Analysis", key="run_btn", type="primary", use_container_width=True)
        if run_clicked:
            entries = []
            if st.session_state.get('uz'):
                with st.spinner("Extracting ZIP..."):
                    entries = build_entries_from_zip(st.session_state['uz'].getvalue())
            elif st.session_state.get('uf'):
                entries = build_entries_from_files(st.session_state['uf'])
            if entries:
                results, logs_out, summary = run_analysis(entries)
                st.session_state.files = results
                st.session_state.logs = logs_out
                st.session_state.analysis = summary
                st.session_state.analyzed = True
                st.session_state.active_file = results[0]['name']
                st.rerun()
            else:
                st.warning("Upload files or ZIP first.")
    else:
        # ── EXPLORER MODE ──
        st.markdown(
            '<div class="ds-pane-header">'
            f'<span class="ds-pane-title">EXPLORER — {nf} files</span>'
            '</div>',
            unsafe_allow_html=True
        )
        # "New Import" reset button — small link style
        if st.button("⊕ New Import", key="btn_new_import"):
            for k in ['files','analysis','analyzed','logs','active_file']:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

        # Spacer between New Import and model selector
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        # ── Model selector label + buttons ──
        st.markdown(
            '<div style="padding:0 8px 4px;font-size:10px;font-weight:600;'
            'text-transform:uppercase;letter-spacing:.08em;color:var(--txt3);">Model</div>',
            unsafe_allow_html=True
        )
        model_map = {'All': 'All Models', 'LR': 'Logistic Regression', 'RF': 'Random Forest', 'NN': 'Neural Network'}
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col_m, (short, full) in zip([mc1, mc2, mc3, mc4], model_map.items()):
            with col_m:
                is_sel = st.session_state.model_choice == full
                clicked = st.button(
                    short,
                    key=f"mc_{short}",
                    type="secondary" if is_sel else "tertiary",
                )
                if clicked:
                    st.session_state.model_choice = full
                    st.rerun()

        # Spacer before file tree
        st.markdown('<div style="height:6px;border-top:1px solid var(--border);margin-top:4px;"></div>', unsafe_allow_html=True)

        # ── File Tree — scrollable, fills remaining left panel space ──
        tree_data = build_tree(files)
        # height = column_height(100vh-222) - pane_header(35) - new_import_btn(38) - model_row(46) - hr(12) ≈ col_h - 131
        # So tree_height ≈ 100vh - 222 - 131 = 100vh - 353 ≈ for 925px screen = ~572px
        tree_container = st.container(height=500, border=False)
        with tree_container:
            render_tree(tree_data)


# ══════════════════════════════════════════════════════════════════
# CENTER PANEL — Editor
# ══════════════════════════════════════════════════════════════════
with col_center:
    if not files or not active_f:
        st.markdown(
            '<div class="ds-welcome">'
            '<span class="ds-welcome-icon">🛡️</span>'
            '<span class="ds-welcome-title">DefectSight</span>'
            '<span class="ds-welcome-sub">AI-Powered Software Defect Prediction<br>Import source files to begin.</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Tab bar
        tab_html = '<div class="ds-tabs">'
        shown = [active_f] + [f for f in files if f['name'] != active_f['name']][:6]
        for f in shown:
            is_on = ' active' if f['name'] == active_f['name'] else ''
            icon = f['lang'][0]
            tab_html += f'<div class="ds-tab{is_on}">{icon} {f["fname"] if "fname" in f else f["name"]}</div>'
        tab_html += '</div>'
        st.markdown(tab_html, unsafe_allow_html=True)

        # Breadcrumb
        bc_risk = ""
        if analyzed:
            rl, rc = get_risk_label(active_f['risk_score'])
            bc_cls = 'hi' if rc == '#f85149' else 'me' if rc == '#d29922' else 'lo'
            bc_risk = f'<span class="ds-bc-risk-{bc_cls}">{rl} {active_f["risk_score"]*100:.1f}%</span>'
        parts_bc = [p for p in active_f['name'].replace('\\', '/').split('/') if p]
        bc_inner = ' <span class="ds-bc-sep">›</span> '.join(
            f'<span class="ds-bc-name">{p}</span>' for p in parts_bc
        )
        st.markdown(
            f'<div class="ds-breadcrumb">{bc_inner}{bc_risk}</div>',
            unsafe_allow_html=True
        )

        # Code viewer — fills remaining height
        code_html = render_file_code(active_f, analyzed)
        st.markdown(f'<div class="ds-code-area">{code_html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# RIGHT PANEL — Analysis
# ══════════════════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="ds-pane-header"><span class="ds-pane-title">Analysis</span></div>', unsafe_allow_html=True)

    # All analysis content goes into one scrollable block
    right_html = '<div class="ds-right-analysis">'

    if analyzed and analysis:
        avg_pct = analysis["avg_risk"] * 100
        risk_clr = "#f85149" if avg_pct >= 50 else "#e2c08d" if avg_pct >= 30 else "#4ec9b0"
        right_html += (
            f'<div style="padding:10px 12px 6px;">'
            f'<div style="font-size:10px;text-transform:uppercase;color:var(--txt3);letter-spacing:.07em;">Workspace Risk</div>'
            f'<div style="font-size:2rem;font-weight:700;color:{risk_clr};line-height:1.2;">{avg_pct:.1f}%</div>'
            f'</div>'
            f'<div class="ds-stat-grid" style="padding:0 12px 10px;">'
            f'<div class="ds-stat"><div class="ds-stat-n" style="color:#f85149;">{analysis["high"]}</div><div class="ds-stat-l">High</div></div>'
            f'<div class="ds-stat"><div class="ds-stat-n" style="color:#e2c08d;">{analysis["med"]}</div><div class="ds-stat-l">Med</div></div>'
            f'<div class="ds-stat"><div class="ds-stat-n" style="color:#4ec9b0;">{analysis["low"]}</div><div class="ds-stat-l">Low</div></div>'
            f'</div>'
            f'<div style="padding:0 12px 8px;font-size:11px;color:var(--txt2);">Total LOC: <strong style="color:var(--txt);">{analysis["total_loc"]}</strong></div>'
            f'<hr style="border:none;border-top:1px solid var(--border);margin:0 12px 8px;">'
        )

        sel = st.session_state.model_choice
        models = analysis.get('models', {})
        show_m = models if sel == 'All Models' else {sel: models.get(sel)}
        for name, m in show_m.items():
            if not m: continue
            c = m['color']
            right_html += (
                f'<div class="ds-mc"><div class="ds-mc-name" style="color:{c};">{name}</div>'
                f'<div class="ds-mc-row"><span>Accuracy</span><span style="color:var(--txt);">{m["accuracy"]*100:.1f}%</span></div>'
                f'<div class="ds-mc-bar"><div class="ds-mc-fill" style="width:{m["accuracy"]*100:.1f}%;background:{c};"></div></div>'
                f'<div class="ds-mc-row"><span>AUC-ROC</span><span style="color:var(--txt);">{m["auc"]*100:.1f}%</span></div>'
                f'<div class="ds-mc-bar"><div class="ds-mc-fill" style="width:{m["auc"]*100:.1f}%;background:{c};"></div></div>'
                f'<div class="ds-mc-row"><span>F1</span><span style="color:var(--txt);">{m["f1"]*100:.1f}%</span></div>'
                f'</div>'
            )
    else:
        right_html += '<div style="padding:20px 12px;color:var(--txt3);font-size:12px;">Run analysis to see results.</div>'

    right_html += '</div>'  # close ds-right-analysis
    st.markdown(right_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TERMINAL PANEL — fixed bottom full-width
# ══════════════════════════════════════════════════════════════════
log_html_lines = []
if logs:
    for log in logs:
        c = "var(--txt3)"
        if '[OK]' in log or '[SUCCESS]' in log: c = "#4ec9b0"
        elif 'HIGH' in log: c = "#f85149"
        elif 'MED' in log:  c = "#e2c08d"
        elif any(x in log for x in ['[RUNNING]','[LOADING]','[METRICS]','[ML]']): c = "#4fc3f7"
        elif '[DEFECTSIGHT]' in log: c = "#c586c0"
        esc_log = log.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        log_html_lines.append(f'<span style="color:{c};">{esc_log}</span>')
else:
    log_html_lines.append('<span style="color:var(--txt3);">Terminal ready. Run analysis to see output.</span>')

st.markdown(
    '<div class="ds-terminal-fixed">'
    '  <div class="ds-terminal-hdr"><div class="ds-terminal-tab">⌨ TERMINAL</div></div>'
    '  <div class="ds-terminal-scroll" id="term-scroll">' +
    '<br>'.join(log_html_lines) +
    '  </div>'
    '</div>'
    # Auto-scroll terminal to bottom
    '<script>'
    '(function(){ var el=document.getElementById("term-scroll"); if(el) el.scrollTop=el.scrollHeight; })();'
    '</script>',
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS BAR (HTML only)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="ds-statusbar">'
    f'<div class="ds-sb-item">📐 main</div>'
    f'<div class="ds-sb-sep"></div>'
    f'<div class="ds-sb-item">{nf} files · {tl} LOC</div>'
    f'<div class="ds-sb-sep"></div>'
    f'<div class="ds-sb-item">{st.session_state.model_choice}</div>'
    f'<div class="ds-sb-right">'
    f'<div class="ds-sb-item">{lang_name}</div>'
    f'<div class="ds-sb-sep"></div>'
    f'<div class="ds-sb-item">UTF-8</div>'
    f'<div class="ds-sb-sep"></div>'
    f'<div class="ds-sb-item" style="color:{risk_col};">🛡 {status_txt}</div>'
    f'</div></div>',
    unsafe_allow_html=True
)
