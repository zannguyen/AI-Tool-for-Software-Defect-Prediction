"""Centralized CSS styles for DefectSight frontend."""

import streamlit as st


def apply_global_styles(theme: str = "dark") -> None:
    if theme == "light":
        css_vars = """
            --bg: #f3f6fa;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --ink: #1e293b;
            --ink-muted: #64748b;
            --line: #e2e8f0;
            --brand: #0f766e;
            --brand-soft: #ccfbf1;
            --danger: #dc2626;
            --warn: #d97706;
            --safe: #16a34a;
            
            --bg-grad-1: #e2e8f0;
            --bg-grad-2: #cbd5e1;
            --topbar-bg: linear-gradient(130deg, #0f766e, #14b8a6 58%, #2dd4bf);
            --topbar-text: #ffffff;
            
            --scroll-thumb: #cbd5e1;
            --scroll-thumb-hover: #94a3b8;
        """
    else:
        css_vars = """
            --bg: #0f1318;
            --panel: #171d24;
            --panel-soft: #1f2731;
            --ink: #e6edf6;
            --ink-muted: #9aabbe;
            --line: #2c3643;
            --brand: #2f9d8a;
            --brand-soft: #173f39;
            --danger: #b8323d;
            --warn: #ad7b2b;
            --safe: #2e6f44;
            
            --bg-grad-1: #1d2530;
            --bg-grad-2: #172c38;
            --topbar-bg: linear-gradient(130deg, #1d2833, #1f5d72 58%, #255b4f);
            --topbar-text: #ffffff;
            
            --scroll-thumb: #2c3643;
            --scroll-thumb-hover: #3e4a59;
        """

    base_css = """
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        .stApp {
            background:
                radial-gradient(1200px 430px at 16% -7%, var(--bg-grad-1) 0%, transparent 62%),
                radial-gradient(980px 420px at 100% 0%, var(--bg-grad-2) 0%, transparent 58%),
                var(--bg);
            color: var(--ink);
            font-family: 'Manrope', sans-serif;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebar"] {
            display: none;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 0.9rem;
            max-width: 100%;
        }

        .ds-topbar {
            background: var(--topbar-bg);
            color: var(--topbar-text);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
            margin-bottom: 12px;
            animation: ds-fade-in 0.45s ease-out;
        }

        .ds-topbar-title {
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: .02em;
            margin: 0;
        }

        .ds-topbar-sub {
            margin: 4px 0 0;
            font-size: 0.86rem;
            opacity: .92;
        }

        .ds-pane-title {
            margin: 0 0 10px;
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--ink-muted);
            font-weight: 700;
        }

        .ds-file-header {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }

        .ds-file-name {
            margin: 0;
            color: var(--ink);
            font-size: .98rem;
            font-weight: 700;
            word-break: break-all;
        }

        .ds-file-sub {
            margin: 4px 0 0;
            color: var(--ink-muted);
            font-size: .82rem;
        }

        .ds-metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 8px;
        }

        .ds-metric {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 8px;
            text-align: center;
        }

        .ds-metric-v {
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--ink);
            line-height: 1.2;
        }

        .ds-metric-l {
            font-size: .68rem;
            text-transform: uppercase;
            letter-spacing: .07em;
            color: var(--ink-muted);
            margin-top: 2px;
        }

        /* Note: Always keeping code box dark for perfect syntax highlighting visibility */
        .ds-code-box {
            border: 1px solid #17222a;
            border-radius: 10px;
            overflow: auto;
            background: #0f1720;
            max-height: 58vh;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.02);
            color: #d4dde8;
        }

        .ds-code-fragment {
            background: #0f1720;
            color: #d4dde8;
            padding: 4px 0;
            border-left: 1px solid #17222a;
            border-right: 1px solid #17222a;
        }
        .ds-code-fragment.top {
            border-top: 1px solid #17222a;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding-top: 8px;
        }
        .ds-code-fragment.bottom {
            border-bottom: 1px solid #17222a;
            border-bottom-left-radius: 10px;
            border-bottom-right-radius: 10px;
            padding-bottom: 8px;
        }
        .ds-code-fragment.single {
            border: 1px solid #17222a;
            border-radius: 10px;
            padding: 8px 0;
        }

        .vs-cl {
            position: relative;
            display: flex;
            align-items: flex-start;
            min-height: 21px;
            font-family: 'JetBrains Mono', monospace;
            font-size: .8rem;
            line-height: 1.45;
        }

        .vs-cl:hover {
            background: rgba(255,255,255,.03);
        }

        .vs-cl.hi {
            background: rgba(184,50,61,.14);
            border-left: 3px solid rgba(184,50,61,.78);
        }

        .vs-cl.me {
            background: rgba(173,123,43,.15);
            border-left: 3px solid rgba(173,123,43,.74);
        }

        .vs-ln {
            width: 52px;
            flex-shrink: 0;
            color: #7c8a98;
            text-align: right;
            padding: 1px 11px 0 8px;
            user-select: none;
            font-size: .75rem;
        }

        .vs-code-txt {
            color: #d4dde8;
            white-space: pre;
            overflow-x: auto;
            flex: 1;
            padding-top: 1px;
        }

        .vs-rb {
            font-size: .58rem;
            margin: 2px 6px 0 8px;
            border-radius: 999px;
            padding: 1px 7px;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .vs-rb.hi {
            color: #ffd2d6;
            border: 1px solid rgba(255,114,128,.58);
            background: rgba(184,50,61,.38);
        }

        .vs-rb.me {
            color: #fde8bf;
            border: 1px solid rgba(255,190,92,.58);
            background: rgba(173,123,43,.33);
        }

        .kw { color: #57a9ff; }
        .fn { color: #f7d779; }
        .cls { color: #69d8bd; }
        .str { color: #ec9f78; }
        .num { color: #bde38a; }
        .dec { color: #8dc5ff; }

        .vs-inline-err {
            padding: 4px 10px 6px 60px;
            font-size: 0.75rem;
            font-family: inherit;
            font-style: italic;
            border-bottom: 1px solid rgba(255,255,255,0.02);
            line-height: 1.3;
            word-wrap: break-word;
            white-space: pre-wrap;
        }

        .vs-inline-err.hi {
            background-color: rgba(184,50,61,0.08);
            border-left: 3px solid rgba(184,50,61,0.78);
            color: #ffadb3;
        }

        .vs-inline-err.me {
            background-color: rgba(173,123,43,0.05);
            border-left: 3px solid rgba(173,123,43,0.74);
            color: #ffdca1;
        }

        .ds-tab-pill {
            background: #202833;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink-muted);
            padding: 4px 10px;
            font-size: .74rem;
            white-space: nowrap;
        }

        .ds-tab-pill.active {
            background: var(--brand-soft);
            color: #7fd8c8;
            border-color: #26645a;
            font-weight: 700;
        }

        /* Note: Also keeping log terminal dark */
        .ds-log {
            background: #131a1f;
            color: #c5d0da;
            border-radius: 10px;
            border: 1px solid #263441;
            font-family: 'JetBrains Mono', monospace;
            font-size: .74rem;
            line-height: 1.45;
            padding: 10px;
            max-height: 170px;
            overflow-y: auto;
        }

        .stButton > button {
            border-radius: 9px;
        }

        .stButton > button[kind="primary"] {
            background-color: var(--brand);
            border-color: var(--brand);
            color: white;
        }

        .stButton > button[kind="secondary"] {
            border-color: var(--line);
            color: var(--ink);
        }

        @keyframes ds-fade-in {
            from { opacity: 0; transform: translateY(-6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes ds-rise {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 960px) {
            .ds-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .ds-code-box {
                max-height: 46vh;
            }
        }

        /* Tối ưu hóa thanh cuộn cho container */
        [data-testid="stVerticalBlock"] > div > div > [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }
        
        /* Đảm bảo ds-card chiếm toàn bộ chiều cao trong container nếu cần */
        .ds-card {
            height: 100%;
            margin-bottom: 0px; 
        }

        /* Tùy chỉnh thanh cuộn cho chuyên nghiệp hơn */
        ::-webkit-scrollbar {
            width: 5px;
            height: 5px;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--scroll-thumb);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--scroll-thumb-hover);
        }
    """
    
    st.markdown(
        "<style>\n:root {\n" + css_vars + "\n}\n" + base_css + "\n</style>", 
        unsafe_allow_html=True
    )

