"""Centralized CSS styles for DefectSight frontend."""

import streamlit as st


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
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
        }

        .stApp {
            background:
                radial-gradient(1200px 430px at 16% -7%, #1d2530 0%, rgba(29,37,48,0) 62%),
                radial-gradient(980px 420px at 100% 0%, #172c38 0%, rgba(23,44,56,0) 58%),
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
            background: linear-gradient(130deg, #1d2833, #1f5d72 58%, #255b4f);
            color: #fff;
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

        .ds-code-box {
            border: 1px solid #17222a;
            border-radius: 10px;
            overflow: auto;
            background: #0f1720;
            max-height: 58vh;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.02);
        }

        .vs-cl {
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
        }

        .stButton > button[kind="secondary"] {
            border-color: #38576d;
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
            background: #2c3643;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3e4a59;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
