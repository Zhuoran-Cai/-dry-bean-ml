"""Streamlit 全局样式。"""

import streamlit as st

PALETTE = {
    "primary": "#1E3A5F",
    "accent": "#2E7D6F",
    "accent_light": "#3D9A8B",
    "surface": "#F4F6F9",
    "card": "#FFFFFF",
    "text": "#1A2332",
    "muted": "#5C6B7A",
    "border": "#E2E8F0",
    "blue": "#4A7BB7",
    "teal": "#5CB8A8",
    "coral": "#E07A5F",
    "purple": "#8B6BB8",
}


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(180deg, #EEF2F7 0%, #F8FAFC 35%, #FFFFFF 100%);
        }}

        /* 避免顶部工具栏遮挡标题 */
        header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(8px);
        }}

        section.main > div {{
            padding-top: 0.5rem;
        }}

        .block-container {{
            padding-top: 4.5rem !important;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        .hero-banner {{
            padding-top: 0.75rem;
            padding-bottom: 0.5rem;
            margin-bottom: 0.5rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PALETTE["primary"]} 0%, #152A45 100%);
        }}

        [data-testid="stSidebar"] * {{
            color: #E8EDF4 !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            font-weight: 500;
            padding: 0.35rem 0;
        }}

        [data-testid="stSidebarNav"] {{
            background: transparent;
        }}

        [data-testid="stSidebarNav"] a {{
            color: #C8D6E8 !important;
            border-radius: 8px;
            margin: 2px 0;
        }}

        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(255, 255, 255, 0.08) !important;
        }}

        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(46, 125, 111, 0.35) !important;
            color: #FFFFFF !important;
            font-weight: 600;
        }}

        h1, h2, h3 {{
            color: {PALETTE["text"]};
            letter-spacing: -0.02em;
        }}

        .hero-title {{
            font-size: 2.4rem;
            font-weight: 700;
            color: {PALETTE["primary"]};
            margin-top: 0;
            margin-bottom: 0.25rem;
            line-height: 1.35;
            padding-top: 0.25rem;
        }}

        .hero-subtitle {{
            font-size: 1.05rem;
            color: {PALETTE["muted"]};
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }}

        .section-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {PALETTE["accent"]};
            margin-bottom: 0.35rem;
        }}

        .section-title {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {PALETTE["text"]};
            margin-bottom: 0.5rem;
        }}

        .section-desc {{
            color: {PALETTE["muted"]};
            font-size: 0.95rem;
            margin-bottom: 1.25rem;
            line-height: 1.55;
        }}

        .kpi-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 4px 18px rgba(30, 58, 95, 0.06);
            height: 100%;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(30, 58, 95, 0.1);
        }}

        .kpi-card .kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            color: {PALETTE["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.5rem;
        }}

        .kpi-card .kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {PALETTE["primary"]};
            line-height: 1.2;
        }}

        .kpi-card .kpi-sub {{
            font-size: 0.82rem;
            color: {PALETTE["accent"]};
            margin-top: 0.4rem;
        }}

        .kpi-accent-blue {{ border-top: 3px solid {PALETTE["blue"]}; }}
        .kpi-accent-teal {{ border-top: 3px solid {PALETTE["teal"]}; }}
        .kpi-accent-coral {{ border-top: 3px solid {PALETTE["coral"]}; }}
        .kpi-accent-purple {{ border-top: 3px solid {PALETTE["purple"]}; }}

        .panel-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 0 4px 18px rgba(30, 58, 95, 0.05);
            margin-bottom: 1rem;
        }}

        .panel-card h4 {{
            margin: 0 0 0.75rem 0;
            font-size: 1rem;
            font-weight: 600;
            color: {PALETTE["text"]};
        }}

        .pipeline {{
            display: flex;
            align-items: stretch;
            gap: 0;
            margin: 1.5rem 0 2rem 0;
            flex-wrap: wrap;
        }}

        .pipeline-step {{
            flex: 1;
            min-width: 140px;
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 12px;
            padding: 1rem 0.85rem;
            text-align: center;
            position: relative;
            box-shadow: 0 2px 10px rgba(30, 58, 95, 0.04);
        }}

        .pipeline-step .step-num {{
            display: inline-block;
            width: 26px;
            height: 26px;
            line-height: 26px;
            border-radius: 50%;
            background: {PALETTE["accent"]};
            color: white;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .pipeline-step .step-title {{
            font-size: 0.88rem;
            font-weight: 600;
            color: {PALETTE["text"]};
            margin-bottom: 0.25rem;
        }}

        .pipeline-step .step-desc {{
            font-size: 0.72rem;
            color: {PALETTE["muted"]};
            line-height: 1.4;
        }}

        .pipeline-arrow {{
            display: flex;
            align-items: center;
            color: {PALETTE["accent"]};
            font-size: 1.2rem;
            padding: 0 0.35rem;
            font-weight: 700;
        }}

        .tag {{
            display: inline-block;
            background: rgba(46, 125, 111, 0.1);
            color: {PALETTE["accent"]};
            border-radius: 6px;
            padding: 0.2rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 500;
            margin: 0.15rem 0.2rem 0.15rem 0;
        }}

        .info-box {{
            background: rgba(74, 123, 183, 0.08);
            border-left: 4px solid {PALETTE["blue"]};
            border-radius: 0 10px 10px 0;
            padding: 0.9rem 1.1rem;
            color: {PALETTE["text"]};
            font-size: 0.9rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }}

        .figure-caption {{
            text-align: center;
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
            margin-top: 0.5rem;
            font-style: italic;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {PALETTE["border"]};
            border-radius: 10px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_page(title: str, icon: str = "🫘") -> None:
    st.set_page_config(
        page_title=f"{title} · 干豆多分类",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_styles()
