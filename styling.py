"""
styling.py

Design system shared across the whole app: navy/teal/gold finance palette,
Syne (headings) + DM Sans (body) fonts, and small HTML-returning helper
functions (KPI cards, avatars, pill badges, successor cards) used throughout
the dashboard. Centralising these here means every page renders consistently
without re-deriving the same CSS/HTML snippets.
"""

import inspect
from contextlib import contextmanager

import streamlit as st

NAVY = "#0B2540"
TEAL = "#0D7377"
GOLD = "#C9A227"
GOLD_LT = "#F0C93A"
BG = "#F0F4F8"
CARD = "#FFFFFF"
BORDER = "#D1DCE8"
TEXT = "#1A2535"
MUTED = "#64748B"

CLUSTER_COLORS = {
    "High-Flyers": GOLD, "Steady Performers": TEAL, "Development Core": MUTED,
    "Emerging Talent": "#2563EB", "Foundational Track": "#7C3AED",
}

SIGNAL_COLORS = {
    "Retention Risk": "#B91C1C", "Hidden HiPo": GOLD, "Pay Equity Risk": "#D97706",
    "Blocked Potential": "#7C3AED", "Promotion Overdue": "#2563EB",
    "Untapped Potential": "#1B7A3E", "Derailment Watch": "#9333EA",
}


def inject_css():
    css = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background-color: {BG}; }}
h1,h2,h3,h4,h5,h6 {{ font-family: 'Syne', sans-serif !important; color: {NAVY}; font-weight: 700; }}
h1 {{ font-size: 1.4rem !important; }}
h2 {{ font-size: 1.15rem !important; }}
h3 {{ font-size: 1.0rem !important; }}
h4 {{ font-size: 0.92rem !important; }}
.app-header {{ background: linear-gradient(135deg, #0B2540 0%, #163A5C 60%, #0D7377 100%); padding: 18px 26px; border-radius: 14px; color: white; margin-bottom: 18px; }}
.app-header h1 {{ color: #fff !important; margin: 0; font-size: 1.3rem !important; }}
.app-header p {{ color: #9EC5D8; margin: 4px 0 0 0; font-size: 0.8rem; }}
.kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }}
.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 20px; flex: 1; min-width: 150px; box-shadow: 0 1px 6px rgba(11,37,64,0.06); }}
.kpi-value {{ font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800; color: {TEAL}; line-height: 1.1; }}
.kpi-label {{ font-size: 0.68rem; color: {MUTED}; margin-top: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-radius: 14px; box-shadow: 0 2px 12px rgba(11,37,64,0.06); }}
.card-header {{ font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.92rem; color: {NAVY}; border-bottom: 2px solid {TEAL}; padding-bottom: 8px; margin-bottom: 12px; }}
.scard {{ border-radius: 14px; padding: 14px 18px; margin-bottom: 8px; border-left: 5px solid {TEAL}; background: {CARD}; box-shadow: 0 2px 10px rgba(11,37,64,0.07); position: relative; }}
.scard-rank {{ position: absolute; top: 10px; right: 12px; font-family: 'Syne', sans-serif; font-size: 0.6rem; font-weight: 800; background: {NAVY}; color: {GOLD_LT}; border-radius: 20px; padding: 3px 10px; letter-spacing: 1px; text-transform: uppercase; }}
.scard h5 {{ font-size: 0.86rem !important; font-weight: 700; margin: 0 0 2px 0; color: {NAVY}; }}
.scard p {{ font-size: 0.78rem; color: {MUTED}; margin: 2px 0 0 0; }}
.rx-card {{ border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; border-left: 5px solid {GOLD}; background: #FFFDF5; box-shadow: 0 1px 6px rgba(11,37,64,0.05); }}
.rx-card h5 {{ font-size: 0.84rem !important; color: {NAVY}; margin: 0 0 3px 0; }}
.rx-card p {{ font-size: 0.78rem; color: {MUTED}; margin: 0; }}
.band-pill {{ display: inline-block; border-radius: 20px; padding: 3px 10px; font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }}
.empty-state {{ background: {CARD}; border: 2px dashed {BORDER}; border-radius: 14px; padding: 40px 30px; text-align: center; color: {MUTED}; margin-bottom: 16px; }}
.empty-state h4 {{ color: {NAVY}; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0B2540 0%, #10314E 100%); }}
section[data-testid="stSidebar"] * {{ color: #E8F0F7 !important; }}
.stButton>button, .stDownloadButton>button {{ background-color: {TEAL}; color: white; border: none; border-radius: 8px; font-weight: 600; font-size: 0.85rem; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background-color: #0a5c60; color: white; }}
[data-testid="stDataFrame"] {{ font-size: 0.82rem; }}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


def avatar_html(name: str, size: int = 44, bg: str = TEAL) -> str:
    parts = str(name).strip().split()
    initials = (parts[0][0] if parts else "?") + (parts[1][0] if len(parts) > 1 else "")
    return (f"<div style='width:{size}px;height:{size}px;border-radius:50%;background:{bg};"
            f"color:white;display:flex;align-items:center;justify-content:center;"
            f"font-family:Syne,sans-serif;font-weight:800;font-size:{size // 3}px;"
            f"flex-shrink:0;'>{initials.upper()}</div>")


def band_pill(label: str, color: str) -> str:
    return f"<span class='band-pill' style='background:{color}20;color:{color}'>{label}</span>"


def kpi_card(value, label: str, color: str = TEAL) -> str:
    return (f"<div class='kpi'><div class='kpi-value' style='color:{color}'>{value}</div>"
            f"<div class='kpi-label'>{label}</div></div>")


def kpi_row(*cards: str):
    st.markdown(f"<div class='kpi-row'>{''.join(cards)}</div>", unsafe_allow_html=True)


def empty_state(msg: str = "Upload a dataset (or load the sample dataset) on the Upload Data page to unlock this view."):
    st.markdown(
        "<div class='empty-state'><div style='font-size:2.2rem;'>\U0001F4C2</div>"
        f"<h4>No Dataset Loaded</h4><p>{msg}</p></div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str):
    st.markdown(f"<div class='app-header'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


@contextmanager
def card(title: str = None):
    """A real bordered container (everything inside the `with` block is
    actually inside the card -- the old open/close-div approach never was)."""
    with st.container(border=True):
        if title:
            st.markdown(f"<div class='card-header'>{title}</div>", unsafe_allow_html=True)
        yield


# Streamlit renamed use_container_width -> width="stretch" in newer versions.
# Pick whichever the installed version understands so the app runs on both.
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}


def show_plot(fig, **kw):
    st.plotly_chart(fig, **STRETCH, **kw)


def show_table(df, **kw):
    st.dataframe(df, **STRETCH, **kw)
