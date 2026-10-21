"""
app.py -- Candidate Development & Succession Planning Portal (Streamlit)

Entry point: page setup, left navigation and routing only. The pages themselves live in
pages_core.py and pages_advanced.py; shared helpers in state.py and styling.py.

Run with:   streamlit run app.py
"""
import os

import streamlit as st

# Friendly check: the app is made of several files that must sit together in ONE folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REQUIRED_FILES = ["styling.py", "state.py", "rules_engine.py", "ml_engine.py", "catalogue.py", "genai.py", "generate_data.py",
                   "pages_core.py", "pages_advanced.py", "pages_recruit.py", "recruit_engine.py", "generate_recruitment_data.py"]
_missing = [f for f in _REQUIRED_FILES if not os.path.exists(os.path.join(_HERE, f))]
if _missing:
    st.set_page_config(page_title="Setup needed", page_icon="\u26A0\uFE0F")
    st.error("Some app files are missing from this folder: " + ", ".join(_missing))
    st.write(f"Folder being used: `{_HERE}`")
    st.write("Copy ALL the files from the delivered package into this one folder (no sub-folders), then refresh this page.")
    st.stop()

from streamlit_option_menu import option_menu  # noqa: E402

st.set_page_config(page_title="Candidate Development & Succession Planning Portal", page_icon="\U0001F3E6",
                   layout="wide", initial_sidebar_state="expanded")

from styling import inject_css                                  # noqa: E402  (after set_page_config)
from pages_core import (render_layer1_page, render_layer2_page, render_overview_page,   # noqa: E402
                        render_upload_page)
from pages_advanced import (render_career_page, render_genai_page, render_glossary_page,  # noqa: E402
                            render_reports_page, render_rx_page, render_succession_page)
from pages_recruit import (render_fairness_page, render_funnel_page, render_insights_page,  # noqa: E402
                           render_recruit_data_page, render_scoring_page)

inject_css()

if "data" not in st.session_state:          # nothing is auto-loaded: upload a file or load the sample
    st.session_state.data = None

# Streamlit forgets a widget's value when its page isn't on screen. Re-saving these keys on every
# run keeps slider/dropdown choices (rule thresholds, gate, algorithm...) when you switch pages.
_PERSIST = ("l1_", "rd_", "ml_", "k_clusters", "rc_")
_PERSIST_SKIP = {"rc_upload", "rc_apply", "rc_sample", "rc_train"}

for _k in list(st.session_state.keys()):
    if isinstance(_k, str) and _k.startswith(_PERSIST) and _k not in _PERSIST_SKIP:
        st.session_state[_k] = st.session_state[_k]

SUCCESSION_PAGES = {
    "Upload Data": ("cloud-upload", render_upload_page),
    "Executive Overview": ("speedometer2", render_overview_page),
    "Layer 1: Rule Engine": ("clipboard2-check", render_layer1_page),
    "Layer 2: Predictive ML": ("cpu", render_layer2_page),
    "Career Trajectory": ("film", render_career_page),
    "Succession Matrix": ("grid-3x3", render_succession_page),
    "Development Rx": ("capsule", render_rx_page),
    "Layer 3: AI Report": ("robot", render_genai_page),
    "Reports & Downloads": ("download", render_reports_page),
    "Glossary": ("book", render_glossary_page),
}
RECRUITMENT_PAGES = {
    "Recruitment Data": ("cloud-upload", render_recruit_data_page),
    "Hiring Funnel & Sources": ("funnel", render_funnel_page),
    "Hogan & PI Insights": ("person-lines-fill", render_insights_page),
    "Candidate Scoring": ("bullseye", render_scoring_page),
    "Model Quality & Fairness": ("shield-check", render_fairness_page),
    "Glossary ": ("book", render_glossary_page),          # trailing space keeps the menu key unique
}
WORKSPACES = {"Succession & Talent": SUCCESSION_PAGES, "Recruitment": RECRUITMENT_PAGES}

with st.sidebar:
    st.markdown("<div style='padding:10px 4px 10px 4px;'><div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.0rem;color:#fff;'>"
                "\U0001F3E6 Talent Portal</div><div style='font-size:0.68rem;color:#9EC5D8;margin-top:2px;'>"
                "IB &middot; Risk &middot; Wealth &middot; Quant Trading</div></div>", unsafe_allow_html=True)
    workspace = st.radio("Workspace", list(WORKSPACES), key="workspace", horizontal=True, label_visibility="collapsed")
    PAGES = WORKSPACES[workspace]
    page = option_menu(None, list(PAGES), icons=[v[0] for v in PAGES.values()], default_index=0, key=f"menu_{workspace}",
                       styles={"container": {"padding": "0", "background-color": "transparent"},
                               "icon": {"font-size": "14px"},
                               "nav-link": {"font-size": "13px", "font-family": "Syne, sans-serif", "font-weight": "600",
                                            "padding": "9px 12px", "border-radius": "8px"},
                               "nav-link-selected": {"background-color": "#0D7377"}})
    if workspace == "Succession & Talent" and st.session_state.data is not None:
        st.caption(f"Employee dataset: {len(st.session_state.data):,} rows")
    if workspace == "Recruitment" and st.session_state.get("rec_data") is not None:
        st.caption(f"Recruitment dataset: {len(st.session_state.rec_data):,} candidates")
    st.markdown("<div style='font-size:0.62rem;color:#5E85A3;margin-top:24px;'>Synthetic Data | For Demonstration Purposes Only</div>",
                unsafe_allow_html=True)

PAGES[page][1]()
