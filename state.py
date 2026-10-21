"""
state.py -- shared session-state helpers: data loading, parameter defaults, and the
"enriched" dataframe (data + Layer 1 flags + readiness score + any trained model outputs)
that most pages read from.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

import ml_engine as ml
from generate_data import generate_employee_dataset
from rules_engine import apply_layer1_rules

LEVELS = ["Analyst", "Associate", "VP", "Director", "Managing Director"]
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "employee_data.csv")

DEFAULT_L1 = {"l1_analytical": 85, "l1_risk": 90, "l1_leadership": 80, "l1_compliance": 60,
              "l1_engagement": 50, "l1_compa": 0.95, "l1_perf_high": 4.5, "l1_perf_low": 3.0}
DEFAULT_PHASE = "Standard 3-5-Year Horizon (Recommended): 30% / 70%"
DEFAULT_GATE = 65

REQUIRED_COLS = ["Employee_ID", "Full_Name", "Business_Unit", "Job_Level", "Years_of_Experience",
                 "Years_at_Company", "Avg_Performance_Rating", "Number_of_Promotions",
                 "Time_Since_Last_Promotion_Years", "Compa_Ratio", "Feedback_360_Score", *ml.COMPETENCY_COLS]
RESULT_KEYS = ["cluster", "retention", "hipo", "perf", "genai", "models_autotrained"]
REC_RESULT_KEYS = ["rec_models", "rec_scored", "rec_autotrained"]
RECRUIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recruitment_data.csv")


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in optional derived columns (with sensible fallbacks) so older files and
    user uploads that lack them still work."""
    df = df.copy()
    if "Career_Velocity_Score" not in df:
        df["Career_Velocity_Score"] = (df["Number_of_Promotions"] / df["Years_at_Company"].clip(lower=0.5)).round(2)
    if "Performance_Trajectory" not in df:
        df["Performance_Trajectory"] = 0.0
    df["Performance_Trajectory"] = df["Performance_Trajectory"].fillna(0)
    if "Promotions_Last_5_Years" not in df:
        df["Promotions_Last_5_Years"] = np.minimum(df["Number_of_Promotions"], (df["Career_Velocity_Score"] * 5).round()).astype(int)
    if "Promotions_Per_Year_Last_5Y" not in df:
        df["Promotions_Per_Year_Last_5Y"] = (df["Promotions_Last_5_Years"] / 5).round(2)
    for col in ml.NUMERIC_PREDICTORS:           # missing optional predictors -> NaN (imputed later)
        if col not in df:
            df[col] = np.nan
    for col in ("Sub_Function", "High_Potential_Program_Flag", "Attrition_Flag"):
        if col not in df:
            df[col] = None
    if set(df["Job_Level"].dropna().unique()).issubset(LEVELS):
        df["Job_Level"] = pd.Categorical(df["Job_Level"], categories=LEVELS, ordered=True)
    return df


def reset_results():
    for k in RESULT_KEYS:
        st.session_state.pop(k, None)


def set_data(df: pd.DataFrame):
    st.session_state.data = ensure_derived_columns(df)
    reset_results()


def load_sample_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        generate_employee_dataset().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def get_data():
    return st.session_state.get("data")


def l1_params() -> dict:
    g = lambda k: st.session_state.get(k, DEFAULT_L1[k])
    return dict(analytical_thresh=g("l1_analytical"), risk_thresh=g("l1_risk"), leadership_thresh=g("l1_leadership"),
                compliance_watch_thresh=g("l1_compliance"), engagement_low_thresh=g("l1_engagement"),
                compa_low_thresh=g("l1_compa"), perf_high_thresh=g("l1_perf_high"), perf_low_thresh=g("l1_perf_low"))


def readiness_params():
    phase = st.session_state.get("rd_phase", DEFAULT_PHASE)
    return ml.HORIZON_PHASES.get(phase, 0.30), st.session_state.get("rd_gate", DEFAULT_GATE), phase


def enriched() -> pd.DataFrame:
    """Data + Layer 1 flag + readiness score + cluster + any trained model outputs."""
    out = apply_layer1_rules(get_data(), **l1_params())
    perf_w, gate, _ = readiness_params()
    out = ml.compute_readiness(out, perf_w, gate)
    cl = st.session_state.get("cluster")
    if cl:
        out = out.merge(cl["assignments"], on="Employee_ID", how="left")
    for k in ("retention", "hipo", "perf"):
        res = st.session_state.get(k)
        if res:
            out = out.merge(res["scored"], on="Employee_ID", how="left")
    return out


def sticky_container(key: str):
    try:
        return st.container(key=key)
    except TypeError:                      # older Streamlit without container keys
        return st.container()


def pretty(col: str) -> str:
    return col.replace("_Score", "").replace("_", " ")


# ----------------------------------------------------------------------------- recruitment workspace
def get_rec():
    return st.session_state.get("rec_data")


def reset_rec_results():
    for k in REC_RESULT_KEYS:
        st.session_state.pop(k, None)


def set_rec_data(df: pd.DataFrame):
    import recruit_engine as rec
    st.session_state.rec_data = rec.prepare(df)
    reset_rec_results()


def load_sample_recruitment() -> pd.DataFrame:
    if not os.path.exists(RECRUIT_PATH):
        from generate_recruitment_data import generate_recruitment_dataset
        generate_recruitment_dataset().to_csv(RECRUIT_PATH, index=False)
    return pd.read_csv(RECRUIT_PATH)
