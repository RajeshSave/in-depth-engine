"""
test_app.py -- automated tests. Run with:  python3 test_app.py
Runs every page inside a simulated Streamlit session (no browser needed), clicks buttons, and
checks that (a) nothing raises and (b) no HTML/CSS is accidentally shown as a code block.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

MD = MarkdownIt("commonmark", {"html": True})
PASSED = []


RECRUIT_FUNCS = ("render_recruit_data_page", "render_funnel_page", "render_insights_page", "render_scoring_page", "render_fairness_page")


def page_script(func):
    if func in RECRUIT_FUNCS:
        return f"""
import sys; sys.path.insert(0, r"{HERE}")
import streamlit as st
from styling import inject_css
from state import set_rec_data, load_sample_recruitment
inject_css()
if st.session_state.get("rec_data") is None:
    set_rec_data(load_sample_recruitment())
from pages_recruit import {func}
{func}()
"""
    mod = "pages_advanced" if func in ("render_career_page", "render_succession_page", "render_rx_page", "render_genai_page",
                                       "render_reports_page", "render_glossary_page") else "pages_core"
    return f"""
import sys; sys.path.insert(0, r"{HERE}")
import streamlit as st
from styling import inject_css
from state import set_data, load_sample_dataset
inject_css()
if st.session_state.get("data") is None:
    set_data(load_sample_dataset())
from {mod} import {func}
{func}()
"""


def new_page(func, timeout=120):
    at = AppTest.from_string(page_script(func), default_timeout=timeout)
    at.run()
    return at


def check(at, name):
    assert not at.exception, f"{name}: exception -> {[e.value for e in at.exception]}"
    for m in at.markdown:
        assert "<pre><code>" not in MD.render(m.value), f"{name}: markdown rendered as a CODE BLOCK: {m.value[:80]!r}"
    PASSED.append(name)
    print("PASS:", name)


# ----------------------------------------------------------------------------- full app
def test_full_app_and_upload_page():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    check(at, "full app loads (Upload page, no data)")
    at.button(key="load_sample").click().run()
    assert at.session_state["data"] is not None and len(at.session_state["data"]) == 520
    check(at, "Load Sample Demo Dataset -> 520 rows")


# ----------------------------------------------------------------------------- pages
def test_pages_render():
    for func in ["render_overview_page", "render_layer1_page", "render_layer2_page", "render_career_page",
                 "render_succession_page", "render_rx_page", "render_genai_page", "render_reports_page", "render_glossary_page"]:
        check(new_page(func), func)


def test_pages_without_data_show_empty_state():
    at = AppTest.from_string("""
import sys; sys.path.insert(0, r"%s")
import streamlit as st
from pages_core import render_overview_page
st.session_state.data = None
render_overview_page()
""" % HERE, default_timeout=30)
    at.run()
    check(at, "no-data empty state")


# ----------------------------------------------------------------------------- behaviour
def test_layer1_sliders_move_the_chart():
    at = new_page("render_layer1_page")
    base = at.session_state["l1_analytical"]
    at.slider(key="l1_analytical").set_value(55).run()
    at.slider(key="l1_risk").set_value(55).run()
    at.slider(key="l1_leadership").set_value(55).run()
    check(at, "Layer 1 sliders accept changes")
    from rules_engine import apply_layer1_rules
    import pandas as pd
    df = pd.read_csv("employee_data.csv")
    n_loose = (apply_layer1_rules(df, analytical_thresh=55, risk_thresh=55, leadership_thresh=55).Layer1_Flag == "Ready for Leadership Track").sum()
    n_tight = (apply_layer1_rules(df, analytical_thresh=95, risk_thresh=95, leadership_thresh=95).Layer1_Flag == "Ready for Leadership Track").sum()
    assert n_loose > n_tight, "Layer 1 thresholds are not responsive"
    n_a = (apply_layer1_rules(df, compliance_watch_thresh=85).Layer1_Flag == "Compliance Watchlist").sum()
    n_b = (apply_layer1_rules(df, compliance_watch_thresh=30).Layer1_Flag == "Compliance Watchlist").sum()
    assert n_a > n_b, "Compliance threshold is not responsive"
    print(f"PASS: rule thresholds are responsive (ready: {n_tight}->{n_loose}, compliance: {n_b}->{n_a})")


def test_layer2_trains_and_retrains():
    at = new_page("render_layer2_page")
    assert at.session_state["retention"] and at.session_state["hipo"] and at.session_state["perf"], "models were not auto-trained"
    assert at.session_state["cluster"], "clustering did not auto-run"
    print("PASS: Layer 2 auto-trains 3 models + clustering on first visit")
    at.radio(key="ml_method").set_value("rf").run()
    at.button(key="train_btn").click().run()
    check(at, "Layer 2 retrain with Random Forest")
    assert at.session_state["retention"]["method"] == "rf"
    at.slider(key="k_clusters").set_value(4).run()
    at.button(key="cluster_btn").click().run()
    check(at, "Layer 2 re-cluster with k=4")
    assert at.session_state["cluster"]["assignments"]["Talent_Cluster"].nunique() == 4
    at.checkbox(key="show_tech").check().run()
    check(at, "Layer 2 technical detail (3D PCA) shown")


def test_succession_controls():
    at = new_page("render_succession_page")
    at.slider(key="rd_gate").set_value(80).run()
    check(at, "Succession gate = 80")
    phases = list(at.selectbox(key="rd_phase").options)
    at.selectbox(key="rd_phase").select(phases[0]).run()
    check(at, "Succession phase switch")


def test_genai_fallback():
    at = new_page("render_genai_page")
    at.button(key="ai_go").click().run()
    check(at, "AI report generated (offline fallback)")
    g = at.session_state["genai"]
    assert g["source"].startswith("Local") and "EXECUTIVE SUMMARY" in g["text"]
    at.text_input(key="ai_key").set_value("not-a-real-key").run()
    at.selectbox(key="ai_provider").select("Anthropic Claude").run()
    at.button(key="ai_go").click().run()
    check(at, "AI report with a bad key falls back gracefully")
    assert at.session_state["genai"]["text"], "no report text after failed live call"


def test_rx_catalogue():
    from catalogue import DEFAULT_CATALOGUE, prescribe
    import pandas as pd
    assert len(DEFAULT_CATALOGUE) == 21
    df = pd.read_csv("employee_data.csv")
    plan = prescribe(df.iloc[3], DEFAULT_CATALOGUE)
    assert len(plan) == 7 and plan["Intervention"].notna().all()
    edited = DEFAULT_CATALOGUE.copy()
    edited.loc[edited.Competency.eq(plan.iloc[0]["Competency"]) & edited.Band.eq(plan.iloc[0]["Band"]), "Intervention"] = "MY CUSTOM PROGRAMME"
    assert "MY CUSTOM PROGRAMME" in prescribe(df.iloc[3], edited)["Intervention"].tolist()
    print("PASS: catalogue prescribes for all 7 competencies and honours edits")


def test_career_animation_figure():
    import pandas as pd
    from pages_advanced import trajectory_figure
    df = pd.read_csv("employee_data.csv")
    ok = 0
    for _, row in df.head(60).iterrows():
        fig = trajectory_figure(row, 1.0)
        if fig is not None:
            assert len(fig.frames) >= 2 and len(fig.data) == 2
            ok += 1
    assert ok > 30
    print(f"PASS: animated trajectory figure builds ({ok}/60 employees have 2+ years of history)")


def test_ml_quality():
    import pandas as pd
    import ml_engine as ml
    df = pd.read_csv("employee_data.csv")
    r, h, p = ml.train_retention_model(df, "glm"), ml.train_hipo_model(df, "glm"), ml.train_performance_model(df, "glm")
    assert r["metrics"]["AUC"] > 0.62 and h["metrics"]["AUC"] > 0.62 and p["metrics"]["R2"] > 0.3
    assert (r["scored"]["Retention_Risk_Tier"] == "High").sum() > 20
    print(f"PASS: models learn real signal (AUC retention {r['metrics']['AUC']:.2f}, HiPo {h['metrics']['AUC']:.2f}, perf R2 {p['metrics']['R2']:.2f})")



def test_every_sidebar_page_through_the_real_router():
    """Runs the real app.py for each sidebar entry (the option_menu widget itself can't be clicked
    in a test, so it is swapped for a function that returns the page we want)."""
    pages = ["Upload Data", "Executive Overview", "Layer 1: Rule Engine", "Layer 2: Predictive ML", "Career Trajectory",
             "Succession Matrix", "Development Rx", "Layer 3: AI Report", "Reports & Downloads", "Glossary"]
    for p in pages:
        script = f"""
import sys; sys.path.insert(0, r"{HERE}")
import streamlit as st
import streamlit_option_menu as som
som.option_menu = lambda *a, **k: {p!r}
from state import set_data, load_sample_dataset
if st.session_state.get("data") is None:
    set_data(load_sample_dataset())
exec(open(r"{HERE}/app.py").read())
"""
        at = AppTest.from_string(script, default_timeout=120)
        at.run()
        check(at, f"router -> {p}")


def test_settings_survive_page_switches():
    """Change a Layer 1 slider, 'navigate' to another page and back: the value must stick."""
    def run(page, at=None):
        script = f"""
import sys; sys.path.insert(0, r"{HERE}")
import streamlit as st
import streamlit_option_menu as som
som.option_menu = lambda *a, **k: st.session_state.get("_page", "Layer 1: Rule Engine")
from state import set_data, load_sample_dataset
if st.session_state.get("data") is None:
    set_data(load_sample_dataset())
exec(open(r"{HERE}/app.py").read())
"""
        return script
    at = AppTest.from_string(run("Layer 1: Rule Engine"), default_timeout=120)
    at.run()
    at.slider(key="l1_analytical").set_value(61).run()
    at.session_state["_page"] = "Glossary"
    at.run()
    at.session_state["_page"] = "Layer 1: Rule Engine"
    at.run()
    assert at.slider(key="l1_analytical").value == 61, "slider value was lost when switching pages"
    check(at, "Layer 1 threshold survives navigating away and back")


# ============================================================================ RECRUITMENT
def test_recruitment_data_is_sound():
    import pandas as pd
    df = pd.read_csv("recruitment_data.csv", parse_dates=["Application_Date", "Screen_Date", "Assessment_Date", "Interview_Date", "Offer_Date", "Decision_Date", "Join_Date"])
    j = df[df.Joined == "Y"]
    assert len(df) == 12000 and len(j) > 1000 and (df.Outcome_Observed == "Y").sum() > 800
    assert (j.Join_Date <= pd.Timestamp("2026-09-30")).all()
    assert df.loc[df.Outcome_Observed == "Y", "Joined"].eq("Y").all(), "outcomes recorded for non-hires"
    assert df.loc[df.Joined != "Y", "Perf_Rating_12M"].isna().all()
    assert df.loc[df.Assessment_Date.isna(), "HPI_Adjustment"].isna().all(), "assessment scores exist before the assessment happened"
    assert df.loc[df.Offer_Date.isna(), "Offer_Comp_USD"].isna().all(), "offer terms exist before an offer"
    assert df.loc[df.Interview_Date.isna(), "Interview_Panel_Score"].isna().all()
    for a, b in (("Application_Date", "Screen_Date"), ("Screen_Date", "Assessment_Date"), ("Assessment_Date", "Interview_Date"), ("Interview_Date", "Offer_Date")):
        x = df.dropna(subset=[a, b]); assert (x[b] >= x[a]).all(), f"{b} before {a}"
    hpi = [c for c in df.columns if c.startswith(("HPI_", "HDS_", "MVPI_"))]
    assert df[hpi].stack().dropna().between(1, 99).all()
    assert df.Funnel_Status.str.contains("In Process|Awaiting|Pending").sum() > 100, "no live pipeline"
    print("PASS: recruitment dataset is internally consistent (no leakage across funnel stages, live pipeline present)")


def test_recruit_models_and_backtest():
    import pandas as pd
    import recruit_engine as re_
    df = pd.read_csv("recruitment_data.csv")
    m = re_.train_recruit_models(df, "glm")
    assert not m["errors"], m["errors"]
    assert m["perf"]["metrics"]["R2 (cross-validated)"] > 0.05
    assert m["ret"]["metrics"]["AUC (cross-validated)"] > 0.60 and m["acc"]["metrics"]["AUC (cross-validated)"] > 0.60
    for key in ("perf", "ret", "acc"):
        feats = set(m[key]["numeric"]) | set(re_.CAT)
        assert not feats & set(re_.PROTECTED), f"{key} model uses a protected attribute"
    bt = re_.backtest(m, df)
    model = bt[bt["Ranked by"].str.startswith("Model")].set_index("Group")
    assert model.loc["Top 25%", "12M Retention"] > model.loc["Bottom 25%", "12M Retention"] + 0.10
    assert model.loc["Top 25%", "Avg Performance"] > model.loc["Bottom 25%", "Avg Performance"] + 0.25
    print(f"PASS: models learn real signal and the out-of-fold backtest separates outcomes "
          f"(retention {model.loc['Bottom 25%', '12M Retention']:.0%} -> {model.loc['Top 25%', '12M Retention']:.0%}); no protected attributes used")
    sc = re_.score_candidates(m, df)
    assert set(sc.Recommendation.unique()) >= {"Strong Hire", "Hire", "Review", "Not Recommended"}
    pipe = sc[sc.Funnel_Status.str.startswith(("In Process", "Offer Extended", "Offer Accepted - Pending"))]
    assert len(pipe) > 100 and pipe.QoH_Pred.notna().all()
    assert not (sc[sc.Interview_Panel_Score.isna() & ~sc.Funnel_Status.str.startswith("In Process")].Recommendation.eq("Awaiting interviews")).any(), "rejected candidates labelled 'Awaiting interviews'"
    r = pipe.iloc[0]
    ex = re_.explain_candidate(m, r)
    assert len(ex) == 2 and all(len(t) == 5 for _, t in ex.values())
    lo = re_.whatif_offer(m, r, ratio=0.90)["Accept_Prob"]; hi = re_.whatif_offer(m, r, ratio=1.20)["Accept_Prob"]
    assert hi > lo, "a richer offer should raise acceptance likelihood"
    print(f"PASS: scoring, explanations and what-if work (acceptance {lo:.0%} at 0.90x -> {hi:.0%} at 1.20x offer)")


def test_fairness_monitor_detects_injected_disparity():
    import pandas as pd
    import recruit_engine as re_
    d = re_.prepare(pd.read_csv("recruitment_data.csv"))
    ai = re_.adverse_impact(d, "Age_Band")
    row = ai[(ai["Funnel step"] == "Interviewed -> Offered") & (ai["Age_Band"] == "40+")].iloc[0]
    assert row["Four-fifths check"] == "Below 0.80", row
    g = re_.adverse_impact(d, "Gender")
    assert (g["Four-fifths check"] != "Below 0.80").all(), "gender should show no disparity in the synthetic data"
    print(f"PASS: fairness monitor flags the injected age disparity (ratio {row['Ratio vs highest group']}) and stays quiet for gender")


def test_recruitment_pages_render():
    for func in RECRUIT_FUNCS:
        check(new_page(func, timeout=180), func)


def test_scoring_page_interactions():
    at = new_page("render_scoring_page", timeout=180)
    assert at.session_state["rec_models"]["perf"] and at.session_state["rec_scored"] is not None
    print("PASS: scoring page auto-trains 3 models on first visit")
    sc = at.session_state["rec_scored"]
    pipe_ids = sc[sc.Funnel_Status.str.startswith(("In Process", "Offer Extended", "Offer Accepted - Pending"))].sort_values("QoH_Pred", ascending=False)["Candidate_ID"].tolist()
    at.selectbox(key="rc_s_cand").set_value(pipe_ids[3]).run()
    check(at, "scoring page: select another candidate")
    at.radio(key="rc_method").set_value("rf").run()
    at.button(key="rc_train").click().run()
    check(at, "scoring page: retrain with Random Forest")
    assert at.session_state["rec_models"]["method"] == "rf"
    at.multiselect(key="rc_s_rec").set_value(["Strong Hire"]).run()
    check(at, "scoring page: filter to Strong Hire")


def test_insights_and_funnel_interactions():
    at = new_page("render_insights_page", timeout=120)
    at.radio(key="rc_val_outcome").set_value("Retention").run()
    check(at, "insights: validity by retention")
    at.selectbox(key="rc_prof_bu").select("Risk Management").run()
    check(at, "insights: role profile for Risk Management")
    at = new_page("render_funnel_page")
    at.multiselect(key="rc_bu").set_value(["Wealth Management"]).run()
    check(at, "funnel: filtered to Wealth Management")


def test_recruitment_minimal_upload_degrades_gracefully():
    import pandas as pd
    full = pd.read_csv("recruitment_data.csv")
    full[["Candidate_ID", "Business_Unit", "Job_Level", "Application_Date", "Source_Channel", "Funnel_Status", "Joined"]].to_csv("/tmp/rc_mini.csv", index=False)
    for fn in ("render_funnel_page", "render_insights_page", "render_scoring_page", "render_fairness_page"):
        script = f"""
import sys; sys.path.insert(0, r"{HERE}")
import pandas as pd
from state import set_rec_data
set_rec_data(pd.read_csv("/tmp/rc_mini.csv"))
from pages_recruit import {fn}
{fn}()
"""
        at = AppTest.from_string(script, default_timeout=120)
        at.run()
        assert not at.exception, f"{fn} crashed on a minimal upload: {[e.value[:120] for e in at.exception]}"
    print("PASS: a minimal recruitment upload (required columns only) does not crash any recruitment page")


def test_recruitment_workspace_through_router():
    for p in ["Recruitment Data", "Hiring Funnel & Sources", "Hogan & PI Insights", "Candidate Scoring", "Model Quality & Fairness", "Glossary "]:
        script = f"""
import sys; sys.path.insert(0, r"{HERE}")
import streamlit as st
import streamlit_option_menu as som
som.option_menu = lambda *a, **k: {p!r}
from state import set_data, load_sample_dataset, set_rec_data, load_sample_recruitment
st.session_state["workspace"] = "Recruitment"
if st.session_state.get("data") is None:
    set_data(load_sample_dataset())
if st.session_state.get("rec_data") is None:
    set_rec_data(load_sample_recruitment())
exec(open(r"{HERE}/app.py").read())
"""
        at = AppTest.from_string(script, default_timeout=180)
        at.run()
        check(at, f"router (Recruitment workspace) -> {p.strip()}")


def test_upload_validation():
    at = new_page("render_upload_page")
    check(at, "upload page with data loaded")


if __name__ == "__main__":
    test_full_app_and_upload_page()
    test_pages_render()
    test_pages_without_data_show_empty_state()
    test_layer1_sliders_move_the_chart()
    test_layer2_trains_and_retrains()
    test_succession_controls()
    test_genai_fallback()
    test_rx_catalogue()
    test_career_animation_figure()
    test_ml_quality()
    test_upload_validation()
    test_every_sidebar_page_through_the_real_router()
    test_settings_survive_page_switches()
    test_recruitment_data_is_sound()
    test_recruit_models_and_backtest()
    test_fairness_monitor_detects_injected_disparity()
    test_recruitment_pages_render()
    test_scoring_page_interactions()
    test_insights_and_funnel_interactions()
    test_recruitment_minimal_upload_degrades_gracefully()
    test_recruitment_workspace_through_router()
    print(f"\nAll {len(PASSED)} page/interaction checks passed.")
