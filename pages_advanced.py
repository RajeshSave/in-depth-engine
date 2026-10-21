"""pages_advanced.py -- Career Trajectory, Succession Matrix, Development Rx, AI Report, Reports, Glossary."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ml_engine as ml
from catalogue import DEFAULT_CATALOGUE, prescribe
from genai import PROVIDERS, generate_report, report_to_html
from state import DEFAULT_GATE, DEFAULT_PHASE, LEVELS, enriched, get_data, pretty, readiness_params
from styling import (GOLD, TEAL, avatar_html, band_pill, card, empty_state, kpi_card, kpi_row, page_header,
                     show_plot, show_table)


def _need_data():
    if get_data() is None:
        empty_state()
        return False
    return True


def _name_options(df):
    return dict(zip(df["Employee_ID"], df["Full_Name"] + " (" + df["Employee_ID"] + ")"))


# ============================================================================ CAREER TRAJECTORY
def trajectory_points(emp: pd.Series):
    """Chronological years (1 = oldest of the 5-year window ... 5 = most recent) with a rating,
    plus the simulated cumulative promotion count at each of those years."""
    years, ratings = [], []
    for cy in range(1, 6):
        v = emp.get(f"Performance_Rating_Y{6 - cy}")
        if pd.notna(v):
            years.append(cy)
            ratings.append(float(v))
    n = int(round(emp.get("Promotions_Last_5_Years", 0) or 0))
    # Simulated: the dataset stores a 5-year promotion COUNT, not dates, so spread it evenly.
    events = [] if n <= 0 else sorted(set(np.clip(np.round(np.linspace(5 / (n + 1), 5, n)), 1, 5).astype(int)))
    promos = [sum(e <= y for e in events) for y in years]
    return years, ratings, promos


def trajectory_figure(emp: pd.Series, seconds_per_year: float):
    x, perf, promo = trajectory_points(emp)
    if len(x) < 2:
        return None
    ms = int(seconds_per_year * 1000)
    mk = lambda k: [go.Scatter(x=x[:k + 1], y=perf[:k + 1]), go.Scatter(x=x[:k + 1], y=promo[:k + 1])]
    base = [go.Scatter(x=x[:1], y=perf[:1], mode="lines+markers", name="Performance Rating (left axis)",
                       line=dict(color=TEAL, width=3), marker=dict(size=9), yaxis="y"),
            go.Scatter(x=x[:1], y=promo[:1], mode="lines+markers", name="Cumulative Promotions (right axis)",
                       line=dict(color=GOLD, width=3, dash="dash", shape="hv"), marker=dict(size=9), yaxis="y2")]
    steps = [dict(method="animate", label=f"Y{x[k]}", args=[[f"Y{x[k]}"], dict(mode="immediate", frame=dict(duration=ms, redraw=True),
              transition=dict(duration=ms // 3))]) for k in range(len(x))]
    fig = go.Figure(
        data=base,
        frames=[go.Frame(data=mk(k), traces=[0, 1], name=f"Y{x[k]}") for k in range(len(x))],
        layout=go.Layout(
            height=430, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.35),
            xaxis=dict(range=[0.7, 5.3], tickvals=[1, 2, 3, 4, 5], ticktext=["Y1", "Y2", "Y3", "Y4", "Y5"], title="Career year (Y5 = most recent)"),
            yaxis=dict(title="Performance Rating", range=[0.5, 5.5], dtick=1, color=TEAL),
            yaxis2=dict(title="Cumulative Promotions", overlaying="y", side="right", range=[-0.2, max(3, max(promo)) + 0.5],
                        dtick=1, color=GOLD, showgrid=False),
            updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.18, xanchor="left", buttons=[
                dict(label="\u25B6 Play", method="animate", args=[None, dict(frame=dict(duration=ms, redraw=True),
                     transition=dict(duration=ms // 3), fromcurrent=True)]),
                dict(label="\u23F8 Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                     mode="immediate", transition=dict(duration=0))])])],
            sliders=[dict(active=0, steps=steps, x=0.15, len=0.85, y=1.12, currentvalue=dict(prefix="Showing: "))]))
    return fig


def render_career_page():
    if not _need_data():
        return
    df = get_data()
    page_header("Career Trajectory", "One employee's performance and promotion history over the last five years -- press Play, or drag the slider.")
    left, right = st.columns([4, 8])
    with left, card("Controls"):
        opts = _name_options(df)
        emp_id = st.selectbox("Select employee", list(opts), format_func=opts.get, key="career_emp")
        secs = st.slider("Seconds per year (animation speed)", 0.3, 3.0, 1.2, 0.1, key="career_speed")
        emp = df[df["Employee_ID"] == emp_id].iloc[0]
        st.caption("Performance Rating (solid teal, left axis) is the employee's actual annual ratings. Cumulative Promotions "
                   "(dashed gold, right axis) is **simulated**: the data holds a 5-year promotion count, not exact dates, "
                   "so promotions are spread evenly across the window.")
        kpi_row(kpi_card(f"{emp['Promotions_Last_5_Years']:.0f}", "Promotions (5 yrs)", "#C9A227"),
                kpi_card(f"{emp['Performance_Trajectory']:+.2f}", "Perf. Trajectory"))
    with right, card("Performance & Promotion Trajectory"):
        fig = trajectory_figure(emp, secs)
        if fig is None:
            st.info("This employee has fewer than 2 years of performance-rating history, so there is no trajectory to animate yet.")
        else:
            show_plot(fig)


# ============================================================================ SUCCESSION MATRIX
RADAR_AXES = ml.COMPETENCY_COLS + ["Career_Velocity_Momentum", "Promotion_Pace_5Y"]


def _radar_values(frame: pd.DataFrame):
    vals = [frame[c].mean() for c in ml.COMPETENCY_COLS]
    vals.append(min(100, frame["Career_Velocity_Score"].mean() * 100))
    vals.append(min(100, frame["Promotions_Per_Year_Last_5Y"].mean() * 100))
    return vals


def render_succession_page():
    if not _need_data():
        return
    page_header("Succession Matrix", "9-box, bench strength and readiness -- weighted for a 3-5 year succession horizon.")
    with card("Succession Readiness Score - 3-5 Year Horizon Weighting"):
        st.caption("Performance (past execution) is an eligibility gate, not the deciding factor; Assessment (future potential) carries more weight the longer the horizon.")
        st.session_state.setdefault("rd_phase", DEFAULT_PHASE)
        st.session_state.setdefault("rd_gate", DEFAULT_GATE)
        c1, c2, c3 = st.columns([5, 4, 3])
        c1.selectbox("Horizon phase", list(ml.HORIZON_PHASES), key="rd_phase")
        c2.slider("Performance gate (minimum Performance Component to be Eligible)", 0, 100, key="rd_gate")
        w, _, _ = readiness_params()
        c3.markdown(f"<div style='font-size:0.68rem;color:#64748B;margin-bottom:3px;'>Weight split</div>"
                    f"<div style='display:flex;height:24px;border-radius:6px;overflow:hidden;font-size:0.65rem;font-weight:700;color:#fff;'>"
                    f"<div style='width:{w * 100:.0f}%;background:#0D7377;display:flex;align-items:center;justify-content:center;'>{w * 100:.0f}% Perf</div>"
                    f"<div style='width:{100 - w * 100:.0f}%;background:#C9A227;display:flex;align-items:center;justify-content:center;'>{100 - w * 100:.0f}% Assess</div></div>",
                    unsafe_allow_html=True)
    en = enriched()
    elig = int((en["Gate_Status"] == "Eligible").sum())
    kpi_row(kpi_card(f"{elig}/{len(en)}", "Gate-Eligible Candidates"), kpi_card(f"{en['Succession_Readiness_Score'].mean():.1f}", "Avg Readiness Score", "#C9A227"),
            kpi_card(int((en["Layer1_Flag"].astype(str) == "Ready for Leadership Track").sum()), "Ready for Leadership", "#1B7A3E"))

    with card("9-Box Grid: Performance x Leadership Potential"):
        d = en.dropna(subset=["Avg_Performance_Rating"])
        fig = px.scatter(d, x="Avg_Performance_Rating", y="Leadership_Potential_Score", color="Business_Unit", height=520, opacity=0.75,
                         hover_name="Full_Name", hover_data={"Job_Level": True, "Succession_Readiness_Score": True, "Gate_Status": True,
                                                              "Avg_Performance_Rating": ":.2f"})
        for xv in (2.5, 3.75): fig.add_vline(x=xv, line_dash="dash", line_color="#CBD5E1")
        for yv in (40, 70): fig.add_hline(y=yv, line_dash="dash", line_color="#CBD5E1")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Performance (avg rating, 1-5)",
                          yaxis_title="Leadership Potential (0-100)", legend_title=None)
        show_plot(fig)

    a, b = st.columns(2)
    with a, card("Bench Strength by Business Unit"):
        bench = en[en["Layer1_Flag"].astype(str).isin(["Ready for Leadership Track", "Fast-Track Talent"])]
        if bench.empty:
            st.info("No Ready / Fast-Track candidates at the current Layer 1 thresholds.")
        else:
            cnt = bench.groupby(["Business_Unit", "Layer1_Flag"], observed=True).size().reset_index(name="Count")
            cnt["Layer1_Flag"] = cnt["Layer1_Flag"].astype(str)
            fig = px.bar(cnt, x="Count", y="Business_Unit", color="Layer1_Flag", orientation="h", barmode="group", height=340,
                         color_discrete_map={"Ready for Leadership Track": "#1B7A3E", "Fast-Track Talent": "#D97706"})
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title=None, legend=dict(orientation="h", y=-0.2), yaxis_title=None)
            show_plot(fig)
    with b, card("Top Bench Candidates (Ready Now / Fast-Track)"):
        bu = st.selectbox("Business unit", ["All Business Units"] + sorted(en["Business_Unit"].unique()), key="bench_bu")
        t = bench if bu == "All Business Units" else bench[bench["Business_Unit"] == bu]
        t = t.sort_values("Succession_Readiness_Score", ascending=False).head(5)
        if t.empty:
            st.info("No candidates match.")
        palette = ["#1B7A3E", "#2563EB", "#D97706", "#7C3AED", "#0D7377"]
        labels = ["#1 Primary", "#2 Secondary", "#3 Tertiary", "#4", "#5"]
        html = ""
        for i, (_, r) in enumerate(t.iterrows()):
            flag_color = "#1B7A3E" if str(r["Layer1_Flag"]) == "Ready for Leadership Track" else "#D97706"
            gate = "" if r["Gate_Status"] == "Eligible" else " (below performance gate)"
            html += (f"<div class='scard' style='border-left-color:{palette[i]}'><span class='scard-rank'>{labels[i]}</span>"
                     f"<div style='display:flex;align-items:center;gap:12px;'>{avatar_html(r['Full_Name'], 42, palette[i])}<div>"
                     f"<h5>{r['Full_Name']}</h5><p>{r['Job_Level']} - {r['Business_Unit']}</p>"
                     f"<p>{band_pill(str(r['Layer1_Flag']), flag_color)} &nbsp; Readiness {r['Succession_Readiness_Score']}/100{gate}</p></div></div></div>")
        if html:
            st.markdown(html, unsafe_allow_html=True)

    with card("Succession Readiness Leaderboard (Gate-Eligible Candidates)"):
        lb = en[en["Gate_Status"] == "Eligible"].sort_values("Succession_Readiness_Score", ascending=False)
        show_table(lb[["Employee_ID", "Full_Name", "Business_Unit", "Job_Level", "Performance_Component", "Assessment_Component",
                       "Succession_Readiness_Score", "Layer1_Flag"]].head(100), height=320)

    with card("Candidate Competency Radar (vs. Business Unit average)"):
        opts = _name_options(en)
        emp_id = st.selectbox("Select candidate", list(opts), format_func=opts.get, key="radar_emp")
        cand = en[en["Employee_ID"] == emp_id]
        bu_avg = en[en["Business_Unit"] == cand["Business_Unit"].iloc[0]]
        labels = [pretty(c) for c in RADAR_AXES]
        fig = go.Figure()
        for vals, name, color, fill in ((_radar_values(cand), cand["Full_Name"].iloc[0], TEAL, "rgba(13,115,119,0.25)"),
                                         (_radar_values(bu_avg), f"{cand['Business_Unit'].iloc[0]} average", GOLD, "rgba(201,162,39,0.12)")):
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself", name=name,
                                          line=dict(color=color), fillcolor=fill))
        fig.update_layout(height=440, polar=dict(radialaxis=dict(visible=True, range=[0, 100])), legend=dict(orientation="h", y=-0.1),
                          margin=dict(l=40, r=40, t=20, b=20))
        show_plot(fig)


# ============================================================================ DEVELOPMENT RX
def render_rx_page():
    if not _need_data():
        return
    page_header("Development Rx", "Prescribed development actions per candidate, driven by an editable intervention catalogue.")
    if st.session_state.get("catalogue") is None:
        st.session_state.catalogue = DEFAULT_CATALOGUE.copy()
    # The editor's input table must stay constant while the editor is on screen (Streamlit keeps the
    # user's edits as a delta on top of it). Re-seed it from the saved catalogue only on a fresh visit.
    if "rx_editor" not in st.session_state or "rx_base" not in st.session_state:
        st.session_state.rx_base = st.session_state.catalogue.copy().reset_index(drop=True)

    plan_area = st.container()          # drawn above, but filled AFTER the editor so edits show immediately
    with card("Development Intervention Catalogue (editable)"):
        st.caption("Edit any cell, add rows with the + at the bottom of the table, or delete rows (select the row, press Delete). "
                   "Prescriptions above update immediately. One entry per Competency + Band is used (the first, if there are duplicates).")
        edited = st.data_editor(
            st.session_state.rx_base, num_rows="dynamic", key="rx_editor", hide_index=True, height=420,
            column_config={"Competency": st.column_config.SelectboxColumn(options=ml.COMPETENCY_COLS, required=True),
                           "Band": st.column_config.SelectboxColumn(options=["Low", "Medium", "High"], required=True),
                           "Row_ID": st.column_config.TextColumn(disabled=True, width="small"),
                           "Description": st.column_config.TextColumn(width="large")})
        live = edited.dropna(subset=["Competency", "Band"]).reset_index(drop=True)
        live["Row_ID"] = [v if isinstance(v, str) and v else f"CUSTOM-{i + 1}" for i, v in enumerate(live["Row_ID"])]
        st.session_state.catalogue = live
        c1, c2 = st.columns(2)
        if c1.button("Reset to default catalogue", key="rx_reset"):
            st.session_state.catalogue = DEFAULT_CATALOGUE.copy()
            st.session_state.pop("rx_editor", None)
            st.session_state.pop("rx_base", None)
            st.rerun()
        c2.download_button("Download catalogue (CSV)", live.to_csv(index=False), "intervention_catalogue.csv", "text/csv")

    with plan_area:
        en = enriched()
        left, right = st.columns([4, 8])
        with left, card("Prescribe for a candidate"):
            opts = _name_options(en)
            emp_id = st.selectbox("Select candidate", list(opts), format_func=opts.get, key="rx_emp")
            r = en[en["Employee_ID"] == emp_id].iloc[0]
            traj = r["Performance_Trajectory"]
            t_label = "Improving" if traj > 0.15 else ("Declining" if traj < -0.15 else "Stable")
            t_color = {"Improving": "#1B7A3E", "Declining": "#B91C1C", "Stable": "#64748B"}[t_label]
            readiness_pill = band_pill("Readiness " + str(r["Succession_Readiness_Score"]) + "/100", TEAL)
            st.markdown("<div style='display:flex;align-items:center;gap:10px;'>" + avatar_html(r["Full_Name"], 46, GOLD) + "<div>"
                        "<b>" + r["Full_Name"] + "</b><br><span style='font-size:0.75rem;color:#64748B;'>" + str(r["Job_Level"]) + " - " + r["Business_Unit"] + "</span><br>"
                        + band_pill("Trajectory: " + t_label, t_color) + " " + readiness_pill + "</div></div>", unsafe_allow_html=True)
            st.caption(f"{int(r['Promotions_Last_5_Years'])} promotion(s) in the last 5 years. Scores below 55 = Low, 55-74.9 = Medium, 75+ = High.")
        with right, card("Prescribed Development Plan"):
            plan = prescribe(r, live)
            colors = {"Low": "#B91C1C", "Medium": "#D97706", "High": "#1B7A3E"}
            n_need = int((plan["Band"] != "High").sum())
            n_str = len(plan) - n_need
            st.caption(f"{n_need} development need(s) and {n_str} stretch opportunit{'y' if n_str == 1 else 'ies'}.")
            html = ""
            for _, p in plan.iterrows():
                kind = "Stretch Opportunity" if p["Band"] == "High" else "Development Need"
                title = p["Intervention"] if pd.notna(p["Intervention"]) else "No intervention in catalogue for this competency/band"
                desc = p["Description"] if pd.notna(p["Description"]) else "Add one in the catalogue editor below."
                fmt = p["Format"] if pd.notna(p["Format"]) else ""
                html += ("<div class='rx-card' style='border-left-color:" + colors[p["Band"]] + "'><div style='display:flex;justify-content:space-between;align-items:baseline;'>"
                         "<h5>" + str(title) + "</h5>" + band_pill(pretty(p["Competency"]) + " - " + kind, colors[p["Band"]]) + "</div><p>" + str(desc) + "</p>"
                         "<p style='color:#0D7377;font-weight:700;font-size:0.7rem;margin-top:4px;'>" + str(fmt) + " | Current score: " + str(p["Score"]) + "/100</p></div>")
            st.markdown(html, unsafe_allow_html=True)


# ============================================================================ AI REPORT
def render_genai_page():
    if not _need_data():
        return
    page_header("Layer 3: AI Executive Report", "A tailored development recommendation from Gemini, Claude or ChatGPT -- with an automatic offline fallback.")
    en = enriched()
    left, right = st.columns([4, 8])
    with left, card("Report configuration"):
        opts = _name_options(en)
        emp_id = st.selectbox("Select candidate", list(opts), format_func=opts.get, key="ai_emp")
        provider = st.selectbox("GenAI engine", PROVIDERS, key="ai_provider")
        key = st.text_input("API key (optional - leave blank for the offline template report)", type="password", key="ai_key")
        go_btn = st.button("Generate Executive Report", key="ai_go")
    r = en[en["Employee_ID"] == emp_id].iloc[0]
    if go_btn:
        w, _, phase = readiness_params()
        ret = f"{r['Retention_Risk_Tier']} ({r['Retention_Risk_Prob']:.2f})" if "Retention_Risk_Tier" in r and pd.notna(r.get("Retention_Risk_Tier")) else "not yet modelled (open Layer 2 first)"
        cluster = r["Talent_Cluster"] if "Talent_Cluster" in r and pd.notna(r.get("Talent_Cluster")) else "not yet clustered"
        ctx = dict(layer1_flag=str(r["Layer1_Flag"]), cluster=cluster, retention=ret, horizon=phase.split(":")[0], perf_pct=round(w * 100))
        with st.spinner("Generating report..."):
            text, source = generate_report(provider, key, r.to_dict(), ctx)
        st.session_state["genai"] = dict(text=text, source=source, name=r["Full_Name"], emp=emp_id)
    with right, card("Executive Development Recommendation"):
        g = st.session_state.get("genai")
        if not g:
            st.info("Select a candidate and click 'Generate Executive Report'.")
        else:
            st.markdown(f"{avatar_html(g['name'], 40)}", unsafe_allow_html=True)
            (st.success if g["source"].startswith("Live") else st.warning)(f"Source: {g['source']}")
            st.markdown(g["text"])
            st.download_button("Download report (HTML)", report_to_html(f"Executive Development Recommendation - {g['name']}", g["text"]),
                               f"executive_report_{g['emp']}.html", "text/html")


# ============================================================================ REPORTS
def _summary_html(en: pd.DataFrame, full: bool = True) -> str:
    bu = en.groupby("Business_Unit").size().reset_index(name="Headcount").to_html(index=False)
    fl = en["Layer1_Flag"].astype(str).value_counts().rename_axis("Flag").reset_index(name="Count").to_html(index=False)
    lb = (en[en["Gate_Status"] == "Eligible"].sort_values("Succession_Readiness_Score", ascending=False)
          [["Full_Name", "Business_Unit", "Job_Level", "Succession_Readiness_Score"]].head(15).to_html(index=False))
    body = ("<h1 style='border-bottom:3px solid #C9A227;'>Executive Talent Summary</h1>"
            f"<p>Candidates: <b>{len(en)}</b> | Avg performance: <b>{en['Avg_Performance_Rating'].mean():.2f}</b> | "
            f"Avg readiness: <b>{en['Succession_Readiness_Score'].mean():.1f}</b> | Gate-eligible: <b>{(en['Gate_Status'] == 'Eligible').sum()}</b></p>"
            f"<h2>Headcount by Business Unit</h2>{bu}<h2>Layer 1 flags</h2>{fl}<h2>Top 15 by readiness (gate-eligible)</h2>{lb}")
    if not full:
        return "<div style='color:#0B2540;'>" + body + "</div>"
    return ("<html><head><meta charset='utf-8'><title>Executive Talent Summary</title></head>"
            "<body style='font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#0B2540;'>" + body + "</body></html>")


def render_reports_page():
    if not _need_data():
        return
    page_header("Reports & Downloads", "Take the analysis with you.")
    en = enriched()
    a, b = st.columns(2)
    with a, card("Candidate data (CSV)"):
        st.caption("All candidates with Layer 1 flags, readiness scores, clusters and any model outputs.")
        st.download_button("Download CSV", en.to_csv(index=False), "candidate_data_export.csv", "text/csv")
    with b, card("Executive summary (HTML)"):
        html = _summary_html(en)
        st.caption("Headcount, flag distribution and the top of the readiness leaderboard.")
        st.download_button("Download HTML summary", html, "executive_summary.html", "text/html")
    with card("Live preview"):
        st.html(_summary_html(en, full=False))


# ============================================================================ GLOSSARY
def render_glossary_page():
    page_header("Glossary", "Every score, rule and formula used in this portal.")
    sections = {
        "Layer 1 - rule-based flags": """
- **Ready for Leadership Track** - at least 2 of 3 elite criteria met: Analytical > threshold, Risk & Compliance > threshold, Leadership Potential > threshold.
- **Fast-Track Talent** - average performance >= 4.5 and 2+ career promotions.
- **Promotion Overdue** - 5+ years since last promotion with average performance >= 4.
- **High Attrition Risk** - engagement < 50 *and* compa-ratio < 0.95.
- **Compliance Watchlist** - Risk & Compliance score below its own threshold (default 60).
- **Performance Improvement Plan Candidate** - average performance < 3.0.
- **Steady / Core Contributor** - none of the above. Usually ~2/3 of staff, so the chart leaves it out.
Rules are checked top to bottom and the first match wins. All thresholds are adjustable on the Layer 1 page.""",
        "Layer 2 - machine learning": """
- **Model 1 - Retention Risk (classification)** predicts `Attrition_Flag = Exited`. **Model 2 - HiPo Likelihood (classification)** predicts `High_Potential_Program_Flag = Y`; people with a High score but no flag are *Hidden HiPos*. **Model 3 - Performance Outlook (regression)** predicts `Avg_Performance_Rating` from the profile (excluding the rating and its trend); `Performance_Gap` = predicted minus actual.
- **Untapped Potential** = actual rating well below what the profile predicts. **Derailment Watch** = actual rating well above it. "Well" means more than 1.25 x the model's typical error (RMSE).
- **Tiers** are rank-based within the population: top 15% = High, next 25% = Medium, rest = Low (fixed probability cut-offs left the High tier empty for rare outcomes).
- **Algorithm family**: Logistic/Linear Regression (simple, explainable) or Random Forest (flexible). Both use a 70/30 train/held-out split.
- **Talent clusters** - K-Means on the 7 standardised competency scores; clusters are named by their average score (High-Flyers highest).
- **3D PCA** compresses the 7 scores into 3 axes for plotting; each axis is named after the competencies that drive it most.
- **AUC** (0.5 = coin flip, 1.0 = perfect), **RMSE/MAE** (average rating error), **R\u00b2** (share of variation explained) live on the Model Diagnostics tab.""",
        "Succession Readiness Score": """
For long-horizon (3-5 year) succession, past performance is an **eligibility gate, not the deciding factor** - strong results in today's role don't reliably predict fit for a very different, larger role.
- **Performance Component** = mean of (average rating rescaled to 0-100, 360 feedback, technical skill, client management).
- **Assessment Component** = mean of (leadership potential, analytical, communication, risk & compliance).
- **Readiness Score** = w x Performance + (1 - w) x Assessment, where w is 20% (development phase, 3-5 yrs out), 30% (standard horizon, default) or 50% (selection phase, ~1 yr out).
- **Gate Status** - *Eligible* if Performance Component >= the gate (default 65); *Below Performance Gate* otherwise, regardless of assessment strength.""",
        "Priority Action List - 7 decision points": """
**Retention Risk**, **Hidden HiPo**, **Untapped Potential**, **Derailment Watch** (from the 3 models) plus **Pay Equity Risk** (rating >= 4 but compa-ratio < 0.90), **Blocked Potential** (assessment >= 70 but below the performance gate) and **Promotion Overdue** (Layer 1 rule). The list shows the top few per signal so no single signal crowds out the rest.""",
        "Development Rx": """
Each of the 7 competency scores is banded **Low (<55)**, **Medium (55-74.9)** or **High (>=75)** and matched to the catalogue (21 built-in entries: 7 competencies x 3 bands). Low/Medium rows are *development needs*; High rows are *stretch opportunities*. The catalogue is fully editable in the app (session only - download the CSV to keep your changes).""",
        "Recruitment engine (Hogan, Predictive Index, ATS)": """
- **Funnel**: Applications -> Screened -> Assessed (Hogan + PI + technical + case) -> Interviewed -> Offered -> Accepted -> Joined.
- **Models** (hiring-time information only; gender, age and age band are never inputs): **Performance** (regression, first-year rating 1-5), **12-month retention** (classification; the model learns early exits and the app shows the probability of staying), **Offer acceptance** (classification).
- **Quality of Hire (predicted)** = 100 x (0.6 x scaled performance + 0.4 x retention probability). Actual Quality of Hire in the data also includes ramp-up time.
- **Recommendation** tiers rank candidates against the interviewed pool: Strong Hire (top 25%), Hire (next 30%), Review (next 25%), Not Recommended (bottom 20%). A red flag moves a Strong Hire to Hire and a Hire to Review.
- **Red flags**: a Hogan derailer at 90+, Hogan job fit below 35 or PI behavioral fit below 30, retention probability below 65%, offer acceptance below 50% (pre-decision), or a background check that is not clear.
- **Hogan HPI** (normal personality), **HDS** (derailers: risk behaviours under pressure; 70+ moderate, 90+ high), **MVPI** (values, compared with the firm culture benchmark). **PI**: four behavioural drives (Dominance, Extraversion, Patience, Formality), a cognitive score and a reference pattern. Scores here are simulated on those scale names; they do not reproduce either vendor's scoring.
- **Job fit scores**: Hogan job fit = role-weighted HPI percentile minus a penalty for elevated role-critical derailers. PI behavioral fit = closeness of the four drives to the role's ideal profile. Culture alignment = closeness of MVPI values to the firm benchmark.
- **Evidence check (backtest)**: past hires scored by models that never saw their own outcome, split into quarters, compared on actual performance and retention - and against ranking by interview score alone.
- **Four-fifths rule**: a group's selection rate divided by the highest group's rate; below 0.80 is a prompt to investigate, not proof of discrimination.""",
        "Other metrics": """
- **Career Velocity** = career promotions / years at the firm (min 0.5). **Promotions (5 yrs)** = velocity projected over 5 years, capped at career promotions.
- **Performance Trajectory** = most recent rating minus the average of the latest three (new hires default to 0).
- **Compa-Ratio** = pay / market midpoint for the role (1.00 = at market).
- **9-box** - performance (x) vs leadership potential (y), split at 2.5 / 3.75 and 40 / 70.""",
    }
    for title, body in sections.items():
        with st.expander(title, expanded=(title == "Layer 1 - rule-based flags")):
            st.markdown(body)
    with st.expander("Data dictionary (key columns)"):
        show_table(pd.DataFrame([
            ("Employee_ID / Full_Name", "Identifier and name"),
            ("Business_Unit / Sub_Function / Job_Level", "Organisation. Levels run Analyst < Associate < VP < Director < Managing Director"),
            ("Years_of_Experience / Years_at_Company", "Total career length vs tenure at this firm"),
            ("Performance_Rating_Y1..Y5", "Annual ratings (1-5), Y1 = most recent; blank before hire"),
            ("Analytical / Risk_Compliance / Leadership_Potential / Communication / Technical_Skill / Client_Management scores", "Assessment-centre scores, 0-100"),
            ("Engagement_Score / Feedback_360_Score", "Survey and 360 composites, 0-100"),
            ("Current_CTC_USD / Bonus_Pct / Compa_Ratio", "Pay, bonus % and pay vs market"),
            ("High_Potential_Program_Flag / Attrition_Flag", "Y/N HiPo nomination; Active/Exited (model targets)"),
        ], columns=["Column", "Meaning"]))
