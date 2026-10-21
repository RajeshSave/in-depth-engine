"""pages_recruit.py -- Recruitment workspace pages."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import recruit_engine as rec
from generate_recruitment_data import CRITICAL_DERAILERS, CULTURE_BENCH, HDS, HPI, MVPI, PI_FACTORS, PI_IDEAL
from state import get_rec, load_sample_recruitment, pretty, reset_rec_results, set_rec_data
from styling import GOLD, STRETCH, TEAL, band_pill, card, empty_state, kpi_card, kpi_row, page_header, show_plot, show_table

REC_COLORS = {"Strong Hire": "#1B7A3E", "Hire": "#0D7377", "Review": "#D97706", "Not Recommended": "#B91C1C",
              "Awaiting interviews": "#64748B", "Not interviewed": "#94A3B8"}
PIPELINE_PREFIX = ("In Process", "Offer Extended", "Offer Accepted - Pending")
DISCLAIMER = ("Decision support, not a decision-maker: scores should inform a structured human review. "
              "Validate against your own hires and check adverse impact before using any score in selection.")


def _need():
    if get_rec() is None:
        empty_state("Load the recruitment dataset (or upload your ATS + Hogan + PI export) on the 'Recruitment Data' page first.")
        return False
    return True


def _label(c):
    return c.replace("HPI_", "").replace("HDS_", "").replace("MVPI_", "").replace("PI_", "PI ").replace("_", " ")


# ============================================================================ DATA
def render_recruit_data_page():
    page_header("Recruitment Data", "ATS funnel + Hogan + Predictive Index + post-hire outcomes. Load the sample, or upload your own export.")
    left, right = st.columns([2, 3])
    with left, card("Load recruitment dataset"):
        up = st.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed", key="upload_rc")
        if up is not None:
            try:
                new = pd.read_csv(up)
                missing = [c for c in rec.REQUIRED_RECRUIT_COLS if c not in new.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                elif st.button("Apply uploaded dataset", key="rc_apply"):
                    set_rec_data(new)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")
        st.caption("No file handy? Use the synthetic sample (12,000 applicants over 5 years, with a live pipeline):")
        if st.button("Load sample recruitment dataset", key="rc_sample"):
            set_rec_data(load_sample_recruitment())
            st.rerun()
        st.caption("**Synthetic data note:** Hogan and PI are proprietary tools. The sample simulates scores on their documented scale "
                   "names; relationships to outcomes are assumptions, not validity evidence. With real data, map your export columns "
                   "to the same names (see the data dictionary below).")
    with right, card("Active dataset"):
        d = get_rec()
        if d is None:
            empty_state("Nothing loaded yet.")
        else:
            obs = int((d["Outcome_Observed"] == "Y").sum())
            pipe = int(d["Funnel_Status"].astype(str).str.startswith(PIPELINE_PREFIX).sum())
            kpi_row(kpi_card(f"{len(d):,}", "Candidates"), kpi_card(f"{int((d['Joined'] == 'Y').sum()):,}", "Hires", GOLD),
                    kpi_card(f"{obs:,}", "Hires with 12M outcome", "#1B7A3E"), kpi_card(f"{pipe:,}", "In pipeline now", "#7C3AED"))
            show_table(d.head(40), height=300)
    with st.expander("Data dictionary: columns the engine looks for"):
        show_table(pd.DataFrame([
            ("Required", ", ".join(rec.REQUIRED_RECRUIT_COLS)),
            ("ATS background", ", ".join(rec.ATS_NUM)),
            ("Offer", ", ".join(rec.OFFER_NUM) + ", Offer_Accepted (Y/N)"),
            ("Hogan HPI (percentiles)", ", ".join(HPI)),
            ("Hogan HDS derailers (percentiles)", ", ".join(HDS)),
            ("Hogan MVPI values (percentiles)", ", ".join(MVPI)),
            ("Predictive Index", ", ".join(PI_FACTORS) + ", PI_Cognitive_Raw, PI_Cognitive_Percentile, PI_Reference_Pattern, PI_Behavioral_Fit"),
            ("Derived / fit (optional)", "Hogan_Job_Fit, Derailer_Risk_Index, Critical_Derailers_Elevated, Max_Derailer_Score, Max_Derailer_Name, Culture_Alignment_Score"),
            ("Outcomes (hires only)", "Outcome_Observed (Y/N), Perf_Rating_12M (1-5), Retained_12M (Y/N), Quality_of_Hire_Index"),
            ("Fairness only (never model inputs)", "Gender, Age, Age_Band"),
        ], columns=["Group", "Columns"]))


# ============================================================================ FUNNEL & SOURCES
def render_funnel_page():
    if not _need():
        return
    d0 = get_rec()
    page_header("Hiring Funnel & Sources", "Where candidates drop out, how fast we move, and which sources produce hires who perform and stay.")
    with card("Filters"):
        c1, c2, c3 = st.columns(3)
        dmin, dmax = d0["Application_Date"].min().date(), d0["Application_Date"].max().date()
        rng = c1.date_input("Application date range", (dmin, dmax), min_value=dmin, max_value=dmax, key="rc_dates")
        bus = c2.multiselect("Business unit", sorted(d0["Business_Unit"].unique()), key="rc_bu")
        lv = c3.multiselect("Level", sorted(d0["Job_Level"].unique()), key="rc_lv")
    d = d0
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        d = d[(d["Application_Date"] >= pd.Timestamp(rng[0])) & (d["Application_Date"] <= pd.Timestamp(rng[1]))]
    if bus: d = d[d["Business_Unit"].isin(bus)]
    if lv: d = d[d["Job_Level"].isin(lv)]
    if d.empty:
        st.warning("No candidates match these filters.")
        return
    obs = d[d["Outcome_Observed"] == "Y"]
    offers = d[d["Offer_Accepted"].isin(["Y", "N"])]
    kpi_row(kpi_card(f"{len(d):,}", "Applications"), kpi_card(f"{int((d['Joined'] == 'Y').sum()):,}", "Hires", GOLD),
            kpi_card("n/a" if offers.empty else f"{100 * (offers['Offer_Accepted'] == 'Y').mean():.0f}%", "Offer accept rate", "#1B7A3E"),
            kpi_card("n/a" if d["Time_to_Offer_Days"].notna().sum() == 0 else f"{d['Time_to_Offer_Days'].median():.0f}", "Median days to offer", "#7C3AED"),
            kpi_card("n/a" if obs.empty else f"{100 * (obs['Retained_12M'] == 'Y').mean():.0f}%", "12M retention of hires", "#0D7377"),
            kpi_card("n/a" if obs.empty else f"{obs['Perf_Rating_12M'].mean():.2f}", "Avg first-year rating"))
    a, b = st.columns([5, 7])
    f = rec.funnel_counts(d)
    with a, card("Funnel"):
        fig = go.Figure(go.Funnel(y=f["Stage"], x=f["Candidates"], textinfo="value+percent previous",
                                  marker=dict(color=[TEAL, "#2E86C1", "#7FB3D5", GOLD, "#D97706", "#1B7A3E", "#0B2540"])))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        show_plot(fig)
    with b, card("Conversion by business unit"):
        g = d.groupby("Business_Unit").agg(Applications=("Candidate_ID", "size"),
                                           Assessed=("Assessment_Date", lambda s: s.notna().sum()),
                                           Interviewed=("Interview_Date", lambda s: s.notna().sum()),
                                           Offered=("Offer_Date", lambda s: s.notna().sum()),
                                           Hired=("Joined", lambda s: (s == "Y").sum())).reset_index()
        g["Hire rate"] = (g["Hired"] / g["Applications"]).round(3)
        g["Interview to offer"] = (g["Offered"] / g["Interviewed"].replace(0, np.nan)).round(3)
        show_table(g, height=170)
        fig = px.box(d.dropna(subset=["Time_to_Offer_Days"]), x="Business_Unit", y="Time_to_Offer_Days", color="Business_Unit",
                     color_discrete_sequence=px.colors.sequential.Teal, height=200, labels={"Time_to_Offer_Days": "Days to offer"})
        fig.update_layout(margin=dict(l=10, r=10, t=5, b=5), showlegend=False, xaxis_title=None)
        show_plot(fig)
    with card("Which sources produce good hires?"):
        sq = rec.source_quality(d)
        c1, c2 = st.columns([6, 5])
        with c1:
            show_table(sq, height=250)
        with c2:
            fig = px.bar(sq.dropna(subset=["12M retention"]), x="Source_Channel", y="12M retention", color="Avg performance", height=250,
                         color_continuous_scale="Teal", labels={"Source_Channel": ""})
            fig.update_layout(margin=dict(l=10, r=10, t=5, b=5), yaxis_tickformat=".0%")
            show_plot(fig)
        st.caption("Retention and performance use only hires with a full 12 months of history. Small groups swing a lot, so read gaps of a few points with caution.")
    q = d.assign(Quarter=d["Application_Date"].dt.to_period("Q").astype(str))
    t = q.groupby("Quarter").agg(Applications=("Candidate_ID", "size"), Hires=("Joined", lambda s: (s == "Y").sum())).reset_index()
    c1, c2 = st.columns(2)
    with c1, card("Applications and hires by quarter"):
        fig = go.Figure()
        fig.add_bar(x=t["Quarter"], y=t["Applications"], name="Applications", marker_color="#7FB3D5")
        fig.add_scatter(x=t["Quarter"], y=t["Hires"], name="Hires", yaxis="y2", line=dict(color=GOLD, width=3))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25))
        show_plot(fig)
    with c2, card("Hiring-manager outcomes (managers with 8+ observed hires)"):
        mo = rec.manager_outcomes(d)
        if mo.empty:
            st.info("Not enough observed hires per manager under these filters.")
        else:
            show_table(mo, height=300)


# ============================================================================ HOGAN & PI INSIGHTS
def _candidate_options(d):
    return dict(zip(d["Candidate_ID"], d["Candidate_ID"] + " | " + d["Business_Unit"] + " | " + d["Job_Level"] + " | " + d["Funnel_Status"].astype(str)))


def render_insights_page():
    if not _need():
        return
    d = get_rec()
    page_header("Hogan & Predictive Index Insights", "What the assessments tell us -- about the role, about your past hires, and about each candidate.")
    t1, t2, t3 = st.tabs(["What predicts success here", "Role profiles", "Candidate assessment profile"])
    with t1:
        v = rec.assessment_validity(d)
        if v.empty:
            st.info("Not enough hires with 12-month outcomes to measure validity.")
        else:
            c1, c2 = st.columns([1, 3])
            outcome = c1.radio("Outcome", ["Performance", "Retention"], key="rc_val_outcome")
            fams = c1.multiselect("Measure families", sorted(v["Family"].unique()), default=sorted(v["Family"].unique()), key="rc_val_fam")
            col, pcol = ("r with performance", "p (perf)") if outcome == "Performance" else ("r with retention", "p (retention)")
            x = v[v["Family"].isin(fams)].copy()
            x["Significant"] = np.where(x[pcol] < 0.05, "p < 0.05", "not significant")
            x["Label"] = x["Measure"].map(_label)
            x = x.reindex(x[col].abs().sort_values(ascending=False).index).head(20).sort_values(col)
            with c2, card(f"Correlation with {outcome.lower()} among {int(v['n'].max()):,} hires with 12-month outcomes"):
                fig = px.bar(x, x=col, y="Label", orientation="h", color="Significant", height=520,
                             color_discrete_map={"p < 0.05": TEAL, "not significant": "#CBD5E1"})
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None, xaxis_title="Correlation (r)", legend_title=None)
                show_plot(fig)
                st.caption("Top 20 by strength. Measured on people we hired, who were already screened, so true effects among all applicants are usually larger "
                           "(range restriction). Correlation is not causation, and the sample is synthetic.")
    with t2:
        _tab_roles(d)
    with t3:
        _tab_candidate(d)


def _tab_roles(d):
    obs = d[(d["Outcome_Observed"] == "Y") & d["Perf_Rating_12M"].notna()]
    if obs.empty:
        st.info("Role profiles need hires with 12-month performance outcomes, and this dataset has none yet.")
        return
    bu = st.selectbox("Business unit", sorted(obs["Business_Unit"].unique()), key="rc_prof_bu")
    o = obs[obs["Business_Unit"] == bu].copy()
    q75, q25 = o["Perf_Rating_12M"].quantile(0.75), o["Perf_Rating_12M"].quantile(0.25)
    top, bot = o[o["Perf_Rating_12M"] >= q75], o[o["Perf_Rating_12M"] <= q25]
    a, b = st.columns(2)
    with a, card(f"Hogan HPI: top vs bottom performers ({bu})"):
        labels = [_label(c) for c in HPI]
        fig = go.Figure()
        for frame, nm, colr in ((top, f"Top performers (n={len(top)})", TEAL), (bot, f"Bottom performers (n={len(bot)})", "#B91C1C")):
            vals = [frame[c].mean() for c in HPI]
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name=nm, line=dict(color=colr)))
        fig.update_layout(height=380, polar=dict(radialaxis=dict(range=[0, 100])), legend=dict(orientation="h", y=-0.1), margin=dict(l=40, r=40, t=20, b=20))
        show_plot(fig)
    with b, card("Derailers and early exits"):
        o["Elevated"] = np.where(o["Critical_Derailers_Elevated"] >= 1, "1+ role-critical derailer elevated (70+)", "None elevated")
        g = o.groupby("Elevated").agg(Hires=("Candidate_ID", "size"), Retention_12M=("Retained_12M", lambda s: (s == "Y").mean()),
                                      Avg_Performance=("Perf_Rating_12M", "mean")).round(3).reset_index()
        show_table(g, height=110)
        st.caption("Role-critical derailers for " + bu + ": " + ", ".join(_label(c) for c in CRITICAL_DERAILERS.get(bu, [])) +
                   ". Hogan flags scores of 70+ as moderate and 90+ as high risk.")
        g2 = o.groupby("PI_Reference_Pattern").agg(Hires=("Candidate_ID", "size"), Retention_12M=("Retained_12M", lambda s: (s == "Y").mean()),
                                                   Avg_Performance=("Perf_Rating_12M", "mean")).round(2).reset_index()
        fig = px.bar(g2[g2["Hires"] >= 10].sort_values("Avg_Performance"), x="Avg_Performance", y="PI_Reference_Pattern", orientation="h", height=240,
                     labels={"PI_Reference_Pattern": "PI pattern"}, color="Retention_12M", color_continuous_scale="Teal")
        fig.update_layout(margin=dict(l=10, r=10, t=5, b=5))
        show_plot(fig)
        st.caption("PI patterns with 10+ hires in this business unit.")


def _tab_candidate(d):
    a = d[d["HPI_Adjustment"].notna()]
    if a.empty:
        st.info("No candidates have Hogan / PI assessment scores in this dataset yet.")
        return
    opts = _candidate_options(a)
    pipe_first = [k for k, v in opts.items() if v.split(" | ")[-1].startswith(PIPELINE_PREFIX)]
    ordered = pipe_first + [k for k in opts if k not in pipe_first]
    cid = st.selectbox("Select an assessed candidate (pipeline candidates first)", ordered, format_func=opts.get, key="rc_cand_profile")
    r = a[a["Candidate_ID"] == cid].iloc[0]
    hires = a[(a["Business_Unit"] == r["Business_Unit"]) & (a["Joined"] == "Y")]
    kpi_row(kpi_card(f"{r['Hogan_Job_Fit']:.0f}", "Hogan job fit", TEAL), kpi_card(f"{r['PI_Behavioral_Fit']:.0f}", "PI behavioral fit", GOLD),
            kpi_card(f"{r['PI_Cognitive_Percentile']:.0f}", "PI cognitive percentile", "#2563EB"),
            kpi_card(f"{r['Culture_Alignment_Score']:.0f}", "Culture alignment", "#1B7A3E"),
            kpi_card(str(r["PI_Reference_Pattern"]), "PI reference pattern", "#7C3AED"))
    c1, c2 = st.columns(2)
    with c1, card("Hogan HPI vs average hire in this business unit"):
        labels = [_label(c) for c in HPI]
        fig = go.Figure()
        for vals, nm, colr in (([r[c] for c in HPI], cid, TEAL), ([hires[c].mean() for c in HPI], f"Avg {r['Business_Unit']} hire", GOLD)):
            fig.add_trace(go.Scatterpolar(r=list(vals) + [vals[0]], theta=labels + labels[:1], name=nm, line=dict(color=colr), fill="toself", opacity=0.6))
        fig.update_layout(height=360, polar=dict(radialaxis=dict(range=[0, 100])), legend=dict(orientation="h", y=-0.1), margin=dict(l=40, r=40, t=20, b=20))
        show_plot(fig)
    with c2, card("Hogan HDS derailers (higher = more risk under pressure)"):
        crit = set(CRITICAL_DERAILERS[r["Business_Unit"]])
        hd = pd.DataFrame({"Derailer": [_label(c) for c in HDS], "Score": [r[c] for c in HDS], "Role-critical": ["Critical for this role" if c in crit else "Other" for c in HDS]})
        fig = px.bar(hd.sort_values("Score"), x="Score", y="Derailer", orientation="h", color="Role-critical", height=360,
                     color_discrete_map={"Critical for this role": "#B91C1C", "Other": "#94A3B8"})
        fig.add_vline(x=70, line_dash="dash", line_color="#D97706"); fig.add_vline(x=90, line_dash="dash", line_color="#B91C1C")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 100], yaxis_title=None, legend_title=None)
        show_plot(fig)
    c3, c4 = st.columns(2)
    with c3, card("Predictive Index drives vs ideal for this role"):
        ideal = PI_IDEAL[r["Business_Unit"]]
        names = ["Dominance", "Extraversion", "Patience", "Formality"]
        fig = go.Figure()
        fig.add_bar(x=names, y=[r[c] for c in PI_FACTORS], name="Candidate", marker_color=TEAL)
        fig.add_scatter(x=names, y=ideal, name="Ideal for role", mode="markers", marker=dict(symbol="diamond", size=14, color=GOLD))
        fig.update_layout(height=300, yaxis_range=[0, 100], margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
        show_plot(fig)
    with c4, card("MVPI values vs firm culture benchmark"):
        mv = pd.DataFrame({"Value": [_label(c) for c in MVPI], "Candidate": [r[c] for c in MVPI], "Firm benchmark": [CULTURE_BENCH[c] for c in MVPI]})
        fig = px.bar(mv.melt("Value", var_name="Series", value_name="Percentile"), x="Value", y="Percentile", color="Series", barmode="group", height=300,
                     color_discrete_map={"Candidate": TEAL, "Firm benchmark": GOLD})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title=None, xaxis_title=None)
        show_plot(fig)


# ============================================================================ CANDIDATE SCORING
def _train(df, method):
    with st.spinner("Training 3 models" + (" (Random Forest takes about 30 seconds)..." if method == "rf" else "...")):
        models = rec.train_recruit_models(df, method)
        st.session_state["rec_models"] = models
        st.session_state["rec_scored"] = rec.score_candidates(models, df)
        st.session_state["rec_autotrained"] = True
    return models


def render_scoring_page():
    if not _need():
        return
    d = get_rec()
    page_header("Candidate Scoring & Hiring Decisions", "Predicted first-year performance, 12-month retention and offer acceptance, combined into a hire recommendation.")
    c1, c2 = st.columns([1, 3])
    method = c1.radio("Algorithm family", ["glm", "rf"], key="rc_method", format_func=lambda m: "Regularised linear (fast, explainable)" if m == "glm" else "Random Forest")
    if c1.button("Train / Refresh models", key="rc_train"):
        _train(d, method)
    elif not st.session_state.get("rec_autotrained"):
        _train(d, method)
    models, scored = st.session_state.get("rec_models"), st.session_state.get("rec_scored")
    for e in (models or {}).get("errors", []):
        c2.warning(e)
    if not models or scored is None or "Recommendation" not in scored:
        st.error("Models could not be trained on this dataset (it needs hires with 12-month performance and retention outcomes, and decided offers).")
        return
    c2.info(f"Models trained on {models['perf']['n']:,} hires with 12-month outcomes and {models['acc']['n']:,} decided offers. {DISCLAIMER}")

    pipe = scored[scored["Funnel_Status"].astype(str).str.startswith(PIPELINE_PREFIX)].copy()
    live_offers = pipe[pipe["Funnel_Status"].isin(["Offer Extended - Awaiting Decision", "Offer Accepted - Pending Join"])]
    kpi_row(kpi_card(len(pipe), "Pipeline candidates scored"), kpi_card(int((pipe["Recommendation"] == "Strong Hire").sum()), "Strong Hire", "#1B7A3E"),
            kpi_card(int((pipe["Red_Flags"] != "").sum()), "With red flags", "#B91C1C"),
            kpi_card(int((live_offers["Accept_Prob"] < 0.5).sum()), "Offers at risk of decline", "#D97706"),
            kpi_card(f"{pipe['QoH_Pred'].mean():.0f}", "Avg predicted quality of hire", GOLD))
    with card("Pipeline: ranked by predicted quality of hire"):
        f1, f2, f3, f4 = st.columns(4)
        bus = f1.multiselect("Business unit", sorted(pipe["Business_Unit"].unique()), key="rc_s_bu")
        lv = f2.multiselect("Level", sorted(pipe["Job_Level"].unique()), key="rc_s_lv")
        stt = f3.multiselect("Funnel status", sorted(pipe["Funnel_Status"].astype(str).unique()), key="rc_s_st")
        rc = f4.multiselect("Recommendation", list(REC_COLORS), key="rc_s_rec")
        t = pipe
        if bus: t = t[t["Business_Unit"].isin(bus)]
        if lv: t = t[t["Job_Level"].isin(lv)]
        if stt: t = t[t["Funnel_Status"].astype(str).isin(stt)]
        if rc: t = t[t["Recommendation"].isin(rc)]
        t = t.sort_values("QoH_Pred", ascending=False)
        st.caption(f"{len(t)} candidates shown. Candidates without an offer yet are scored as if given a typical offer; use the what-if panel below to test offer terms.")
        cols = ["Candidate_ID", "Business_Unit", "Job_Level", "Funnel_Status", "Recommendation", "QoH_Pred", "Pred_Perf_12M", "Retention_12M_Prob", "Accept_Prob", "Red_Flags"]
        st.dataframe(t[cols], **STRETCH, height=330, hide_index=True, column_config={
            "QoH_Pred": st.column_config.ProgressColumn("Quality of hire", min_value=0, max_value=100, format="%.0f"),
            "Pred_Perf_12M": st.column_config.NumberColumn("Pred. rating", format="%.2f"),
            "Retention_12M_Prob": st.column_config.ProgressColumn("12M retention", min_value=0, max_value=1, format="%.2f"),
            "Accept_Prob": st.column_config.ProgressColumn("Accept prob.", min_value=0, max_value=1, format="%.2f")})
    if t.empty:
        st.info("No candidates match the filters.")
    else:
        opts = dict(zip(t["Candidate_ID"], t["Candidate_ID"] + " | " + t["Business_Unit"] + " " + t["Job_Level"] + " | " + t["Recommendation"]))
        with card("Candidate decision view"):
            cid = st.selectbox("Select candidate", list(opts), format_func=opts.get, key="rc_s_cand")
            r = t[t["Candidate_ID"] == cid].iloc[0]
            colr = REC_COLORS.get(r["Recommendation"], "#64748B")
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'><span style='font-family:Syne,sans-serif;font-weight:800;'>{cid}</span>"
                        f"{band_pill(r['Recommendation'], colr)}<span style='font-size:0.78rem;color:#64748B;'>{r['Business_Unit']} | {r['Job_Level']} | {r['Funnel_Status']}</span></div>",
                        unsafe_allow_html=True)
            kpi_row(kpi_card(f"{r['QoH_Pred']:.0f}", f"Quality of hire (top {100 - r['QoH_Percentile']:.0f}% of pool)", colr),
                    kpi_card(f"{r['Pred_Perf_12M']:.2f}", "Predicted first-year rating", TEAL),
                    kpi_card(f"{100 * r['Retention_12M_Prob']:.0f}%", "Predicted 12M retention", "#1B7A3E"),
                    kpi_card(f"{100 * r['Accept_Prob']:.0f}%", "Offer acceptance likelihood", GOLD))
            if r["Red_Flags"]:
                st.warning("Red flags: " + r["Red_Flags"])
            else:
                st.success("No red flags.")
            a, b = st.columns(2)
            with a:
                st.markdown("**What is driving this score** (versus a typical candidate)")
                ex = rec.explain_candidate(models, r)
                for label, (base, tbl) in ex.items():
                    tbl = tbl.assign(Driver=tbl["Feature"].map(_label), Direction=np.where(tbl["Effect"] >= 0, "Helps", "Hurts"))
                    fig = px.bar(tbl.sort_values("Effect"), x="Effect", y="Driver", orientation="h", color="Direction", height=230,
                                 color_discrete_map={"Helps": TEAL, "Hurts": "#B91C1C"}, title=label)
                    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), yaxis_title=None, legend_title=None, title_font_size=12)
                    show_plot(fig)
            with b:
                st.markdown("**What-if: change the offer**")
                base_ratio = float(r["Offer_to_Expected_Ratio"]) if pd.notna(r["Offer_to_Expected_Ratio"]) else 1.0
                ratio = st.slider("Offer as a multiple of expected comp", 0.85, 1.25, float(np.clip(base_ratio, 0.85, 1.25)), 0.01, key=f"rc_w_ratio_{cid}")
                sign = st.slider("Sign-on bonus (USD)", 0, 40000, int(r["Sign_On_Bonus_USD"]) if pd.notna(r["Sign_On_Bonus_USD"]) else 0, 5000, key=f"rc_w_sign_{cid}")
                comp = st.slider("Competing offers the candidate holds", 0, 3, int(r["Competing_Offers"]) if pd.notna(r["Competing_Offers"]) else 1, key=f"rc_w_comp_{cid}")
                base_pred = rec.whatif_offer(models, r, ratio=base_ratio, sign_on=float(r["Sign_On_Bonus_USD"]) if pd.notna(r["Sign_On_Bonus_USD"]) else 0.0,
                                             competing=float(r["Competing_Offers"]) if pd.notna(r["Competing_Offers"]) else 1.0)
                new_pred = rec.whatif_offer(models, r, ratio=ratio, sign_on=float(sign), competing=float(comp))
                m1, m2, m3 = st.columns(3)
                m1.metric("Offer acceptance", f"{100 * new_pred['Accept_Prob']:.0f}%", f"{100 * (new_pred['Accept_Prob'] - base_pred['Accept_Prob']):+.1f} pts")
                m2.metric("12M retention", f"{100 * new_pred['Retention_12M_Prob']:.0f}%", f"{100 * (new_pred['Retention_12M_Prob'] - base_pred['Retention_12M_Prob']):+.1f} pts")
                m3.metric("Predicted rating", f"{new_pred['Pred_Perf_12M']:.2f}", f"{new_pred['Pred_Perf_12M'] - base_pred['Pred_Perf_12M']:+.2f}")
                st.caption("Compared with the offer as currently set (or a typical offer if none yet). Associations in past hires, not a guarantee.")
    with card("Evidence check: do the scores actually separate good hires from weak ones?"):
        bt = rec.backtest(models, d)
        if bt is None:
            st.info("Not enough outcome data for a backtest.")
        else:
            st.caption("Past hires only, scored by models that never saw their own outcome (cross-validated). Hires are split into quarters by each ranking method, "
                       "then compared on what actually happened.")
            fig = px.bar(bt, x="Group", y="12M Retention", color="Ranked by", barmode="group", height=300, text_auto=".0%",
                         category_orders={"Group": ["Bottom 25%", "Lower-mid", "Upper-mid", "Top 25%"]})
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%", legend_title=None, legend=dict(orientation="h", y=-0.25))
            show_plot(fig)
            show_table(bt.round(2), height=250)
            st.caption("A useful model shows a clear step from bottom to top. Interview score alone usually separates far less.")


# ============================================================================ FAIRNESS & MODEL QUALITY
def _fairness_tab(d, scored):
    groups = [g for g in ("Age_Band", "Gender") if g in d and d[g].notna().any()]
    if not groups:
        st.info("This dataset has no Gender or Age_Band column, so the fairness monitor cannot run. Add those columns (for monitoring only; "
                "they are never model inputs) to enable it.")
        return
    c1, c2, c3 = st.columns(3)
    group = c1.radio("Group by", groups, key="rc_f_group")
    lvl = c2.multiselect("Compare within level", sorted(d["Job_Level"].unique()), key="rc_f_lvl")
    c3.caption("Comparing within a level removes the effect of different level mixes (older candidates skew senior).")
    dd = d[d["Job_Level"].isin(lvl)] if lvl else d
    ai = rec.adverse_impact(dd, group)
    with card("Selection rate at each funnel step (four-fifths rule)"):
        n_bad = int((ai["Four-fifths check"] == "Below 0.80").sum())
        (st.warning if n_bad else st.success)(f"{n_bad} funnel-step / group combination(s) fall below the 0.80 ratio." if n_bad else "No step falls below the 0.80 ratio.")
        fig = px.bar(ai.dropna(subset=["Ratio vs highest group"]), x="Funnel step", y="Ratio vs highest group", color=group, barmode="group", height=320,
                     text="Ratio vs highest group")
        fig.add_hline(y=0.8, line_dash="dash", line_color="#B91C1C", annotation_text="0.80")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title=None, yaxis_range=[0, 1.15])
        show_plot(fig)
        show_table(ai.style.apply(lambda r: ["background-color:#FDECEA" if r["Four-fifths check"] == "Below 0.80" else "" for _ in r], axis=1), height=300)
    if scored is not None and "Recommendation" in scored:
        sc = scored[scored["Job_Level"].isin(lvl)] if lvl else scored
        with card("Model recommendations by group (candidates interviewed or beyond)"):
            ri = rec.recommendation_impact(sc, group)
            show_table(ri, height=170)
            m = sc[sc["Recommendation"].isin(["Strong Hire", "Hire", "Review", "Not Recommended"])].groupby(group)["QoH_Pred"].mean().round(1).reset_index()
            st.caption("Average predicted quality of hire by group: " + " | ".join(f"{r[group]}: {r['QoH_Pred']}" for _, r in m.iterrows()))
    st.info("Gender, age and age band are never inputs to the models. The four-fifths rule is a screening heuristic, not a legal conclusion: a ratio below 0.80 "
            "means look closer (sample size, level mix, job-relatedness of the criteria), not that discrimination has been proven. "
            "In the synthetic sample, an age-related disparity was deliberately injected into interview scoring so this monitor has something to find.")


def render_fairness_page():
    if not _need():
        return
    d = get_rec()
    page_header("Model Quality & Fairness", "How good are the models, and are selection outcomes balanced across groups?")
    models, scored = st.session_state.get("rec_models"), st.session_state.get("rec_scored")
    t1, t2 = st.tabs(["Fairness monitor", "Model quality"])
    with t1:
        _fairness_tab(d, scored)
    with t2:
        if not models:
            st.info("Open 'Candidate Scoring' once to train the models.")
        else:
            for key, title in (("perf", "Performance model (regression)"), ("ret", "12-month retention model (classification)"), ("acc", "Offer-acceptance model (classification)")):
                m = models.get(key)
                with card(title):
                    if not m:
                        st.info("Not trained (not enough data).")
                        continue
                    a, b = st.columns([2, 3])
                    with a:
                        for k, v in m["metrics"].items():
                            st.metric(k, f"{v:.2f}")
                        st.caption(f"{m['method'].upper()} | trained on {m['n']:,} rows | 5-fold cross-validation, so these are honest out-of-sample numbers.")
                    with b:
                        imp = m["importance"].assign(Feature=lambda x: x["Feature"].map(_label)).sort_values("Importance")
                        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", height=300)
                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
                        show_plot(fig)
            st.caption("Models learn from people who were hired, so they have never seen how rejected candidates would have performed (selection bias). Treat scores as a "
                       "structured input to judgement, and re-check them as new outcomes arrive.")
