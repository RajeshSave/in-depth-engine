"""pages_core.py -- Upload, Executive Overview, Layer 1 (rules), Layer 2 (ML) pages."""
import pandas as pd
import plotly.express as px
import streamlit as st

import ml_engine as ml
from rules_engine import FLAG_COLORS, FLAG_ORDER
from state import (DEFAULT_L1, LEVELS, REQUIRED_COLS, enriched, get_data, load_sample_dataset, pretty,
                   reset_results, set_data, sticky_container)
from styling import (CLUSTER_COLORS, SIGNAL_COLORS, avatar_html, band_pill, card, empty_state, kpi_card,
                     kpi_row, page_header, show_plot, show_table)

LEVEL_COLORS = ["#AEB6BF", "#7FB3D5", "#2E86C1", "#1B4F72", "#C9A227"]


def _need_data():
    if get_data() is None:
        empty_state()
        return False
    return True


# ============================================================================ UPLOAD
def render_upload_page():
    page_header("Bring Your Own Candidate Dataset",
                "This portal ships with no data pre-loaded. Upload your own candidate CSV, or load the bundled synthetic demo dataset.")
    left, right = st.columns([2, 3])
    with left:
        with card("Upload Custom Dataset"):
            up = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
            if up is not None:
                try:
                    new_df = pd.read_csv(up)
                    missing = [c for c in REQUIRED_COLS if c not in new_df.columns]
                    if missing:
                        st.error("Missing required columns: " + ", ".join(missing))
                    elif st.button("Apply Uploaded Dataset", key="apply_upload"):
                        set_data(new_df)
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")
            st.caption("Don't have a file handy? Explore with realistic synthetic data instead:")
            if st.button("Load Sample Demo Dataset", key="load_sample"):
                set_data(load_sample_dataset())
                st.rerun()
    with right:
        with card("Active Dataset Preview"):
            df = get_data()
            if df is None:
                empty_state("Nothing loaded yet -- upload a file or load the sample dataset.")
            else:
                st.success(f"Dataset loaded: {len(df):,} rows, {df.shape[1]} columns.")
                show_table(df.head(50), height=340)


# ============================================================================ OVERVIEW
def render_overview_page():
    if not _need_data():
        return
    df = get_data()
    page_header("Executive Overview", "Headcount, performance, retention and pay positioning at a glance.")
    hipo = 100 * (df["High_Potential_Program_Flag"] == "Y").mean() if df["High_Potential_Program_Flag"].notna().any() else float("nan")
    attr = 100 * (df["Attrition_Flag"] == "Exited").mean() if df["Attrition_Flag"].notna().any() else float("nan")
    kpi_row(kpi_card(f"{len(df):,}", "Total Candidates"),
            kpi_card("n/a" if pd.isna(hipo) else f"{hipo:.1f}%", "High Potential (HiPo) %", "#C9A227"),
            kpi_card(f"{df['Avg_Performance_Rating'].mean():.2f}", "Avg. Performance Rating", "#1B7A3E"),
            kpi_card("n/a" if pd.isna(attr) else f"{attr:.1f}%", "Historical Attrition Rate", "#B91C1C"))
    c1, c2 = st.columns(2)
    with c1, card("Headcount by Business Unit & Job Level"):
        counts = df.groupby(["Business_Unit", "Job_Level"], observed=True).size().reset_index(name="Count")
        fig = px.bar(counts, x="Business_Unit", y="Count", color="Job_Level", color_discrete_sequence=LEVEL_COLORS,
                     category_orders={"Job_Level": LEVELS})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend_title=None, xaxis_title=None)
        show_plot(fig)
    with c2, card("Performance Distribution by Business Unit"):
        fig = px.box(df, x="Business_Unit", y="Avg_Performance_Rating", color="Business_Unit",
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_title=None)
        show_plot(fig)
    c3, c4 = st.columns(2)
    with c3, card("Career Velocity vs. Tenure (one panel per Business Unit)"):
        fig = px.scatter(df, x="Years_at_Company", y="Career_Velocity_Score", color="Job_Level", facet_col="Business_Unit",
                         facet_col_wrap=2, color_discrete_sequence=LEVEL_COLORS, opacity=0.6, height=400,
                         category_orders={"Job_Level": LEVELS})
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        show_plot(fig)
    with c4, card("Compensation Positioning (Compa-Ratio) by Level"):
        fig = px.box(df, x="Job_Level", y="Compa_Ratio", color="Job_Level", color_discrete_sequence=LEVEL_COLORS,
                     category_orders={"Job_Level": LEVELS})
        fig.add_hline(y=1.0, line_dash="dash", line_color="#943126")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_title=None)
        show_plot(fig)


# ============================================================================ LAYER 1
def _slider(label, key, lo, hi, step=None):
    st.session_state.setdefault(key, DEFAULT_L1[key])
    return st.slider(label, lo, hi, key=key, step=step)


def render_layer1_page():
    if not _need_data():
        return
    page_header("Layer 1: Rule-Based Screening Engine",
                "Deterministic HR policy rules. Move any threshold and the distribution updates instantly.")
    left, right = st.columns([3, 9])
    with left, card("Business Rule Thresholds"):
        st.caption("'Ready for Leadership' needs 2 of 3 elite criteria (Analytical / Risk / Leadership) -- "
                   "deliberately selective, so small counts are normal.")
        _slider("Analytical Test Score >", "l1_analytical", 0, 100)
        _slider("Risk & Compliance Score > (elite criteria)", "l1_risk", 0, 100)
        _slider("Leadership Potential Score >", "l1_leadership", 0, 100)
        _slider("Compliance Watchlist Threshold <", "l1_compliance", 0, 100)
        _slider("Engagement Score < (flight risk)", "l1_engagement", 0, 100)
        _slider("Compa-Ratio < (flight risk)", "l1_compa", 0.7, 1.3, 0.01)
        _slider("Avg Performance >= (fast-track)", "l1_perf_high", 1.0, 5.0, 0.1)
        _slider("Avg Performance < (PIP candidate)", "l1_perf_low", 1.0, 5.0, 0.1)
    flagged = enriched()
    with right:
        with sticky_container("l1_chart_sticky"), card("Flag Distribution (live)"):
            order = [f for f in FLAG_ORDER if f != "Steady / Core Contributor"]
            counts = flagged["Layer1_Flag"].astype(str).value_counts().reindex(order).fillna(0).astype(int).reset_index()
            counts.columns = ["Flag", "Count"]
            counts = counts.sort_values("Count")
            fig = px.bar(counts, x="Count", y="Flag", orientation="h", text="Count", color="Flag", color_discrete_map=FLAG_COLORS)
            fig.update_traces(textposition="outside")
            fig.update_layout(height=290, margin=dict(l=10, r=40, t=10, b=10), showlegend=False, yaxis_title=None)
            show_plot(fig)
            n_steady = int((flagged["Layer1_Flag"].astype(str) == "Steady / Core Contributor").sum())
            st.caption(f"+ {n_steady} candidates ({100 * n_steady / len(flagged):.1f}%) are 'Steady / Core Contributor' "
                       "(no rule triggered) and are left off the chart so the actionable flags stay visible.")
    with card("Screened Candidates"):
        f1, f2, f3 = st.columns(3)
        flags = f1.multiselect("Filter by Flag", FLAG_ORDER)
        bus = f2.multiselect("Filter by Business Unit", sorted(flagged["Business_Unit"].unique()))
        lvls = f3.multiselect("Filter by Job Level", [l for l in LEVELS if l in set(flagged["Job_Level"].astype(str))])
        t = flagged
        if flags: t = t[t["Layer1_Flag"].astype(str).isin(flags)]
        if bus: t = t[t["Business_Unit"].isin(bus)]
        if lvls: t = t[t["Job_Level"].astype(str).isin(lvls)]
        cols = ["Employee_ID", "Full_Name", "Business_Unit", "Job_Level", "Avg_Performance_Rating", "Analytical_Test_Score",
                "Risk_Compliance_Score", "Leadership_Potential_Score", "Engagement_Score", "Layer1_Flag"]
        st.caption(f"{len(t)} candidates shown")
        show_table(t[cols], height=340)


# ============================================================================ LAYER 2
def _train_all(method):
    errors = []
    for key, fn in (("retention", ml.train_retention_model), ("hipo", ml.train_hipo_model), ("perf", ml.train_performance_model)):
        try:
            st.session_state[key] = fn(get_data(), method)
        except Exception as e:
            st.session_state[key] = None
            errors.append(f"{key}: {e}")
    st.session_state["models_autotrained"] = True
    return errors


def _signal_card(r) -> str:
    color = SIGNAL_COLORS.get(r["Signal"], "#64748B")
    pill = band_pill(f"{r['Signal']} ({r['Score'] * 100:.0f}%)", color)
    return (f"<div class='rx-card' style='border-left-color:{color}'><div style='display:flex;justify-content:space-between;align-items:baseline;'>"
            f"<h5>{r['Full_Name']} - {r['Job_Level']}, {r['Business_Unit']}</h5>{pill}</div><p>{r['Action']}</p></div>")


def _predictions_tab():
    c1, c2 = st.columns([1, 2])
    method = c1.radio("Algorithm family", ["glm", "rf"], key="ml_method",
                      format_func=lambda m: "Logistic / Linear Regression" if m == "glm" else "Random Forest")
    errors = []
    if c1.button("Train / Refresh Predictions", key="train_btn"):
        with st.spinner("Training 3 models..."):
            errors = _train_all(method)
    elif not st.session_state.get("models_autotrained"):
        with st.spinner("Training the 3 models for the first time..."):
            errors = _train_all(method)
    for e in errors:
        c2.warning(f"Could not train {e}")
    ret, hip, per = (st.session_state.get(k) for k in ("retention", "hipo", "perf"))
    live = sum(x is not None for x in (ret, hip, per))
    c2.info(f"{live} of 3 models live. Model 1 & 2 are classifiers, Model 3 is a regression.")

    en = enriched()
    n_ret = int((ret["scored"]["Retention_Risk_Tier"] == "High").sum()) if ret else "-"
    if hip:
        h = en[(en["HiPo_Likelihood_Tier"] == "High") & (en["High_Potential_Program_Flag"] == "N")]
        n_hip = len(h)
    else:
        n_hip = "-"
    n_watch = int((per["scored"]["Performance_Outlook"] == "Watch - Possible Derailment Risk").sum()) if per else "-"
    kpi_row(kpi_card(n_ret, "High Retention Risk", "#B91C1C"), kpi_card(n_hip, "Hidden HiPo Candidates", "#C9A227"),
            kpi_card(n_watch, "Derailment Watch", "#7C3AED"), kpi_card(f"{live}/3", "Predictions Live"))
    st.caption("**Model 1 - Retention Risk (classification):** how closely an employee's profile resembles people who have left. "
               "**Model 2 - Hidden HiPo (classification):** how closely it resembles current High-Potential employees, even if not yet nominated. "
               "**Model 3 - Performance Outlook (regression):** the rating the profile implies, vs. the actual rating.")
    with card("Priority Action List - Who Needs Attention This Week"):
        acts, totals = ml.build_priority_actions(en, ret, hip, per, per_signal=st.session_state.get("ml_per_signal", 6))
        f1, f2 = st.columns([3, 1])
        picked = f1.multiselect("Show signals", list(SIGNAL_COLORS), default=list(SIGNAL_COLORS), key="ml_signals")
        f2.selectbox("Top N per signal", [3, 6, 10, 20], index=1, key="ml_per_signal")
        if totals:
            st.caption("Total flagged per signal: " + " | ".join(f"{k}: {v}" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])))
        shown = acts[acts["Signal"].isin(picked)]
        if shown.empty:
            st.success("Nothing flagged for the selected signals.")
        else:
            st.markdown("".join(_signal_card(r) for _, r in shown.iterrows()), unsafe_allow_html=True)


def _clustering_tab():
    df = get_data()
    st.markdown("**What this predicts:** talent segments from K-Means clustering on the 7 competency scores -- "
                "groups people with similar strength/gap profiles, independent of the 3 predictions.")
    c1, c2 = st.columns([1, 3])
    k = c1.slider("Number of clusters (k)", 2, 5, 3, key="k_clusters")
    run = c1.button("Run / Refresh Clustering", key="cluster_btn")
    if run or "cluster" not in st.session_state:
        try:
            st.session_state["cluster"] = ml.run_clustering(df, k)
        except Exception as e:
            st.error(f"Clustering failed: {e}")
            return
    cl = st.session_state["cluster"]
    members = df.merge(cl["assignments"], on="Employee_ID")
    members["Competency_Avg"] = members[ml.COMPETENCY_COLS].mean(axis=1).round(1)

    st.markdown("#### What this means - recommended action per cluster")
    summ = members.groupby("Talent_Cluster").agg(n=("Employee_ID", "count"), avg=("Competency_Avg", "mean"),
                                                 top=("Business_Unit", lambda s: s.value_counts().index[0])).sort_values("avg", ascending=False)
    html = ""
    for name, r in summ.iterrows():
        color = CLUSTER_COLORS.get(name, "#0D7377")
        html += (f"<div class='scard' style='border-left-color:{color}'><div style='display:flex;justify-content:space-between;align-items:baseline;'>"
                 f"<h5>{name}</h5>{band_pill(f'{int(r.n)} employees', color)}</div>"
                 f"<p>Avg competency: {r.avg:.1f}/100 &nbsp;|&nbsp; Largest presence: {r.top}</p>"
                 f"<p style='color:#1A2535'><b>Recommended action:</b> {ml.CLUSTER_ACTIONS.get(name, 'Review individually.')}</p></div>")
    st.markdown(html, unsafe_allow_html=True)

    if st.checkbox("Show technical detail (3D PCA projection + cluster profile)", key="show_tech"):
        a, b = st.columns([7, 5])
        with a, card("Cluster Separation (3D PCA)"):
            pca, ax = cl["pca"], cl["axis_names"]
            if "PC3" in pca:
                fig = px.scatter_3d(pca, x="PC1", y="PC2", z="PC3", color="Talent_Cluster", color_discrete_map=CLUSTER_COLORS,
                                    hover_name="Employee_ID", height=460,
                                    labels={f"PC{i+1}": f"PC{i+1}: {ax[f'PC{i+1}']}" for i in range(3)})
                fig.update_traces(marker=dict(size=3.5, opacity=0.8))
            else:
                fig = px.scatter(pca, x="PC1", y="PC2", color="Talent_Cluster", color_discrete_map=CLUSTER_COLORS, height=460)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
            show_plot(fig)
            st.caption(" | ".join(f"PC{i+1}: {ax[f'PC{i+1}']} ({cl['var_explained'][i]}% of variance)" for i in range(len(ax))))
        with b, card("Cluster Profile (standardised mean score)"):
            long = cl["centers"].melt(id_vars="Talent_Cluster", var_name="Competency", value_name="z")
            long["Competency"] = long["Competency"].map(pretty)
            fig = px.bar(long, x="z", y="Competency", color="Talent_Cluster", barmode="group", orientation="h",
                         color_discrete_map=CLUSTER_COLORS, height=460)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None, xaxis_title="z-score")
            show_plot(fig)

    with card("Who's in which cluster"):
        f1, f2, f3 = st.columns(3)
        cs = f1.multiselect("Cluster", sorted(members["Talent_Cluster"].unique()), key="cm_cluster")
        bus = f2.multiselect("Business Unit", sorted(members["Business_Unit"].unique()), key="cm_bu")
        lv = f3.multiselect("Job Level", [l for l in LEVELS if l in set(members["Job_Level"].astype(str))], key="cm_lv")
        t = members
        if cs: t = t[t["Talent_Cluster"].isin(cs)]
        if bus: t = t[t["Business_Unit"].isin(bus)]
        if lv: t = t[t["Job_Level"].astype(str).isin(lv)]
        st.caption(f"{len(t)} employees shown")
        show_table(t[["Employee_ID", "Full_Name", "Business_Unit", "Job_Level", "Talent_Cluster", "Competency_Avg"]]
                   .sort_values(["Talent_Cluster", "Competency_Avg"], ascending=[True, False]), height=330)


def _diagnostics_tab():
    st.caption("Statistical detail for model validation -- not needed for day-to-day use. "
               "See 'Predictions & Prescriptions' for the actionable view.")
    df = get_data()
    specs = [("retention", "Model 1: Retention Risk (classification) - predicts Attrition_Flag = 'Exited'", "Retention_Risk_Prob"),
             ("hipo", "Model 2: HiPo Likelihood (classification) - predicts High_Potential_Program_Flag = 'Y'", "HiPo_Likelihood_Prob"),
             ("perf", "Model 3: Performance Outlook (regression) - predicts Avg_Performance_Rating", "Performance_Gap")]
    for key, title, sort_col in specs:
        res = st.session_state.get(key)
        with card(title):
            if not res:
                st.info("Not trained yet (or the dataset lacks the target column). Use 'Train / Refresh Predictions'.")
                continue
            m = res["metrics"]
            st.caption(f"Algorithm: {res['method'].upper()} | train n = {res['n_train']} | held-out test n = {res['n_test']}")
            a, b = st.columns(2)
            with a:
                if res["kind"] == "classification":
                    x, y = st.columns(2)
                    x.metric("AUC (held-out)", f"{m['AUC']:.2f}")
                    y.metric("Accuracy", f"{m['Accuracy']:.2f}")
                    cm = pd.DataFrame(m["Confusion"], index=["Actual: No", "Actual: Yes"], columns=["Predicted: No", "Predicted: Yes"])
                    st.dataframe(cm)
                else:
                    x, y, z = st.columns(3)
                    x.metric("RMSE", f"{m['RMSE']:.2f}")
                    y.metric("R\u00b2", f"{m['R2']:.2f}")
                    z.metric("MAE", f"{m['MAE']:.2f}")
                    st.caption(f"Employees are flagged only when actual and predicted rating differ by more than {m['Gap_Threshold']:.2f} (1.25 x RMSE).")
            with b:
                imp = res["importance"].sort_values("Importance")
                fig = px.bar(imp, x="Importance", y="Feature", orientation="h", height=330)
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                show_plot(fig)
            with st.expander("Full-population scores"):
                names = df[["Employee_ID", "Full_Name", "Business_Unit", "Job_Level"]]
                show_table(names.merge(res["scored"], on="Employee_ID").sort_values(sort_col, ascending=False), height=300)


def render_layer2_page():
    if not _need_data():
        return
    page_header("Layer 2: Machine-Learning Predictions & Prescriptions",
                "Three predictions (two classifiers, one regression) plus talent segmentation, turned into concrete next actions.")
    t1, t2, t3 = st.tabs(["Predictions & Prescriptions", "Talent Clustering", "Model Diagnostics (Technical)"])
    with t1:
        _predictions_tab()
    with t2:
        _clustering_tab()
    with t3:
        _diagnostics_tab()
