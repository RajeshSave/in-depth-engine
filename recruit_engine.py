"""
recruit_engine.py -- recruitment analytics (pure functions, no Streamlit).

Models
  1. Performance   (regression)      first-year performance rating (1-5)
  2. Retention     (classification)  probability of still being employed at 12 months
  3. Offer accept  (classification)  probability a candidate accepts an offer
  Quality-of-Hire prediction = 100 x (0.6 x performance score + 0.4 x retention probability)

Only information available at hiring time is used. Gender, age and age band are NEVER model inputs.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_recruitment_data import CRITICAL_DERAILERS, HDS, HPI, MVPI, PI_FACTORS

PROTECTED = ["Gender", "Age", "Age_Band"]            # never used as model inputs
DERIVED_ASSESS = ["PI_Cognitive_Raw", "PI_Cognitive_Percentile", "PI_Behavioral_Fit", "Hogan_Job_Fit", "Derailer_Risk_Index",
                  "Critical_Derailers_Elevated", "Max_Derailer_Score", "Culture_Alignment_Score"]
ASSESS_NUM = HPI + HDS + MVPI + PI_FACTORS + DERIVED_ASSESS
ATS_NUM = ["Years_Experience", "Certifications_Count", "Prior_Employers_Count", "Avg_Tenure_Prior_Jobs", "Notice_Period_Days",
           "Resume_Screen_Score", "Technical_Test_Score", "Case_Study_Score", "Interview_Rounds", "Interview_Panel_Score",
           "Culture_Fit_Rating", "Reference_Check_Score"]
OFFER_NUM = ["Time_to_Offer_Days", "Offer_to_Expected_Ratio", "Offer_vs_Market_Compa", "Sign_On_Bonus_USD", "Competing_Offers"]
CAT = ["Business_Unit", "Job_Level", "Source_Channel", "Prior_Employer_Type", "University_Tier", "Education_Level"]
PERF_RET_NUM = ATS_NUM + ASSESS_NUM + OFFER_NUM
ACCEPT_NUM = OFFER_NUM + ["Notice_Period_Days", "Interview_Panel_Score", "Years_Experience", "Resume_Screen_Score", "Hogan_Job_Fit"]

REQUIRED_RECRUIT_COLS = ["Candidate_ID", "Business_Unit", "Job_Level", "Application_Date", "Source_Channel", "Funnel_Status", "Joined"]
STATUS_PIPELINE = ("In Process", "Offer Extended", "Offer Accepted - Pending")

CAT_READABLE = {"Business_Unit": "Business unit", "Job_Level": "Level", "Source_Channel": "Source", "Education_Level": "Education"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numerics, add any missing optional columns, tidy categoricals."""
    d = df.copy()
    for c in set(PERF_RET_NUM + ACCEPT_NUM):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in CAT:
        d[c] = d[c].fillna("Unknown").astype(str) if c in d else "Unknown"
    for c in ("Perf_Rating_12M", "Quality_of_Hire_Index"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in ("Retained_12M", "Outcome_Observed", "Offer_Accepted", "Background_Check_Clear", "Current_Status", "Gender", "Age_Band",
              "Hiring_Manager_ID", "Max_Derailer_Name", "PI_Reference_Pattern", "Exit_Type"):
        if c not in d:
            d[c] = None
    for c in ("Application_Date", "Screen_Date", "Assessment_Date", "Interview_Date", "Offer_Date", "Decision_Date", "Join_Date"):
        d[c] = pd.to_datetime(d[c], errors="coerce") if c in d else pd.NaT
    return d


def _pre(numeric, scale):
    steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(steps), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def _importance(pipe, numeric):
    names = list(numeric) + list(pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT))
    m = pipe.named_steps["model"]
    vals = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(np.ravel(m.coef_))
    return pd.DataFrame({"Feature": names, "Importance": vals}).sort_values("Importance", ascending=False).head(15).reset_index(drop=True)


def _baseline(train: pd.DataFrame, numeric):
    base = {c: float(train[c].median()) if train[c].notna().any() else 0.0 for c in numeric}
    base.update({c: train[c].mode().iloc[0] for c in CAT})
    return base


def _fit_one(kind, X, y, numeric, method, seed=42):
    if kind == "reg":
        model = Ridge(alpha=300.0) if method == "glm" else RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=0.4, random_state=seed, n_jobs=-1)
        cv = KFold(5, shuffle=True, random_state=seed)
    else:
        model = (LogisticRegression(C=0.05, max_iter=3000) if method == "glm"
                 else RandomForestClassifier(n_estimators=200, min_samples_leaf=8, max_features=0.4, random_state=seed, n_jobs=-1))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = Pipeline([("pre", _pre(numeric, scale=(method == "glm"))), ("model", model)])
    X = X[numeric + CAT]
    if kind == "reg":
        oof = pd.Series(cross_val_predict(pipe, X, y, cv=cv), index=X.index)
        metrics = {"R2 (cross-validated)": r2_score(y, oof), "RMSE (cross-validated)": float(np.sqrt(mean_squared_error(y, oof)))}
    else:
        oof = pd.Series(cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1], index=X.index)
        metrics = {"AUC (cross-validated)": roc_auc_score(y, oof)}
    pipe.fit(X, y)
    return {"pipe": pipe, "numeric": numeric, "oof": oof, "metrics": metrics, "importance": _importance(pipe, numeric),
            "n": len(y), "baseline": _baseline(X, numeric), "kind": kind, "method": method}


def train_recruit_models(df: pd.DataFrame, method: str = "glm") -> dict:
    d = prepare(df)
    obs = d[(d["Outcome_Observed"] == "Y")].copy()
    out, errors = {}, []
    perf = obs[obs["Perf_Rating_12M"].notna()]
    ret = obs[obs["Retained_12M"].isin(["Y", "N"])]
    acc = d[d["Offer_Accepted"].isin(["Y", "N"])]
    specs = [("perf", perf, "reg", perf["Perf_Rating_12M"], PERF_RET_NUM, 150),
             ("ret", ret, "clf", (ret["Retained_12M"] == "N").astype(int), PERF_RET_NUM, 150),    # target: EARLY EXIT
             ("acc", acc, "clf", (acc["Offer_Accepted"] == "Y").astype(int), ACCEPT_NUM, 200)]
    for key, sub, kind, y, numeric, min_n in specs:
        try:
            if len(sub) < min_n or (kind == "clf" and (y.sum() < 20 or (len(y) - y.sum()) < 20)):
                raise ValueError(f"only {len(sub)} usable rows (need {min_n}+ with both outcomes)")
            out[key] = _fit_one(kind, sub, y, numeric, method)
        except Exception as e:
            out[key] = None
            errors.append(f"{ {'perf': 'Performance', 'ret': 'Retention', 'acc': 'Offer-accept'}[key] } model: {e}")
    out["errors"] = errors
    out["method"] = method
    return out


def _predict(models, d):
    res = pd.DataFrame(index=d.index)
    if models.get("perf"):
        res["Pred_Perf_12M"] = np.clip(models["perf"]["pipe"].predict(d[PERF_RET_NUM + CAT]), 1, 5)
    if models.get("ret"):
        res["Retention_12M_Prob"] = 1 - models["ret"]["pipe"].predict_proba(d[PERF_RET_NUM + CAT])[:, 1]
    if models.get("acc"):
        res["Accept_Prob"] = models["acc"]["pipe"].predict_proba(d[ACCEPT_NUM + CAT])[:, 1]
    return res


def quality_of_hire(perf, ret):
    return 100 * (0.6 * (perf - 1) / 4 + 0.4 * ret)


def score_candidates(models, df) -> pd.DataFrame:
    """Predictions + recommendation for every candidate that has Hogan/PI assessments."""
    d = prepare(df)
    d = d[d["HPI_Adjustment"].notna()].copy()
    s = _predict(models, d)
    d = pd.concat([d, s], axis=1)
    if {"Pred_Perf_12M", "Retention_12M_Prob"} <= set(d.columns):
        d["QoH_Pred"] = quality_of_hire(d["Pred_Perf_12M"], d["Retention_12M_Prob"]).round(1)
        pool = d[d["Interview_Panel_Score"].notna()]["QoH_Pred"]          # decisions are made from the interviewed pool
        d["QoH_Percentile"] = d["QoH_Pred"].apply(lambda v: (pool < v).mean() * 100).round(0)
        d["Red_Flags"] = d.apply(_flags, axis=1)
        d["Recommendation"] = d.apply(_recommend, axis=1)
    return d


def _flags(r) -> str:
    f = []
    if pd.notna(r.get("Max_Derailer_Score")) and r["Max_Derailer_Score"] >= 90:
        f.append(f"High-risk derailer: {r['Max_Derailer_Name']} ({r['Max_Derailer_Score']:.0f})")
    if (pd.notna(r.get("Hogan_Job_Fit")) and r["Hogan_Job_Fit"] < 35) or (pd.notna(r.get("PI_Behavioral_Fit")) and r["PI_Behavioral_Fit"] < 30):
        f.append("Role fit below benchmark (Hogan/PI)")
    if pd.notna(r.get("Retention_12M_Prob")) and r["Retention_12M_Prob"] < 0.65:
        f.append("Elevated early-attrition risk")
    if pd.notna(r.get("Accept_Prob")) and r["Accept_Prob"] < 0.5 and str(r.get("Offer_Accepted")) not in ("Y", "N"):
        f.append("Offer may be declined")
    if r.get("Background_Check_Clear") == "N":
        f.append("Background check not clear")
    return "; ".join(f)


def _recommend(r) -> str:
    if pd.isna(r.get("Interview_Panel_Score")):
        return "Awaiting interviews" if str(r.get("Funnel_Status", "")).startswith("In Process") else "Not interviewed"
    p = r["QoH_Percentile"]
    tier = "Strong Hire" if p >= 75 else ("Hire" if p >= 45 else ("Review" if p >= 20 else "Not Recommended"))
    if r["Red_Flags"] and tier in ("Strong Hire", "Hire"):
        tier = {"Strong Hire": "Hire", "Hire": "Review"}[tier]
    return tier


# ----------------------------------------------------------------------------- explanations & what-if
def explain_candidate(models, row: pd.Series, top_k=5):
    """One-at-a-time ablation: how much does each input move the prediction vs. a typical candidate?"""
    out = {}
    for key, label in (("perf", "Performance (rating points)"), ("ret", "12-month retention (probability)")):
        m = models.get(key)
        if not m:
            continue
        X = pd.DataFrame([row[PERF_RET_NUM + CAT].to_dict()])
        X[PERF_RET_NUM] = X[PERF_RET_NUM].apply(pd.to_numeric, errors="coerce")
        base_val = _one(m, X)
        rows = []
        for f in PERF_RET_NUM + CAT:
            Xb = X.copy()
            Xb[f] = m["baseline"][f]
            rows.append((f, _one(m, X) - _one(m, Xb)))
        t = pd.DataFrame(rows, columns=["Feature", "Effect"])
        t = t.reindex(t["Effect"].abs().sort_values(ascending=False).index).head(top_k)
        t["Value"] = [row[f] for f in t["Feature"]]
        out[label] = (base_val, t.reset_index(drop=True))
    return out


def _one(m, X):
    if m["kind"] == "reg":
        return float(m["pipe"].predict(X)[0])
    p_exit = float(m["pipe"].predict_proba(X)[0, 1])
    return 1 - p_exit


def whatif_offer(models, row: pd.Series, ratio=None, sign_on=None, competing=None) -> dict:
    r = row.copy()
    old_ratio = r.get("Offer_to_Expected_Ratio")
    if ratio is not None:
        if pd.notna(r.get("Offer_vs_Market_Compa")) and pd.notna(old_ratio) and old_ratio:
            r["Offer_vs_Market_Compa"] = r["Offer_vs_Market_Compa"] * ratio / old_ratio
        r["Offer_to_Expected_Ratio"] = ratio
    if sign_on is not None:
        r["Sign_On_Bonus_USD"] = sign_on
    if competing is not None:
        r["Competing_Offers"] = competing
    X = pd.DataFrame([r.to_dict()])
    out = {}
    for c in set(PERF_RET_NUM + ACCEPT_NUM):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    s = _predict(models, X)
    out.update({k: float(s[k].iloc[0]) for k in s.columns})
    return out


# ----------------------------------------------------------------------------- backtest
def backtest(models, df):
    """Out-of-fold predictions for past hires: do the model's top/bottom groups really differ in outcomes?
    Compared with ranking by interview score alone (what the process relied on)."""
    if not (models.get("perf") and models.get("ret")):
        return None
    d = prepare(df)
    obs = d[(d["Outcome_Observed"] == "Y") & d["Perf_Rating_12M"].notna() & d["Retained_12M"].isin(["Y", "N"])].copy()
    obs["qoh"] = quality_of_hire(models["perf"]["oof"].reindex(obs.index), models["ret"]["oof"].reindex(obs.index).rsub(1))
    obs["retained"] = (obs["Retained_12M"] == "Y").astype(int)
    rows = []
    for name, col in (("Model (Quality-of-Hire prediction)", "qoh"), ("Interview score only", "Interview_Panel_Score"),
                      ("Hogan job fit only", "Hogan_Job_Fit")):
        x = obs[obs[col].notna()]
        q = pd.qcut(x[col].rank(method="first"), 4, labels=["Bottom 25%", "Lower-mid", "Upper-mid", "Top 25%"])
        g = x.groupby(q, observed=True).agg(Hires=("retained", "size"), Avg_Performance=("Perf_Rating_12M", "mean"), Retention_12M=("retained", "mean"))
        g["Ranked by"] = name
        rows.append(g.reset_index().rename(columns={col: "Group", "index": "Group", "qoh": "Group", "Interview_Panel_Score": "Group", "Hogan_Job_Fit": "Group"}))
    res = pd.concat(rows, ignore_index=True)
    res.columns = ["Group", "Hires", "Avg Performance", "12M Retention", "Ranked by"]
    return res[["Ranked by", "Group", "Hires", "Avg Performance", "12M Retention"]]


# ----------------------------------------------------------------------------- funnel / source / validity / managers
def funnel_counts(d: pd.DataFrame) -> pd.DataFrame:
    stages = [("Applications", len(d)), ("Screened", d["Screen_Date"].notna().sum()),
              ("Assessed (Hogan / PI)", d["Assessment_Date"].notna().sum()), ("Interviewed", d["Interview_Date"].notna().sum()),
              ("Offered", d["Offer_Date"].notna().sum()), ("Accepted", (d["Offer_Accepted"] == "Y").sum()), ("Joined", (d["Joined"] == "Y").sum())]
    f = pd.DataFrame(stages, columns=["Stage", "Candidates"])
    f["Of previous stage"] = (f["Candidates"] / f["Candidates"].shift(1)).round(3)
    f["Of applications"] = (f["Candidates"] / f["Candidates"].iloc[0]).round(3)
    return f


def source_quality(d: pd.DataFrame) -> pd.DataFrame:
    obs = d[d["Outcome_Observed"] == "Y"]
    offers = d[d["Offer_Accepted"].isin(["Y", "N"])]
    g = d.groupby("Source_Channel").agg(Applications=("Candidate_ID", "size"), Hires=("Joined", lambda s: (s == "Y").sum()))
    g["Offer accept rate"] = offers.groupby("Source_Channel")["Offer_Accepted"].apply(lambda s: (s == "Y").mean())
    g["Hires with 12M outcome"] = obs.groupby("Source_Channel").size()
    g["12M retention"] = obs.groupby("Source_Channel")["Retained_12M"].apply(lambda s: (s == "Y").mean())
    g["Avg performance"] = obs.groupby("Source_Channel")["Perf_Rating_12M"].mean()
    g["Avg quality of hire"] = obs.groupby("Source_Channel")["Quality_of_Hire_Index"].mean()
    g["Avg days to offer"] = d.groupby("Source_Channel")["Time_to_Offer_Days"].mean()
    g["Hire rate"] = g["Hires"] / g["Applications"]
    return g.round(3).reset_index().sort_values("Avg quality of hire", ascending=False)


def assessment_validity(d: pd.DataFrame) -> pd.DataFrame:
    obs = d[(d["Outcome_Observed"] == "Y") & d["Perf_Rating_12M"].notna()].copy()
    obs["Retained"] = (obs["Retained_12M"] == "Y").astype(float)
    rows = []
    for c in HPI + HDS + MVPI + PI_FACTORS + DERIVED_ASSESS + ["Technical_Test_Score", "Case_Study_Score", "Interview_Panel_Score", "Resume_Screen_Score"]:
        x = obs[[c, "Perf_Rating_12M", "Retained"]].dropna()
        if len(x) < 40 or x[c].std() == 0:
            continue
        rp, pp = pearsonr(x[c], x["Perf_Rating_12M"])
        rr, pr = pearsonr(x[c], x["Retained"])
        family = ("Hogan HPI" if c in HPI else "Hogan HDS (derailers)" if c in HDS else "Hogan MVPI (values)" if c in MVPI
                  else "Predictive Index" if c in PI_FACTORS or c.startswith("PI_") else "Derived fit scores" if c in DERIVED_ASSESS else "ATS / process")
        rows.append((c, family, len(x), rp, pp, rr, pr))
    return pd.DataFrame(rows, columns=["Measure", "Family", "n", "r with performance", "p (perf)", "r with retention", "p (retention)"]).round(3)


def manager_outcomes(d: pd.DataFrame, min_hires=8) -> pd.DataFrame:
    obs = d[d["Outcome_Observed"] == "Y"]
    g = obs.groupby("Hiring_Manager_ID").agg(Business_Unit=("Business_Unit", "first"), Hires=("Candidate_ID", "size"),
                                              Avg_Performance=("Perf_Rating_12M", "mean"),
                                              Retention_12M=("Retained_12M", lambda s: (s == "Y").mean()))
    return g[g["Hires"] >= min_hires].round(2).reset_index().sort_values("Retention_12M")


# ----------------------------------------------------------------------------- fairness
def adverse_impact(d: pd.DataFrame, group_col: str, min_n=30) -> pd.DataFrame:
    """Selection rate at each funnel step by group; ratio vs the highest-rate group (four-fifths rule)."""
    steps = [("Screened -> Assessed", d["Screen_Date"].notna(), d["Assessment_Date"].notna()),
             ("Assessed -> Interviewed", d["Assessment_Date"].notna(), d["Interview_Date"].notna()),
             ("Interviewed -> Offered", d["Interview_Date"].notna(), d["Offer_Date"].notna()),
             ("Offered -> Joined", d["Offer_Accepted"].isin(["Y", "N"]), d["Joined"] == "Y")]
    rows = []
    for name, denom, num in steps:
        sub = d[denom & d[group_col].notna()]
        g = sub.groupby(group_col).apply(lambda x: pd.Series({"n": len(x), "selected": int(num.loc[x.index].sum())}), include_groups=False)
        g["rate"] = g["selected"] / g["n"]
        ok = g[g["n"] >= min_n]
        top = ok["rate"].max() if len(ok) else np.nan
        for grp, r in g.iterrows():
            ratio = r["rate"] / top if (r["n"] >= min_n and top and top > 0) else np.nan
            rows.append((name, grp, int(r["n"]), int(r["selected"]), round(r["rate"], 3), round(ratio, 2) if pd.notna(ratio) else np.nan,
                         "Below 0.80" if pd.notna(ratio) and ratio < 0.8 else ("OK" if pd.notna(ratio) else "Too few")))
    return pd.DataFrame(rows, columns=["Funnel step", group_col, "Candidates", "Selected", "Selection rate", "Ratio vs highest group", "Four-fifths check"])


def recommendation_impact(scored: pd.DataFrame, group_col: str, min_n=30) -> pd.DataFrame:
    pool = scored[scored["Recommendation"].isin(["Strong Hire", "Hire", "Review", "Not Recommended"])].copy()
    pool["pos"] = pool["Recommendation"].isin(["Strong Hire", "Hire"])
    g = pool.groupby(group_col).agg(Candidates=("pos", "size"), Recommended=("pos", "sum"))
    g["Recommend rate"] = (g["Recommended"] / g["Candidates"]).round(3)
    ok = g[g["Candidates"] >= min_n]
    top = ok["Recommend rate"].max() if len(ok) else np.nan
    g["Ratio vs highest group"] = np.where(g["Candidates"] >= min_n, (g["Recommend rate"] / top).round(2), np.nan)
    g["Four-fifths check"] = np.where(g["Ratio vs highest group"].isna(), "Too few", np.where(g["Ratio vs highest group"] < 0.8, "Below 0.80", "OK"))
    return g.reset_index()
