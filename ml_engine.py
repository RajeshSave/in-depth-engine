"""
ml_engine.py  --  LAYER 2 (machine learning) + Succession Readiness + Priority Actions

Pure functions (no Streamlit code) so they can be tested on their own.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COMPETENCY_COLS = ["Analytical_Test_Score", "Risk_Compliance_Score", "Leadership_Potential_Score",
                   "Communication_Score", "Technical_Skill_Score", "Client_Management_Score",
                   "Engagement_Score"]

NUMERIC_PREDICTORS = [
    "Age", "Years_of_Experience", "Years_at_Company", "Years_in_Current_Role",
    "Time_Since_Last_Promotion_Years", "Number_of_Promotions", *COMPETENCY_COLS,
    "Feedback_360_Score", "Training_Hours_Annual", "Absenteeism_Days_Annual", "Compa_Ratio",
    "Bonus_Pct", "Avg_Performance_Rating", "Career_Velocity_Score", "Performance_Trajectory",
    "Promotions_Last_5_Years",
]
CATEGORICAL_PREDICTORS = ["Job_Level", "Business_Unit"]

# Model 3 predicts Avg_Performance_Rating, so it must NOT see the rating itself or
# Performance_Trajectory (which is computed from the same rating columns -- leakage).
PERF_NUMERIC = [c for c in NUMERIC_PREDICTORS if c not in ("Avg_Performance_Rating", "Performance_Trajectory")]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Performance_Trajectory" in out:
        out["Performance_Trajectory"] = out["Performance_Trajectory"].fillna(0)
    out["Job_Level"] = out["Job_Level"].astype(str)
    out["Business_Unit"] = out["Business_Unit"].astype(str)
    return out


def _preprocessor(numeric, scale: bool):
    num_steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_PREDICTORS),
    ])


def _feature_importance(pipe, numeric) -> pd.DataFrame:
    names = list(numeric) + list(pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_PREDICTORS))
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        vals = np.abs(np.ravel(model.coef_))
    imp = pd.DataFrame({"Feature": names, "Importance": vals}).sort_values("Importance", ascending=False)
    return imp.head(15).reset_index(drop=True)


def assign_tiers(prob: pd.Series) -> pd.Series:
    """Rank-based tiers: top 15% = High, next 25% = Medium, rest = Low.
    (Fixed 0.33/0.66 cut-offs left the High tier empty for imbalanced targets.)"""
    q_hi, q_med = prob.quantile(0.85), prob.quantile(0.60)
    return pd.Series(np.where(prob >= q_hi, "High", np.where(prob >= q_med, "Medium", "Low")), index=prob.index)


def train_classifier(df, target, positive, negative, method="glm", seed=42, label="Score"):
    d = _prep(df)
    d = d[d[target].isin([positive, negative])].copy()
    X, y = d[NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS], (d[target] == positive).astype(int)
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        raise ValueError(f"Not enough examples of both classes in '{target}' to train a model.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    model = (LogisticRegression(max_iter=2000) if method == "glm"
             else RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed))
    pipe = Pipeline([("pre", _preprocessor(NUMERIC_PREDICTORS, scale=(method == "glm"))), ("model", model)])
    pipe.fit(Xtr, ytr)
    p_te = pipe.predict_proba(Xte)[:, 1]
    metrics = {"AUC": roc_auc_score(yte, p_te), "Accuracy": accuracy_score(yte, p_te >= 0.5),
               "Confusion": confusion_matrix(yte, (p_te >= 0.5).astype(int), labels=[0, 1])}
    prob = pd.Series(pipe.predict_proba(X)[:, 1], index=d.index)
    scored = pd.DataFrame({"Employee_ID": d["Employee_ID"], f"{label}_Prob": prob.round(3),
                           f"{label}_Tier": assign_tiers(prob)})
    return {"scored": scored, "metrics": metrics, "importance": _feature_importance(pipe, NUMERIC_PREDICTORS),
            "method": method, "n_train": len(Xtr), "n_test": len(Xte), "kind": "classification"}


def train_retention_model(df, method="glm"):
    return train_classifier(df, "Attrition_Flag", "Exited", "Active", method, label="Retention_Risk")


def train_hipo_model(df, method="glm"):
    return train_classifier(df, "High_Potential_Program_Flag", "Y", "N", method, label="HiPo_Likelihood")


def train_performance_model(df, method="glm", seed=42):
    d = _prep(df)
    d = d[d["Avg_Performance_Rating"].notna()].copy()
    X, y = d[PERF_NUMERIC + CATEGORICAL_PREDICTORS], d["Avg_Performance_Rating"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = LinearRegression() if method == "glm" else RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed)
    pipe = Pipeline([("pre", _preprocessor(PERF_NUMERIC, scale=(method == "glm"))), ("model", model)])
    pipe.fit(Xtr, ytr)
    pred_te = pipe.predict(Xte)
    metrics = {"RMSE": float(np.sqrt(mean_squared_error(yte, pred_te))), "R2": float(r2_score(yte, pred_te)),
               "MAE": float(mean_absolute_error(yte, pred_te))}
    pred = pd.Series(np.clip(pipe.predict(X), 1, 5), index=d.index)
    gap = (pred - d["Avg_Performance_Rating"]).round(2)
    # Flag only gaps that are large relative to the model's own typical error
    # (1.25 x RMSE) -- a fixed +/-0.3 cut-off flagged ~40% of employees as "unusual".
    thr = max(0.3, 1.25 * metrics["RMSE"])
    metrics["Gap_Threshold"] = thr
    outlook = np.where(gap >= thr, "Untapped Potential",
                       np.where(gap <= -thr, "Watch - Possible Derailment Risk", "Stable / As Expected"))
    scored = pd.DataFrame({"Employee_ID": d["Employee_ID"], "Predicted_Performance_Score": pred.round(2),
                           "Performance_Gap": gap, "Performance_Outlook": outlook})
    imp = _feature_importance(pipe, PERF_NUMERIC)
    return {"scored": scored, "metrics": metrics, "importance": imp, "method": method,
            "n_train": len(Xtr), "n_test": len(Xte), "kind": "regression"}


# ---------------------------------------------------------------------------
# K-Means clustering + 3-component PCA
# ---------------------------------------------------------------------------
CLUSTER_LABELS = ["High-Flyers", "Steady Performers", "Development Core", "Emerging Talent", "Foundational Track"]
CLUSTER_ACTIONS = {
    "High-Flyers": "Fast-track into the succession pipeline: prioritise for stretch assignments, executive sponsorship and accelerated promotion review.",
    "Steady Performers": "Core retention focus: sustain engagement via targeted upskilling and recognition; watch for stagnation / flight risk.",
    "Development Core": "Structured development required: route through the Development Rx page for competency-specific interventions.",
    "Emerging Talent": "Early-stage potential: invest in mentorship and visibility; re-assess readiness in 12-18 months.",
    "Foundational Track": "Foundational skill-building priority: pair with formal training and close manager coaching.",
}


def run_clustering(df, k=3, seed=42):
    d = df.dropna(subset=COMPETENCY_COLS).copy()
    Z = StandardScaler().fit_transform(d[COMPETENCY_COLS])
    km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(Z)
    # Re-label by mean competency (highest first) so names stay meaningful.
    order = pd.Series(Z.mean(axis=1)).groupby(km.labels_).mean().sort_values(ascending=False).index.tolist()
    label_map = {raw: CLUSTER_LABELS[i] for i, raw in enumerate(order)}
    d["Talent_Cluster"] = [label_map[c] for c in km.labels_]
    n_pc = min(3, Z.shape[1])
    pca = PCA(n_components=n_pc).fit(Z)
    scores = pd.DataFrame(pca.transform(Z), columns=[f"PC{i+1}" for i in range(n_pc)], index=d.index)
    scores["Talent_Cluster"], scores["Employee_ID"] = d["Talent_Cluster"], d["Employee_ID"]

    def axis_name(loadings):
        top = pd.Series(np.abs(loadings), index=COMPETENCY_COLS).sort_values(ascending=False).index[:2]
        return " & ".join(c.replace("_Score", "").replace("_", " ") for c in top) + " Axis"

    axis_names = {f"PC{i+1}": axis_name(pca.components_[i]) for i in range(n_pc)}
    centers = pd.DataFrame(km.cluster_centers_, columns=COMPETENCY_COLS)
    centers["Talent_Cluster"] = [label_map[i] for i in range(k)]
    return {"assignments": d[["Employee_ID", "Talent_Cluster"]], "pca": scores, "axis_names": axis_names,
            "var_explained": (pca.explained_variance_ratio_ * 100).round(1), "centers": centers}


# ---------------------------------------------------------------------------
# Succession Readiness Score (3-5 year horizon framework)
# ---------------------------------------------------------------------------
HORIZON_PHASES = {
    "Development Phase (3-5 Years Out): 20% Performance / 80% Assessment": 0.20,
    "Standard 3-5-Year Horizon (Recommended): 30% / 70%": 0.30,
    "Selection & Readiness Phase (~1 Year Out): 50% / 50%": 0.50,
}


def compute_readiness(df, perf_weight=0.30, gate_threshold=50):
    out = df.copy()
    out["Performance_Component"] = pd.concat([
        out["Avg_Performance_Rating"] / 5 * 100, out["Feedback_360_Score"],
        out["Technical_Skill_Score"], out["Client_Management_Score"]], axis=1).mean(axis=1).round(1)
    out["Assessment_Component"] = out[["Leadership_Potential_Score", "Analytical_Test_Score",
                                       "Communication_Score", "Risk_Compliance_Score"]].mean(axis=1).round(1)
    out["Succession_Readiness_Score"] = (perf_weight * out["Performance_Component"]
                                         + (1 - perf_weight) * out["Assessment_Component"]).round(1)
    out["Gate_Status"] = np.where(out["Performance_Component"] >= gate_threshold, "Eligible", "Below Performance Gate")
    return out


# ---------------------------------------------------------------------------
# Priority Action List: 7 decision points
# ---------------------------------------------------------------------------
def build_priority_actions(df, retention=None, hipo=None, perf=None, per_signal=8):
    """df must already contain Layer1_Flag and the readiness columns."""
    base = df[["Employee_ID", "Full_Name", "Business_Unit", "Job_Level", "High_Potential_Program_Flag",
               "Layer1_Flag", "Compa_Ratio", "Avg_Performance_Rating", "Assessment_Component", "Gate_Status"]].copy()
    parts = []

    def add(frame, signal, score, action):
        if len(frame):
            parts.append(pd.DataFrame({"Full_Name": frame["Full_Name"], "Business_Unit": frame["Business_Unit"],
                                       "Job_Level": frame["Job_Level"], "Signal": signal,
                                       "Score": np.clip(score, 0, 1), "Action": action}))

    if retention is not None:
        f = base.merge(retention["scored"], on="Employee_ID")
        f = f[f["Retention_Risk_Tier"] == "High"]
        add(f, "Retention Risk", f["Retention_Risk_Prob"], "Schedule a stay conversation; review compensation positioning and workload.")
    if hipo is not None:
        f = base.merge(hipo["scored"], on="Employee_ID")
        f = f[(f["HiPo_Likelihood_Tier"] == "High") & (f["High_Potential_Program_Flag"] == "N")]
        add(f, "Hidden HiPo", f["HiPo_Likelihood_Prob"], "Not HiPo-flagged but profile suggests otherwise -- consider nominating.")
    f = base[(base["Avg_Performance_Rating"] >= 4) & (base["Compa_Ratio"] < 0.90)]
    add(f, "Pay Equity Risk", (0.90 - f["Compa_Ratio"]) / 0.20 + 0.5, "High performer paid below market (Compa-Ratio < 0.90) -- review compensation at next cycle.")
    f = base[(base["Gate_Status"] == "Below Performance Gate") & (base["Assessment_Component"] >= 70)]
    add(f, "Blocked Potential", f["Assessment_Component"] / 100, "High assessment potential but blocked by the performance gate -- coach on execution before advancing.")
    f = base[base["Layer1_Flag"].astype(str) == "Promotion Overdue - Review Needed"]
    add(f, "Promotion Overdue", pd.Series(0.7, index=f.index), "Strong, tenured performer with no recent promotion -- bring to the next calibration / promotion review.")
    if perf is not None:
        f = base.merge(perf["scored"], on="Employee_ID")
        u = f[f["Performance_Outlook"] == "Untapped Potential"]
        add(u, "Untapped Potential", u["Performance_Gap"].abs() / 2 + 0.5, "Profile predicts a higher rating than achieved -- explore role fit, workload or manager support.")
        w = f[f["Performance_Outlook"] == "Watch - Possible Derailment Risk"]
        add(w, "Derailment Watch", w["Performance_Gap"].abs() / 2 + 0.5, "Rating runs ahead of what the profile predicts -- validate sustainability before a scope increase.")
    cols = ["Full_Name", "Business_Unit", "Job_Level", "Signal", "Score", "Action"]
    if not parts:
        return pd.DataFrame(columns=cols), {}
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["Full_Name", "Signal"])
    totals = out["Signal"].value_counts().to_dict()          # full counts per signal
    # Keep only the top few per signal so one noisy signal can't crowd out the others.
    out = (out.sort_values("Score", ascending=False).groupby("Signal", group_keys=False).head(per_signal)
           .sort_values("Score", ascending=False).reset_index(drop=True))
    return out, totals
