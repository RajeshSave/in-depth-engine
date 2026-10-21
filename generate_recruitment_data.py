"""
generate_recruitment_data.py

SYNTHETIC RECRUITMENT DATASET: ATS funnel + Hogan + Predictive Index (PI) + post-hire outcomes.

IMPORTANT HONESTY NOTES
  * Hogan and PI are proprietary instruments. This file does NOT reproduce their scoring models.
    It simulates scores on the publicly documented scale names (Hogan HPI / HDS / MVPI percentiles;
    PI behavioural drives, cognitive score, reference pattern) so the app can be built and tested.
    With real data, map your Hogan / PI export columns onto the same names.
  * The relationships between assessments and job outcomes are ASSUMPTIONS chosen to be plausible
    (moderate validity, lots of noise). They are not validity evidence for any real instrument.
    Validate against your own hires before relying on any score for decisions.
  * A mild age-related disparity is DELIBERATELY injected into interview scoring (candidates 40+)
    so the Fairness page has something real to detect. Protected attributes are never model inputs.

STRUCTURE
  Applicants -> Resume screen -> Assessments (Hogan, PI, technical, case) -> Interviews -> Offer
  -> Decision -> Join -> 12-month outcomes (performance rating, retention).
  Candidates who applied recently are still in process (a live pipeline to score), and outcomes are
  only recorded for hires who joined at least 12 months before AS_OF (the rest are blank, as in a real ATS).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.special import erf, expit

warnings.filterwarnings("ignore")

AS_OF = pd.Timestamp("2026-09-30")
START = pd.Timestamp("2021-10-01")
N_APPLICANTS = 12000
SEED = 20260930

BUS = ["Investment Banking", "Risk Management", "Wealth Management", "Quantitative Trading"]
BU_P = [0.34, 0.20, 0.28, 0.18]
BU_CODE = {"Investment Banking": "IB", "Risk Management": "RM", "Wealth Management": "WM", "Quantitative Trading": "QT"}
LEVELS = ["Analyst", "Associate", "VP"]
LOCATIONS = ["New York", "London", "Singapore", "Hong Kong", "Mumbai", "Dubai"]
LOC_MULT = {"New York": 1.15, "London": 1.10, "Singapore": 1.05, "Hong Kong": 1.08, "Mumbai": 0.55, "Dubai": 0.95}
LEVEL_BASE = {"Analyst": 105000, "Associate": 160000, "VP": 240000}

HPI = ["HPI_Adjustment", "HPI_Ambition", "HPI_Sociability", "HPI_Interpersonal_Sensitivity", "HPI_Prudence",
       "HPI_Inquisitive", "HPI_Learning_Approach"]
HDS = ["HDS_Excitable", "HDS_Skeptical", "HDS_Cautious", "HDS_Reserved", "HDS_Leisurely", "HDS_Bold",
       "HDS_Mischievous", "HDS_Colorful", "HDS_Imaginative", "HDS_Diligent", "HDS_Dutiful"]
MVPI = ["MVPI_Recognition", "MVPI_Power", "MVPI_Hedonism", "MVPI_Altruistic", "MVPI_Affiliation", "MVPI_Tradition",
        "MVPI_Security", "MVPI_Commerce", "MVPI_Aesthetics", "MVPI_Science"]
PI_FACTORS = ["PI_Dominance", "PI_Extraversion", "PI_Patience", "PI_Formality"]

# Firm culture benchmark on the MVPI scales (percentile) -- used for the culture-alignment score.
CULTURE_BENCH = dict(zip(MVPI, [60, 55, 35, 50, 55, 50, 45, 70, 40, 60]))
# Ideal PI behavioural profile (Dominance, Extraversion, Patience, Formality) per business unit.
PI_IDEAL = {"Investment Banking": [60, 55, 35, 65], "Risk Management": [40, 35, 65, 80],
            "Wealth Management": [50, 75, 65, 50], "Quantitative Trading": [55, 30, 50, 65]}
# Role weights over HPI scales (same order as HPI list) for the Hogan job-fit score.
HOGAN_FIT_W = {"Investment Banking": [.30, .25, .10, .05, .15, .05, .10], "Risk Management": [.15, .05, .05, .10, .35, .15, .15],
               "Wealth Management": [.15, .15, .30, .20, .10, .00, .10], "Quantitative Trading": [.15, .10, .00, .00, .15, .35, .25]}
CRITICAL_DERAILERS = {"Investment Banking": ["HDS_Excitable", "HDS_Bold", "HDS_Mischievous", "HDS_Diligent"],
                      "Risk Management": ["HDS_Mischievous", "HDS_Bold", "HDS_Imaginative", "HDS_Colorful"],
                      "Wealth Management": ["HDS_Reserved", "HDS_Skeptical", "HDS_Mischievous", "HDS_Excitable"],
                      "Quantitative Trading": ["HDS_Excitable", "HDS_Cautious", "HDS_Leisurely", "HDS_Reserved"]}
# Illustrative PI reference-pattern prototypes (D, E, P, F). Replace with the real PI pattern from your export.
PI_PATTERNS = {"Maverick": [80, 70, 20, 20], "Controller": [75, 25, 50, 75], "Captain": [80, 60, 40, 60],
               "Strategist": [70, 30, 40, 80], "Persuader": [60, 80, 40, 30], "Promoter": [55, 85, 45, 35],
               "Altruist": [30, 55, 80, 40], "Collaborator": [35, 75, 75, 40], "Specialist": [30, 30, 75, 75],
               "Analyzer": [40, 25, 55, 85], "Operator": [45, 40, 70, 60], "Adapter": [50, 50, 50, 50]}
# Latent trait order: g, adjustment, ambition, sociability, interpersonal, prudence, inquisitive, learning
PERF_W = {"Investment Banking": [.25, .30, .20, .05, 0, .10, 0, .10], "Risk Management": [.25, .10, 0, 0, 0, .35, .15, .15],
          "Wealth Management": [.10, .15, .15, .35, .20, .05, 0, 0], "Quantitative Trading": [.40, .10, 0, 0, 0, .10, .25, .15]}
BU_SHIFT = {"Investment Banking": [0.10, 0.20, 0.40, 0.10, -0.10, 0.00, 0.00, 0.00],
            "Risk Management": [0.05, 0.00, -0.10, -0.10, 0.00, 0.50, 0.20, 0.00],
            "Wealth Management": [-0.05, 0.10, 0.10, 0.50, 0.30, 0.00, -0.20, -0.10],
            "Quantitative Trading": [0.40, 0.00, 0.00, -0.30, -0.10, 0.10, 0.50, 0.20]}
MVPI_SHIFT = {"Investment Banking": {"MVPI_Commerce": .6, "MVPI_Recognition": .5, "MVPI_Power": .5},
              "Risk Management": {"MVPI_Security": .6, "MVPI_Tradition": .5, "MVPI_Science": .3},
              "Wealth Management": {"MVPI_Affiliation": .5, "MVPI_Altruistic": .4, "MVPI_Commerce": .4},
              "Quantitative Trading": {"MVPI_Science": .7, "MVPI_Aesthetics": .3, "MVPI_Hedonism": .2}}
SOURCES = ["Employee Referral", "Campus", "Recruitment Agency", "Job Board / LinkedIn", "Careers Site", "Internal Mobility"]
EMPLOYER_TYPES = ["Bulge Bracket Bank", "Boutique / Mid-Market", "Commercial Bank", "Asset / Wealth Manager",
                  "Hedge Fund / Prop Firm", "Fintech / Tech", "Big 4 / Consulting", "Other"]


def _pct(z):
    return np.clip(np.round(100 * 0.5 * (1 + erf(z / np.sqrt(2)))), 1, 99)


def _std(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _noisy(z, rng, reliability=0.81):
    """Assessment score = true standing + measurement error (reliability ~0.8, typical for these tools)."""
    return np.sqrt(reliability) * z + np.sqrt(1 - reliability) * rng.standard_normal(len(z))


def generate_recruitment_dataset(n=N_APPLICANTS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bu = rng.choice(BUS, n, p=BU_P)
    level = rng.choice(LEVELS, n, p=[0.50, 0.32, 0.18])
    yoe = np.where(level == "Analyst", rng.uniform(0, 3, n), np.where(level == "Associate", rng.uniform(2, 8, n), rng.uniform(6, 15, n))).round(1)

    # ----------------------------------------------------------------- demographics (not model inputs)
    gender = rng.choice(["Male", "Female"], n, p=[0.57, 0.43])
    edu = np.empty(n, dtype=object)
    for lv, pr in (("Analyst", [.55, .40, .05]), ("Associate", [.30, .35, .35]), ("VP", [.20, .30, .50])):
        m = level == lv
        edu[m] = rng.choice(["Bachelors", "Masters", "MBA"], m.sum(), p=pr)
    age = np.clip(np.round(22 + yoe + np.select([edu == "Masters", edu == "MBA"], [1.5, 3.5], 0) + rng.normal(0.5, 1.6, n)), 21, 62).astype(int)
    age_band = np.select([age < 30, age < 40], ["Under 30", "30-39"], "40+")
    uni_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.22, 0.43, 0.35])
    location = np.empty(n, dtype=object)
    for b in BUS:
        m = bu == b
        pr = {"Investment Banking": [.30, .25, .12, .13, .12, .08], "Risk Management": [.20, .22, .14, .12, .22, .10],
              "Wealth Management": [.22, .20, .20, .16, .10, .12], "Quantitative Trading": [.30, .22, .14, .12, .14, .08]}[b]
        location[m] = rng.choice(LOCATIONS, m.sum(), p=pr)

    # ----------------------------------------------------------------- ATS background
    src_p = np.where(level[:, None] == "Analyst", [[0.14, 0.26, 0.10, 0.28, 0.18, 0.04]], [[0.18, 0.00, 0.26, 0.32, 0.18, 0.06]])
    source = np.array([rng.choice(SOURCES, p=p / p.sum()) for p in src_p])
    source = np.where((source == "Campus") & (yoe > 1.5), "Job Board / LinkedIn", source)
    campus = source == "Campus"
    employer = rng.choice(EMPLOYER_TYPES, n, p=[.17, .14, .15, .14, .08, .10, .12, .10])
    employer = np.where(campus, "None (Campus)", employer)
    n_emp = np.where(campus, 0, np.clip(rng.poisson(np.maximum(yoe, 0.5) / 3.0) + 1, 1, 7))
    avg_tenure = np.where(n_emp > 0, np.round(np.clip(yoe / np.maximum(n_emp, 1) * rng.uniform(0.7, 1.3, n), 0.3, 12), 1), np.nan)
    certs = rng.binomial(2, np.where(np.isin(bu, ["Risk Management", "Quantitative Trading"]), 0.35, 0.25))
    notice = np.where(campus, rng.choice([0, 30], n), rng.choice([30, 60, 90], n, p=[.35, .40, .25]))

    # ----------------------------------------------------------------- latent traits -> assessments
    # order: g, adjustment, ambition, sociability, interpersonal, prudence, inquisitive, learning
    R = np.eye(8)
    for i, j, v in [(0, 6, .35), (0, 7, .25), (1, 2, .25), (1, 3, .30), (1, 4, .35), (1, 5, .25), (2, 3, .35), (3, 4, .20),
                    (5, 7, .15), (6, 7, .45), (4, 5, .15), (0, 2, .05), (0, 1, .05)]:
        R[i, j] = R[j, i] = v
    w, vec = np.linalg.eigh(R)
    R = (vec * np.clip(w, 1e-3, None)) @ vec.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    Z = rng.standard_normal((n, 8)) @ np.linalg.cholesky(R).T
    for b in BUS:
        Z[bu == b] += np.array(BU_SHIFT[b])
    Zo = np.column_stack([_noisy(Z[:, k], rng) for k in range(8)])       # what the assessments "see"

    out = {}
    for k, name in enumerate(HPI):
        out[name] = _pct(Zo[:, k + 1])                                   # Hogan HPI: adjustment .. learning approach

    def hds(weights):
        raw = sum(w_ * Z[:, idx] for idx, w_ in weights.items()) + rng.standard_normal(n) * 0.85
        return _pct(_std(raw))
    # trait indices: 1 adj, 2 amb, 3 soc, 4 ints, 5 prud, 6 inq
    hds_spec = {"HDS_Excitable": {1: -.55}, "HDS_Skeptical": {4: -.40, 1: -.20}, "HDS_Cautious": {2: -.40, 1: -.30},
                "HDS_Reserved": {3: -.60}, "HDS_Leisurely": {2: -.30, 4: -.20}, "HDS_Bold": {2: .50, 4: -.25},
                "HDS_Mischievous": {5: -.40, 2: .20}, "HDS_Colorful": {3: .50}, "HDS_Imaginative": {6: .40, 5: -.10},
                "HDS_Diligent": {5: .50}, "HDS_Dutiful": {4: .40, 5: .10}}
    for name in HDS:
        out[name] = hds(hds_spec[name])
    for name in MVPI:
        z = rng.standard_normal(n)
        for b in BUS:
            z[bu == b] += MVPI_SHIFT[b].get(name, 0.0)
        out[name] = _pct(z)
    out = pd.DataFrame(out)
    bench = np.array([CULTURE_BENCH[c] for c in MVPI])
    culture = np.clip(100 - 1.6 * np.abs(out[MVPI].values - bench).mean(axis=1), 0, 100).round(1)

    # Predictive Index behavioural drives (0-100, 50 = population average) + cognitive
    def pi(weights):
        raw = sum(w_ * Zo[:, idx] for idx, w_ in weights.items()) + rng.standard_normal(n) * 0.75
        return np.clip(np.round(50 + 18 * _std(raw)), 1, 99)
    out["PI_Dominance"] = pi({2: .55, 4: -.30, 1: .10})
    out["PI_Extraversion"] = pi({3: .65, 4: .15})
    out["PI_Patience"] = pi({1: .30, 4: .30, 2: -.30, 5: .10})
    out["PI_Formality"] = pi({5: .60, 6: -.15})
    cog_raw = np.clip(np.round(26 + 7.5 * Zo[:, 0] + rng.normal(0, 2.0, n)), 0, 50)
    out["PI_Cognitive_Raw"] = cog_raw
    out["PI_Cognitive_Percentile"] = np.clip(np.round(100 * 0.5 * (1 + erf(((cog_raw - 26) / 8) / np.sqrt(2)))), 1, 99)
    pf = out[PI_FACTORS].values
    protos = np.array(list(PI_PATTERNS.values()))
    out["PI_Reference_Pattern"] = np.array(list(PI_PATTERNS))[np.argmin(((pf[:, None, :] - protos[None]) ** 2).sum(-1), axis=1)]
    ideal = np.array([PI_IDEAL[b] for b in bu])
    out["PI_Behavioral_Fit"] = np.round(100 * np.exp(-((np.sqrt(((pf - ideal) ** 2).sum(axis=1))) / 55) ** 2), 1)

    # Derailer summary + Hogan job fit (role-specific)
    hds_df = out[HDS]
    crit = np.array([[hds_df[c].iloc[i] for c in CRITICAL_DERAILERS[bu[i]]] for i in range(n)])
    out["Derailer_Risk_Index"] = crit.mean(axis=1).round(1)
    out["Critical_Derailers_Elevated"] = (crit >= 70).sum(axis=1)
    out["Max_Derailer_Score"] = hds_df.max(axis=1)
    out["Max_Derailer_Name"] = hds_df.idxmax(axis=1).str.replace("HDS_", "")
    hw = np.array([HOGAN_FIT_W[b] for b in bu])
    weighted = (out[HPI].values * hw).sum(axis=1)
    out["Hogan_Job_Fit"] = np.clip(weighted - 0.5 * np.maximum(0, out["Derailer_Risk_Index"].values - 55), 0, 100).round(1)
    out["Culture_Alignment_Score"] = culture

    # ----------------------------------------------------------------- resume screen + test scores
    skill = 0.5 * Z[:, 0] + np.sqrt(0.75) * rng.standard_normal(n)
    prestige = np.select([employer == "Bulge Bracket Bank", employer == "Hedge Fund / Prop Firm", employer == "Boutique / Mid-Market",
                          employer == "Asset / Wealth Manager"], [8, 6, 5, 3], 0)
    resume = np.clip(48 + 7 * np.select([uni_tier == "Tier 1", uni_tier == "Tier 3"], [1, -1], 0) + 3 * certs + prestige
                     + 3.0 * np.minimum(yoe, 6) / 6 + 2.5 * Z[:, 0] + rng.normal(0, 11, n), 0, 100).round(1)
    tech = np.clip(60 + 10 * (0.6 * Z[:, 0] + 0.4 * skill) + rng.normal(0, 9, n), 0, 100).round(1)
    case = np.clip(58 + 8 * (0.45 * Z[:, 0] + 0.25 * Z[:, 3] + 0.15 * Z[:, 4] + 0.15 * skill) + rng.normal(0, 10, n), 0, 100).round(1)

    # ----------------------------------------------------------------- funnel decisions
    passed_screen = rng.random(n) < expit((resume - 49) / 8)
    assessed = passed_screen & (rng.random(n) < 0.85)
    invited = assessed & (rng.random(n) < expit((0.5 * tech + 0.3 * case + 0.2 * resume - 56) / 6))
    age_pen = np.where(age_band == "40+", -11.0, 0.0)                   # deliberately injected disparity (see header)
    interview = np.clip(55 + 4 * Z[:, 0] + 3 * Z[:, 3] + 2.5 * Z[:, 1] + 3 * (uni_tier == "Tier 1") + age_pen + rng.normal(0, 8.5, n), 0, 100).round(1)
    culture_fit = np.clip(np.round((3 + 0.012 * (culture - 60) + 0.25 * Z[:, 3] + rng.normal(0, 0.6, n)) * 2) / 2, 1, 5)
    refcheck = np.clip(72 + 4 * Z[:, 5] + rng.normal(0, 8, n), 30, 100).round(1)
    bg_clear = rng.random(n) < 0.97
    comp = 0.4 * interview + 0.25 * tech + 0.2 * case + 0.15 * culture_fit * 20
    offered = invited & bg_clear & (rng.random(n) < expit((comp - 59) / 5))

    exp_comp = np.round(np.array([LEVEL_BASE[l] for l in level]) * np.array([LOC_MULT[x] for x in location])
                        * (1 + 0.012 * yoe) * rng.normal(1.0, 0.07, n), -2)
    ratio = np.clip(0.98 + 0.03 * _std(comp) + rng.normal(0, 0.05, n), 0.8, 1.25).round(3)
    offer_comp = np.round(exp_comp * ratio, -2)
    market_mid = np.array([LEVEL_BASE[l] for l in level]) * np.array([LOC_MULT[x] for x in location])
    offer_compa = (offer_comp / market_mid).round(3)
    signon = np.where(rng.random(n) < 0.45, rng.choice([5000, 10000, 20000, 40000], n), 0)
    competing = np.clip(rng.poisson(0.7 + 0.5 * (_std(comp) > 0.5)), 0, 3)
    t_offer_days = np.clip(rng.gamma(4, 4, n), 5, 60).round()
    src_acc = np.select([source == "Employee Referral", source == "Internal Mobility", source == "Campus", source == "Recruitment Agency"],
                        [0.5, 0.8, 0.4, -0.2], 0.0)
    acc_logit = (1.0 + 7.0 * (ratio - 1) + 0.00003 * signon - 0.55 * competing + src_acc - 0.008 * (notice - 60)
                 - 0.02 * (t_offer_days - 30) + np.select([bu == "Investment Banking", bu == "Quantitative Trading"], [0.2, 0.1], 0)
                 + rng.normal(0, 0.5, n))
    accepted = offered & (rng.random(n) < expit(acc_logit))
    noshow = accepted & (rng.random(n) < 0.03)
    joined_flag = accepted & ~noshow

    # ----------------------------------------------------------------- timeline
    app_date = START + pd.to_timedelta(rng.integers(0, (AS_OF - START).days + 1, n), unit="D")
    d = lambda lo, hi: pd.to_timedelta(rng.integers(lo, hi, n), unit="D")
    t1 = app_date + d(2, 10)
    t2 = t1 + d(4, 14)
    t2r = t2 + d(2, 6)
    t3 = t2r + d(7, 22)
    t3r = t3 + pd.to_timedelta(t_offer_days.astype(int), unit="D")
    t4 = t3r + d(3, 10)
    tj = t4 + pd.to_timedelta(notice.astype(int), unit="D")
    screened = t1 <= AS_OF
    assess_done = assessed & (t2 <= AS_OF)
    interview_done = invited & (t3 <= AS_OF)
    offer_made = offered & (t3r <= AS_OF)
    decision_known = offered & (t4 <= AS_OF)
    status = np.select([
        ~screened, ~assessed, ~(t2r <= AS_OF), ~invited, ~(t3r <= AS_OF), ~offered, ~(t4 <= AS_OF), ~accepted, noshow, ~(tj <= AS_OF)],
        ["In Process - Screening", "Rejected - Screen", "In Process - Assessment", "Rejected - Post-Assessment",
         "In Process - Interviews", "Rejected - Post-Interview", "Offer Extended - Awaiting Decision", "Offer Declined",
         "Offer Accepted - No-Show", "Offer Accepted - Pending Join"], "Joined")
    # a candidate whose screen decision is not yet due but who would have been rejected stays "In Process"
    final_joined = (status == "Joined")

    # ----------------------------------------------------------------- post-hire outcomes (joiners only)
    mgr_ids = {b: [f"HM-{BU_CODE[b]}-{i + 1:02d}" for i in range(k)] for b, k in zip(BUS, [22, 12, 16, 10])}
    manager = np.array([rng.choice(mgr_ids[b]) for b in bu])
    all_m = sorted({m for v in mgr_ids.values() for m in v})
    perf_eff = dict(zip(all_m, rng.normal(0, 0.35, len(all_m))))
    ret_eff = dict(zip(all_m, rng.normal(0, 0.40, len(all_m))))
    perf_lat = np.array([Z[i] @ np.array(PERF_W[bu[i]]) for i in range(n)])
    dri_z = _std(out["Derailer_Risk_Index"].values)
    rating = (3.35 + 0.55 * perf_lat - 0.12 * np.maximum(0, dri_z) + 0.45 * np.array([perf_eff[m] for m in manager])
              + 0.010 * (out["PI_Behavioral_Fit"].values - 50) / 10 + rng.normal(0, 0.55, n))
    perf12 = np.clip(np.round(rating * 2) / 2, 1, 5)
    perf_z = _std(rating)
    hjf_z, pif_z, cult_z = _std(out["Hogan_Job_Fit"].values), _std(out["PI_Behavioral_Fit"].values), _std(culture)
    hop_z = _std(np.nan_to_num(avg_tenure, nan=np.nanmedian(avg_tenure)) * -1)
    src_ret = np.select([source == "Employee Referral", source == "Recruitment Agency", source == "Campus", source == "Internal Mobility",
                         source == "Job Board / LinkedIn"], [-0.45, 0.35, -0.25, -0.5, 0.10], 0.0)
    ret_logit = (-1.50 - 0.40 * hjf_z - 0.25 * pif_z - 0.25 * cult_z + 0.30 * dri_z + 0.55 * (competing >= 2)
                 - 4.5 * (ratio - 1) + 0.30 * hop_z + src_ret - 0.30 * perf_z
                 + np.array([ret_eff[m] for m in manager]) + np.select([bu == "Investment Banking", bu == "Risk Management"], [0.25, -0.2], 0)
                 + 0.2 * (level == "Analyst") + rng.normal(0, 0.4, n))
    early_exit = rng.random(n) < expit(ret_logit)
    exit_month = np.where(early_exit, np.clip(np.ceil(12 * rng.random(n) ** 0.8), 1, 12), np.nan)
    invol = early_exit & (rng.random(n) < expit(-1.0 - 1.5 * (perf12 - 3)))
    ramp = np.clip(np.round(5.5 - 1.2 * perf_z + rng.normal(0, 1.2, n), 1), 1, 12)
    join_age_days = (AS_OF - tj).days
    observed = final_joined & (join_age_days >= 365)
    exited_by_now = final_joined & early_exit & (exit_month * 30.4 <= join_age_days)

    df = pd.DataFrame({
        "Candidate_ID": [f"CAND-{i:05d}" for i in range(1, n + 1)],
        "Requisition_ID": [f"REQ-{BU_CODE[b]}-{a:%Y%m}-{k}" for b, a, k in zip(bu, app_date, rng.integers(1, 9, n))],
        "Hiring_Manager_ID": manager, "Business_Unit": bu, "Job_Level": level, "Location": location,
        "Application_Date": app_date, "Source_Channel": source,
        "Gender": gender, "Age": age, "Age_Band": age_band,
        "Education_Level": edu, "University_Tier": uni_tier, "Years_Experience": yoe, "Certifications_Count": certs,
        "Prior_Employer_Type": employer, "Prior_Employers_Count": n_emp, "Avg_Tenure_Prior_Jobs": avg_tenure, "Notice_Period_Days": notice,
        "Resume_Screen_Score": resume,
    })
    df["Screen_Date"] = pd.Series(t1).where(screened).values
    df["Assessment_Date"] = pd.Series(t2).where(assess_done).values
    df["Interview_Date"] = pd.Series(t3).where(interview_done).values
    df["Offer_Date"] = pd.Series(t3r).where(offer_made).values
    df["Decision_Date"] = pd.Series(t4).where(decision_known).values
    df["Join_Date"] = pd.Series(tj).where(final_joined).values
    df["Technical_Test_Score"] = np.where(assess_done, tech, np.nan)
    df["Case_Study_Score"] = np.where(assess_done, case, np.nan)
    for c in out.columns:
        df[c] = np.where(assess_done, out[c].values, np.nan) if out[c].dtype != object else np.where(assess_done, out[c].values, None)
    df["Interview_Rounds"] = np.where(interview_done, rng.integers(2, 6, n), np.nan)
    df["Interview_Panel_Score"] = np.where(interview_done, interview, np.nan)
    df["Culture_Fit_Rating"] = np.where(interview_done, culture_fit, np.nan)
    df["Reference_Check_Score"] = np.where(interview_done, refcheck, np.nan)
    df["Background_Check_Clear"] = np.where(interview_done, np.where(bg_clear, "Y", "N"), None)
    df["Time_to_Offer_Days"] = np.where(offer_made, (t3r - app_date).days, np.nan)
    df["Expected_Comp_USD"] = np.where(offer_made, exp_comp, np.nan)
    df["Offer_Comp_USD"] = np.where(offer_made, offer_comp, np.nan)
    df["Offer_to_Expected_Ratio"] = np.where(offer_made, ratio, np.nan)
    df["Offer_vs_Market_Compa"] = np.where(offer_made, offer_compa, np.nan)
    df["Sign_On_Bonus_USD"] = np.where(offer_made, signon, np.nan)
    df["Competing_Offers"] = np.where(offer_made, competing, np.nan)
    df["Offer_Accepted"] = np.where(decision_known, np.where(accepted, "Y", "N"), None)
    df["Funnel_Status"] = status
    df["Joined"] = np.where(final_joined, "Y", "N")
    df["Outcome_Observed"] = np.where(observed, "Y", "N")
    df["Perf_Rating_12M"] = np.where(observed, perf12, np.nan)
    df["Retained_12M"] = np.where(observed, np.where(early_exit, "N", "Y"), None)
    df["Exit_Month"] = np.where(observed & early_exit, exit_month, np.nan)
    df["Exit_Type"] = np.where(observed & early_exit, np.where(invol, "Involuntary", "Voluntary"), None)
    df["Ramp_Months"] = np.where(observed, ramp, np.nan)
    df["Quality_of_Hire_Index"] = np.where(
        observed, np.round(100 * (0.5 * (perf12 - 1) / 4 + 0.3 * (~early_exit) + 0.2 * (1 - (ramp - 1) / 11)), 1), np.nan)
    df["Current_Status"] = np.where(final_joined, np.where(exited_by_now, "Exited", "Active"), None)
    for c in ("Screen_Date", "Assessment_Date", "Interview_Date", "Offer_Date", "Decision_Date", "Join_Date", "Application_Date"):
        df[c] = pd.to_datetime(df[c]).dt.date
    return df


if __name__ == "__main__":
    data = generate_recruitment_dataset()
    data.to_csv("recruitment_data.csv", index=False)
    print(f"Recruitment dataset: {len(data)} candidates, {data.shape[1]} columns -> recruitment_data.csv")
    print(data["Funnel_Status"].value_counts().to_string())
