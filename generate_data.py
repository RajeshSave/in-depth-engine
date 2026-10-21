"""
generate_data.py

SYNTHETIC DATA GENERATOR
"Candidate Development Assessment Report & Succession Planning Portal"

Generates a clean, internally-consistent, realistic synthetic dataset of
520 Financial Services employees spanning Investment Banking, Risk,
Wealth Management and Quantitative Trading.

This script is intentionally kept SEPARATE from the Streamlit application
(app.py). It writes a single artifact -- "employee_data.csv" -- which the
app reads on startup (and regenerates automatically if missing).

DESIGN PRINCIPLES (to avoid "noisy"/illogical records)
  1. Age is always generated FROM Years_of_Experience (never independently),
     so Age >= Years_of_Experience + a realistic career-start age.
  2. Years_at_Company <= Years_of_Experience (tenure can never exceed a
     person's total career length).
  3. Years_in_Current_Role <= Years_at_Company.
  4. Time_Since_Last_Promotion_Years <= Years_at_Company.
  5. Job_Level is sampled conditionally on Years_of_Experience bands, so a
     2-year Analyst can never appear as an MD.
  6. Five years of historical Annual Performance Ratings are generated per
     employee (Performance_Rating_Y1..Y5 = most recent 5 review cycles),
     but are only populated for years the employee was actually employed
     (Years_at_Company >= n); everything else is NaN -- not zero, not noise.
  7. Compensation (Current_CTC_USD) is built from Job_Level + Business_Unit
     + a location cost-of-living multiplier + a performance-linked premium,
     so pay is internally coherent with role and performance.
"""

import warnings

import numpy as np
import pandas as pd
from datetime import date, timedelta

# Expected/benign: nanmean() on employees with 0 years of rating history
# (all-NaN row) warns "Mean of empty slice" -- handled immediately after via
# np.where(isnan, 3, ...), so this is not masking a real bug.
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ============================================================================
# 0. REPRODUCIBILITY
# ============================================================================
RNG = np.random.default_rng(20240601)
N_EMPLOYEES = 520

# ============================================================================
# 1. REFERENCE / LOOKUP TABLES
# ============================================================================
BUSINESS_UNITS = ["Investment Banking", "Risk Management",
                   "Wealth Management", "Quantitative Trading"]
BU_PROBS = [0.32, 0.24, 0.26, 0.18]

SUB_FUNCTIONS = {
    "Investment Banking":   ["M&A Advisory", "Equity Capital Markets",
                              "Debt Capital Markets", "Leveraged Finance"],
    "Risk Management":      ["Credit Risk", "Market Risk",
                              "Operational Risk", "Regulatory & Compliance Risk"],
    "Wealth Management":    ["Private Banking", "Portfolio Management",
                              "Financial Planning", "Client Advisory"],
    "Quantitative Trading": ["Algorithmic Trading", "Quant Research",
                              "Derivatives Structuring", "Market Making"],
}

JOB_LEVELS_ORDERED = ["Analyst", "Associate", "VP", "Director", "Managing Director"]

LOCATIONS = ["New York", "London", "Singapore", "Hong Kong", "Mumbai", "Dubai"]
LOCATION_PROBS = [0.24, 0.20, 0.16, 0.14, 0.16, 0.10]
LOCATION_MULTIPLIER = {"New York": 1.15, "London": 1.10, "Singapore": 1.05,
                        "Hong Kong": 1.08, "Mumbai": 0.55, "Dubai": 0.95}

EDUCATION_LEVELS = ["Bachelors", "Masters", "MBA"]
UNIVERSITY_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
CERTIFICATION_POOL = ["CFA", "FRM", "CPA", "CAIA", "PMP"]

FIRST_NAMES_M = ["James", "Michael", "Arjun", "David", "Robert", "Wei", "Daniel", "Omar",
                  "Kabir", "Thomas", "Ethan", "Rohan", "Ahmed", "William", "Ravi", "Lucas",
                  "Henry", "Karan", "Nikhil", "Sameer"]
FIRST_NAMES_F = ["Sarah", "Priya", "Emily", "Fatima", "Laura", "Mei", "Olivia", "Ananya",
                  "Grace", "Noor", "Sophia", "Ishita", "Chloe", "Aisha", "Hannah", "Riya",
                  "Charlotte", "Zara", "Meera", "Natasha"]
LAST_NAMES = ["Sharma", "Patel", "Smith", "Johnson", "Lee", "Chen", "Khan", "Williams",
               "Brown", "Gupta", "Kapoor", "Wilson", "Al-Farsi", "Nair", "Taylor", "Wong",
               "Mehta", "Anderson", "Rahman", "Singh"]

COMPETENCY_COLS = ["Analytical_Test_Score", "Risk_Compliance_Score",
                    "Leadership_Potential_Score", "Communication_Score",
                    "Technical_Skill_Score", "Client_Management_Score"]
# (Engagement_Score is also a "competency-space" column used by the app's
#  clustering step, but is generated separately below since it isn't part
#  of the level-scaled assessment-centre battery.)


def level_band(yoe: np.ndarray) -> np.ndarray:
    """Vectorised conditional Job_Level band from years of experience."""
    bands = np.select(
        [yoe < 3, (yoe >= 3) & (yoe < 6), (yoe >= 6) & (yoe < 10),
         (yoe >= 10) & (yoe < 16), yoe >= 16],
        ["Analyst", "Associate", "VP", "Director", "Managing Director"],
        default="Analyst",
    )
    return bands


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 100)


def generate_employee_dataset(n: int = N_EMPLOYEES) -> pd.DataFrame:
    rng = RNG

    # -- Years of experience: right-skewed (more junior than senior) -------
    years_of_experience = np.round(
        np.clip(rng.gamma(shape=2.1, scale=4.3, size=n), 0.5, 30), 1
    )

    gender = rng.choice(["Male", "Female"], size=n, p=[0.58, 0.42])

    full_names = []
    last_name_draw = rng.choice(LAST_NAMES, size=n)
    for g, ln in zip(gender, last_name_draw):
        fn = rng.choice(FIRST_NAMES_M) if g == "Male" else rng.choice(FIRST_NAMES_F)
        full_names.append(f"{fn} {ln}")
    full_names = np.array(full_names)

    education_level = rng.choice(EDUCATION_LEVELS, size=n, p=[0.35, 0.35, 0.30])
    start_age_map = {"Bachelors": 22, "Masters": 24, "MBA": 26}
    start_age = np.vectorize(start_age_map.get)(education_level) + np.round(rng.normal(0, 0.6, n))

    # Age is DERIVED from experience -> guarantees Age >= Experience + start age
    age = np.round(start_age + years_of_experience + rng.normal(0.3, 0.4, n))
    age = np.clip(age, 23, 65)

    # -- Business unit & sub-function ---------------------------------------
    business_unit = rng.choice(BUSINESS_UNITS, size=n, p=BU_PROBS)
    sub_function = np.array([rng.choice(SUB_FUNCTIONS[bu]) for bu in business_unit])

    # -- Job level: conditional on experience band, with slight randomness --
    band = level_band(years_of_experience)
    job_level = []
    for b in band:
        idx = JOB_LEVELS_ORDERED.index(b)
        if rng.random() < 0.75 or idx == 0:
            job_level.append(JOB_LEVELS_ORDERED[idx])
        else:
            job_level.append(JOB_LEVELS_ORDERED[idx - 1])
    job_level = np.array(job_level)

    # -- Tenure structure (guaranteed internally consistent) -----------------
    years_at_company = np.minimum(
        years_of_experience,
        np.round(np.maximum(0.3, years_of_experience * rng.uniform(0.35, 1.0, n)), 1)
    )
    years_in_current_role = np.minimum(
        years_at_company,
        np.round(np.maximum(0.2, years_at_company * rng.uniform(0.25, 1.0, n)), 1)
    )
    time_since_last_promotion = np.minimum(
        years_at_company,
        np.round(np.maximum(0, years_at_company * rng.uniform(0, 0.9, n)), 1)
    )
    number_of_promotions = np.maximum(
        0, np.round(years_at_company / (2.2 + rng.uniform(-0.4, 1.2, n)))
    )

    # -- Location / University tier ------------------------------------------
    location = rng.choice(LOCATIONS, size=n, p=LOCATION_PROBS)
    university_tier = rng.choice(UNIVERSITY_TIERS, size=n, p=[0.30, 0.45, 0.25])

    # -- Certifications --------------------------------------------------------
    cert_base_rate = np.where(np.isin(business_unit, ["Risk Management", "Quantitative Trading"]),
                               0.55, 0.35)
    certifications_count = rng.binomial(len(CERTIFICATION_POOL), cert_base_rate)
    certifications_held = []
    for k in certifications_count:
        if k == 0:
            certifications_held.append("None")
        else:
            certifications_held.append(", ".join(rng.choice(CERTIFICATION_POOL, size=k, replace=False)))
    certifications_held = np.array(certifications_held)

    # -- Core competency / assessment scores (correlated with level) --------
    level_bonus_map = {"Analyst": 0, "Associate": 3, "VP": 6, "Director": 9, "Managing Director": 12}
    level_bonus = np.vectorize(level_bonus_map.get)(job_level)

    analytical_test_score = clamp01(np.round(rng.normal(68 + level_bonus, 11)))
    risk_compliance_score = clamp01(np.round(rng.normal(72 + level_bonus * 0.6, 10)))
    leadership_potential = clamp01(np.round(rng.normal(60 + level_bonus * 1.3, 13)))
    communication_score = clamp01(np.round(rng.normal(70 + level_bonus * 0.8, 10)))
    technical_skill_score = clamp01(np.round(rng.normal(71 + level_bonus * 0.5, 11)))
    client_management_score = clamp01(np.round(rng.normal(66 + level_bonus * 0.9, 12)))

    # -- 5-year historical performance ratings (1-5 scale) --------------------
    # Centre of 68 (not 55) so population mean lands around ~3.4/5 with a
    # believable underperformer tail (~15-20% below the 3.0 PIP threshold)
    # rather than ~65% of the population landing below it.
    # Performance now partly reflects underlying capability (average of four
    # execution-related competency scores) -- previously it was independent of
    # every competency, so no model could learn it and "predicted vs actual
    # rating" gaps were pure noise.
    _comp_avg = (technical_skill_score + client_management_score + analytical_test_score + communication_score) / 4
    base_perf = clamp01(np.round(66 + 0.9 * (_comp_avg - 72) + level_bonus + rng.normal(0, 6, n))) / 20
    perf_matrix = np.full((n, 5), np.nan)
    for i in range(n):
        tenure_i = years_at_company[i]
        drift = rng.normal(0.05, 0.05)
        years_covered = min(5, int(np.floor(tenure_i)))
        for y in range(1, years_covered + 1):  # y=1 most recent
            raw = base_perf[i] + drift * (years_covered - y) + rng.normal(0, 0.35)
            perf_matrix[i, y - 1] = round(min(5, max(1, raw)) * 2) / 2
    perf_cols = [f"Performance_Rating_Y{i}" for i in range(1, 6)]

    # -- Engagement, feedback, training, absenteeism ---------------------------
    engagement_score = clamp01(np.round(rng.normal(68, 14, n)))
    feedback_360_score = clamp01(np.round(rng.normal(70, 12, n)))
    training_hours_annual = np.maximum(0, np.round(rng.normal(42, 18, n)))
    absenteeism_days_annual = np.maximum(0, np.round(rng.normal(5, 3, n)))

    # HiPo nomination is driven by the employee's measurable profile (leadership,
    # analytical strength, communication, performance) plus noise -- NOT random.
    # (An earlier version assigned it randomly, so no model could learn it.)
    _perf_for_hipo = np.where(np.isnan(np.nanmean(perf_matrix, axis=1)), 3.5, np.nanmean(perf_matrix, axis=1))
    hipo_logit = (-2.0
                  + 0.06 * (leadership_potential - 66)
                  + 0.03 * (analytical_test_score - 72)
                  + 0.02 * (communication_score - 74)
                  + 0.9 * (_perf_for_hipo - 3.7)
                  + rng.normal(0, 0.5, n))
    high_potential_flag = np.where(rng.random(n) < 1 / (1 + np.exp(-hipo_logit)), "Y", "N")
    mentorship_flag = rng.choice(["Y", "N"], size=n, p=[0.30, 0.70])
    cross_functional_flag = rng.choice(["Y", "N"], size=n, p=[0.28, 0.72])
    international_assignment_flag = rng.choice(["Y", "N"], size=n, p=[0.20, 0.80])

    # -- Compensation (coherent with level + location + performance) ----------
    level_base_usd_map = {"Analyst": 95000, "Associate": 145000, "VP": 220000,
                           "Director": 330000, "Managing Director": 480000}
    level_base_usd = np.vectorize(level_base_usd_map.get)(job_level)
    loc_mult = np.vectorize(LOCATION_MULTIPLIER.get)(location)
    avg_perf_for_pay = np.nanmean(perf_matrix, axis=1)
    avg_perf_for_pay = np.where(np.isnan(avg_perf_for_pay), 3, avg_perf_for_pay)
    perf_premium = 1 + (avg_perf_for_pay - 3) * 0.06
    current_ctc_usd = np.round(level_base_usd * loc_mult * perf_premium * rng.uniform(0.93, 1.07, n), -2)
    bonus_pct = np.round(clamp01(rng.normal(22 + level_bonus * 1.5, 8, n)), 1)
    compa_ratio = np.round(rng.uniform(0.85, 1.20, n), 2)

    # -- Manager linkage (self-referential org chart, simplified) -------------
    employee_id = np.array([f"EMP-{i:05d}" for i in range(10001, 10001 + n)])
    level_rank = np.vectorize(JOB_LEVELS_ORDERED.index)(job_level)
    manager_id = []
    for i in range(n):
        if job_level[i] == "Managing Director":
            manager_id.append(None)
            continue
        senior_pool = np.where(level_rank > level_rank[i])[0]
        if len(senior_pool) == 0:
            manager_id.append(None)
        else:
            manager_id.append(employee_id[rng.choice(senior_pool)])
    manager_id = np.array(manager_id)

    hire_date = [date.today() - timedelta(days=int(round(y * 365.25))) for y in years_at_company]

    # -- Historical Attrition ground-truth label (for supervised ML target) ---
    perf_trend_numeric = np.zeros(n)
    for i in range(n):
        vals = perf_matrix[i, ~np.isnan(perf_matrix[i])]
        if len(vals) >= 2:
            perf_trend_numeric[i] = vals[0] - vals[-1]  # recent - oldest

    # Coefficients are calibrated (not arbitrary) to target a realistic
    # ~15-20% historical attrition rate at population-average covariate
    # values -- an earlier version of this formula used raw 0-100
    # Engagement_Score with a coefficient sized for a 0-1 scale, which drove
    # mean attrition probability down to ~1.6% (near-unusable as an ML
    # target: only ~5-8 positive examples out of 520). Engagement is
    # rescaled to 0-1 here specifically to fix that.
    logit_p = (1.85
               + (-6.0) * (engagement_score / 100)
               + (-3.0) * (compa_ratio - 1)
               + (0.25) * time_since_last_promotion
               + (-1.0) * perf_trend_numeric
               + (-0.5) * (training_hours_annual / 50)
               + rng.normal(0, 0.35, n))
    attrition_prob = 1 / (1 + np.exp(-logit_p))
    attrition_flag = np.where(rng.random(n) < attrition_prob, "Exited", "Active")

    # ---- ASSEMBLE FINAL DATAFRAME ----------------------------------------
    df = pd.DataFrame({
        "Employee_ID": employee_id,
        "Full_Name": full_names,
        "Gender": gender,
        "Age": age.astype(int),
        "Business_Unit": business_unit,
        "Sub_Function": sub_function,
        "Job_Level": pd.Categorical(job_level, categories=JOB_LEVELS_ORDERED, ordered=True),
        "Location": location,
        "Manager_ID": manager_id,
        "Hire_Date": hire_date,
        "Years_of_Experience": years_of_experience,
        "Years_at_Company": years_at_company,
        "Years_in_Current_Role": years_in_current_role,
        "Time_Since_Last_Promotion_Years": time_since_last_promotion,
        "Number_of_Promotions": number_of_promotions.astype(int),
        "Education_Level": education_level,
        "University_Tier": university_tier,
        "Certifications_Held": certifications_held,
        "Certifications_Count": certifications_count,
        **{perf_cols[i]: perf_matrix[:, i] for i in range(5)},
        "Analytical_Test_Score": analytical_test_score,
        "Risk_Compliance_Score": risk_compliance_score,
        "Leadership_Potential_Score": leadership_potential,
        "Communication_Score": communication_score,
        "Technical_Skill_Score": technical_skill_score,
        "Client_Management_Score": client_management_score,
        "Engagement_Score": engagement_score,
        "Feedback_360_Score": feedback_360_score,
        "Training_Hours_Annual": training_hours_annual,
        "Absenteeism_Days_Annual": absenteeism_days_annual,
        "High_Potential_Program_Flag": high_potential_flag,
        "Mentorship_Flag": mentorship_flag,
        "Cross_Functional_Experience_Flag": cross_functional_flag,
        "International_Assignment_Flag": international_assignment_flag,
        "Current_CTC_USD": current_ctc_usd,
        "Bonus_Pct": bonus_pct,
        "Compa_Ratio": compa_ratio,
        "Attrition_Flag": attrition_flag,
    })

    # -- Derived convenience columns (still RAW data, not "algorithm" output) -
    df["Avg_Performance_Rating"] = df[perf_cols].mean(axis=1, skipna=True).round(2)
    df["Career_Velocity_Score"] = (df["Number_of_Promotions"] /
                                    df["Years_at_Company"].clip(lower=0.5)).round(2)

    # Performance Trajectory: most-recent-year rating vs. trailing 3-year avg.
    # Positive = improving, negative = declining, ~0 = stable. NaN (new hires
    # with <2 years of history) default to 0 downstream in the app, not here
    # (kept as NaN in the raw CSV so it's visibly "no data" rather than
    # silently "stable").
    recent3 = df[["Performance_Rating_Y1", "Performance_Rating_Y2", "Performance_Rating_Y3"]]
    df["Performance_Trajectory"] = (df["Performance_Rating_Y1"] - recent3.mean(axis=1, skipna=True)).round(2)

    # Promotions in the last 5 years specifically (vs. career-long count),
    # derived as career velocity projected over a 5-year window, capped at
    # the employee's total career promotion count.
    df["Promotions_Last_5_Years"] = np.minimum(
        df["Number_of_Promotions"], (df["Career_Velocity_Score"] * 5).round()
    ).astype(int)
    df["Promotions_Per_Year_Last_5Y"] = (df["Promotions_Last_5_Years"] / 5).round(2)

    # -- Final sanity assertions (fail loudly if any logical rule is broken) --
    assert (df["Age"] >= df["Years_of_Experience"] + 18).all(), "Age/Experience inconsistency"
    assert (df["Years_at_Company"] <= df["Years_of_Experience"] + 0.01).all(), "Tenure > Experience"
    assert (df["Years_in_Current_Role"] <= df["Years_at_Company"] + 0.01).all(), "Role tenure > Company tenure"
    assert (df["Time_Since_Last_Promotion_Years"] <= df["Years_at_Company"] + 0.01).all(), "Promotion recency > tenure"
    assert (df["Promotions_Last_5_Years"] <= df["Number_of_Promotions"] + 0.01).all(), "Recent promos > career promos"

    return df


if __name__ == "__main__":
    employee_data = generate_employee_dataset(N_EMPLOYEES)
    employee_data.to_csv("employee_data.csv", index=False)
    print(f"Synthetic dataset generated: {employee_data.shape[0]} rows, "
          f"{employee_data.shape[1]} columns -> employee_data.csv")
