"""
rules_engine.py

LAYER 1 -- RULE-BASED BUSINESS LOGIC ENGINE
Deterministic HR policy rules, evaluated top-to-bottom (first match wins).
Thresholds are parameterised so the UI can let an HRBP tune them live.

Ported from the R/Shiny version with two bug fixes carried forward
(see HANDOFF_BRIEF.md Section 4):
  - "Ready for Leadership Track" uses a 2-of-3 majority rule, not a strict
    3-way AND (which was a near-empty, slider-unresponsive intersection).
  - "Compliance Watchlist" uses its own adjustable threshold, not a
    hardcoded value.
"""

import numpy as np
import pandas as pd

FLAG_ORDER = [
    "Ready for Leadership Track", "Fast-Track Talent",
    "Promotion Overdue - Review Needed", "Steady / Core Contributor",
    "Compliance Watchlist", "Performance Improvement Plan Candidate",
    "High Attrition Risk",
]

FLAG_COLORS = {
    "Ready for Leadership Track": "#1B7A3E",
    "Fast-Track Talent": "#2563EB",
    "Promotion Overdue - Review Needed": "#D97706",
    "Steady / Core Contributor": "#94A3B8",
    "Compliance Watchlist": "#EA580C",
    "Performance Improvement Plan Candidate": "#B91C1C",
    "High Attrition Risk": "#991B1B",
}


def apply_layer1_rules(
    df: pd.DataFrame,
    analytical_thresh: float = 85,
    risk_thresh: float = 90,
    leadership_thresh: float = 80,
    engagement_low_thresh: float = 50,
    compa_low_thresh: float = 0.95,
    perf_high_thresh: float = 4.5,
    perf_low_thresh: float = 3.0,
    compliance_watch_thresh: float = 60,
    promo_overdue_years: float = 5,
) -> pd.DataFrame:
    df = df.copy()

    n_criteria_met = (
        (df["Analytical_Test_Score"] > analytical_thresh).astype(int)
        + (df["Risk_Compliance_Score"] > risk_thresh).astype(int)
        + (df["Leadership_Potential_Score"] > leadership_thresh).astype(int)
    )

    conditions = [
        n_criteria_met >= 2,
        (df["Avg_Performance_Rating"] >= perf_high_thresh) & (df["Number_of_Promotions"] >= 2),
        (df["Time_Since_Last_Promotion_Years"] >= promo_overdue_years) & (df["Avg_Performance_Rating"] >= 4),
        (df["Engagement_Score"] < engagement_low_thresh) & (df["Compa_Ratio"] < compa_low_thresh),
        df["Risk_Compliance_Score"] < compliance_watch_thresh,
        df["Avg_Performance_Rating"] < perf_low_thresh,
    ]
    choices = [
        "Ready for Leadership Track", "Fast-Track Talent",
        "Promotion Overdue - Review Needed", "High Attrition Risk",
        "Compliance Watchlist", "Performance Improvement Plan Candidate",
    ]
    df["Layer1_Flag"] = np.select(conditions, choices, default="Steady / Core Contributor")
    df["Layer1_Flag"] = pd.Categorical(df["Layer1_Flag"], categories=FLAG_ORDER, ordered=False)
    return df


if __name__ == "__main__":
    # Quick smoke test with the generated dataset
    df = pd.read_csv("employee_data.csv")
    out = apply_layer1_rules(df)
    print(out["Layer1_Flag"].value_counts())
