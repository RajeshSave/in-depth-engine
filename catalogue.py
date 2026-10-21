"""
catalogue.py -- Development Rx: intervention catalogue + prescription logic.
21 built-in interventions (7 competencies x 3 score bands). High-band rows are
STRETCH opportunities, not remedial fixes. The in-app copy is editable.
"""
import numpy as np
import pandas as pd

from ml_engine import COMPETENCY_COLS

_ROWS = [
 ("Analytical_Test_Score","Low","Quantitative Reasoning Bootcamp","Formal Training","Intensive refresher on financial modelling, statistics and valuation math for candidates below benchmark on analytical assessments."),
 ("Analytical_Test_Score","Medium","Case-Based Analytics Workshop","Workshop","Applied case studies to sharpen structured problem-solving and quantitative decision-making."),
 ("Analytical_Test_Score","High","Advanced Quant Research Rotation","Stretch Assignment","Time-boxed rotation into the Quant Research desk on a live modelling project to pressure-test elite analytical skill."),
 ("Risk_Compliance_Score","Low","Mandatory Compliance Recertification","Certification","Formal recertification on regulatory obligations (KYC/AML, conduct risk) with a supervised assessment."),
 ("Risk_Compliance_Score","Medium","Risk Scenario Simulation Lab","Simulation","Scenario-based simulations covering market, credit and operational risk decision-making under pressure."),
 ("Risk_Compliance_Score","High","Risk Committee Observer Seat","Stretch Assignment","Non-voting observer seat on the Risk Committee for two quarters to build governance-level exposure."),
 ("Leadership_Potential_Score","Low","Leadership Foundations Programme","Programme","A structured multi-month programme covering delegation, feedback and team leadership fundamentals."),
 ("Leadership_Potential_Score","Medium","Executive Mentorship Pairing","Mentorship","Paired with a senior leader for a 6-12 month structured mentorship focused on stretch leadership exposure."),
 ("Leadership_Potential_Score","High","P&L / Team Ownership Assignment","Stretch Assignment","Direct ownership of a small P&L or team as a proving ground for the next level of leadership scope."),
 ("Communication_Score","Low","Executive Presence & Communication Coaching","Coaching","1:1 coaching on executive communication, storytelling with data and stakeholder influence."),
 ("Communication_Score","Medium","Client-Facing Presentation Skills Workshop","Workshop","Practice-based workshop on pitch delivery and handling challenging stakeholder questions."),
 ("Communication_Score","High","Board / Investor Presentation Opportunity","Stretch Assignment","Present directly to the Board or a major investor audience on a live initiative."),
 ("Technical_Skill_Score","Low","Technical Upskilling Track (Role-Specific)","Formal Training","A tailored technical curriculum (e.g. derivatives pricing, credit analysis, trading systems) for the employee's function."),
 ("Technical_Skill_Score","Medium","Cross-Desk Technical Rotation","Rotation","A short rotation onto an adjacent desk/team to broaden technical depth and product exposure."),
 ("Technical_Skill_Score","High","Subject-Matter-Expert (SME) Designation","Stretch Assignment","Formal SME designation with a mandate to mentor others and represent the function in cross-firm technical forums."),
 ("Client_Management_Score","Low","Client Relationship Management Certification","Certification","Structured certification covering relationship management, cross-selling and client lifecycle management."),
 ("Client_Management_Score","Medium","Shadow Senior Relationship Manager Programme","Shadowing","Shadowing a senior RM/banker on live client engagements over one quarter."),
 ("Client_Management_Score","High","Marquee Client Relationship Ownership","Stretch Assignment","Direct ownership of a top-tier client relationship, including renewal/expansion responsibility."),
 ("Engagement_Score","Low","Stay Interview & Career Conversation","1:1 Intervention","A structured stay-interview with HRBP/manager to identify and address root causes of disengagement."),
 ("Engagement_Score","Medium","Team Engagement Action Planning","Team Intervention","Manager-led action planning session using engagement survey themes to co-create improvement commitments."),
 ("Engagement_Score","High","Engagement Champion / Culture Ambassador Role","Stretch Assignment","Formal recognition as a team engagement champion, involved in shaping culture initiatives firm-wide."),
]
DEFAULT_CATALOGUE = pd.DataFrame(
    [(f"R{i+1:02d}", *r) for i, r in enumerate(_ROWS)],
    columns=["Row_ID", "Competency", "Band", "Intervention", "Format", "Description"],
)


def score_band(score, low=55, high=75):
    return "Low" if score < low else ("Medium" if score < high else "High")


def prescribe(candidate: pd.Series, catalogue: pd.DataFrame, low=55, high=75) -> pd.DataFrame:
    """One row per competency: score, band, and the catalogue's intervention for that
    competency+band (first match if the user added duplicates)."""
    rows = [{"Competency": c, "Score": round(float(candidate[c]), 1),
             "Band": score_band(float(candidate[c]), low, high)}
            for c in COMPETENCY_COLS if pd.notna(candidate.get(c))]
    plan = pd.DataFrame(rows)
    cat = catalogue.dropna(subset=["Competency", "Band"]).drop_duplicates(["Competency", "Band"], keep="first")
    plan = plan.merge(cat, on=["Competency", "Band"], how="left")
    plan["_ord"] = plan["Band"].map({"Low": 0, "Medium": 1, "High": 2})   # needs first, stretch last
    return plan.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
