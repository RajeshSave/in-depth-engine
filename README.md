# Talent Portal (Streamlit): Succession & Talent + Recruitment

## Run it (first time)
1. Install Python 3.10+ from python.org (Windows: tick **"Add Python to PATH"**).
2. Put ALL files from this folder together in one folder (no sub-folders).
3. Open a terminal in that folder (Windows: click the folder's address bar, type `cmd`, Enter).
4. `pip install -r requirements.txt`
5. `streamlit run app.py`   (if that fails: `python -m streamlit run app.py`)

Your browser opens at http://localhost:8501. Click **Load Sample Demo Dataset** on the first page.
Keep the terminal open while using the app; Ctrl+C stops it.

## Two workspaces (switch at the top of the sidebar)
- **Succession & Talent**: employee data -> rules, ML predictions, readiness, development prescriptions.
- **Recruitment**: ATS funnel + Hogan + Predictive Index -> predicted first-year performance, 12-month retention and offer
  acceptance, combined into a hire recommendation. Start on *Recruitment Data* and click **Load sample recruitment dataset**.
  Pages: Hiring Funnel & Sources, Hogan & PI Insights, Candidate Scoring, Model Quality & Fairness.
- Each workspace has its own dataset; loading one does not affect the other.

## What's in each file (so you know where to make changes)
| File | What it does | Change it when you want to... |
|---|---|---|
| `app.py` | Sidebar menu + page routing | add/rename/reorder pages |
| `styling.py` | Colours, fonts, card styles | change the look |
| `pages_core.py` | Upload, Overview, Layer 1, Layer 2 pages | change those screens |
| `pages_advanced.py` | Career, Succession, Development Rx, AI Report, Reports, Glossary | change those screens |
| `rules_engine.py` | Layer 1 rules | change rule logic |
| `ml_engine.py` | 3 ML models, clustering, readiness score, priority actions | change models / scoring |
| `catalogue.py` | The 21 built-in development interventions | change defaults |
| `genai.py` | AI report (Gemini / Claude / ChatGPT + offline fallback) | change prompts / model names |
| `state.py` | Shared helpers, default settings | change defaults (e.g. gate = 65) |
| `generate_data.py` | Creates the synthetic `employee_data.csv` | change the sample employee data |
| `generate_recruitment_data.py` | Creates the synthetic `recruitment_data.csv` (12,000 applicants, Hogan, PI, outcomes) | change the sample recruitment data |
| `recruit_engine.py` | Recruitment models, recommendations, explanations, backtest, fairness | change hiring logic |
| `pages_recruit.py` | The five recruitment pages | change those screens |
| `test_app.py` | 49 automated checks: `python test_app.py` (takes a few minutes) | run after any change |

## Notes
- Upload your own CSV on the first page; it needs these columns: Employee_ID, Full_Name, Business_Unit,
  Job_Level, Years_of_Experience, Years_at_Company, Avg_Performance_Rating, Number_of_Promotions,
  Time_Since_Last_Promotion_Years, Compa_Ratio, Feedback_360_Score and the 7 competency/engagement scores.
  Attrition_Flag and High_Potential_Program_Flag are optional (models that need them are skipped).
- Development Rx edits and settings live for the session only; use the CSV download to keep catalogue edits.
- AI reports work with no key (offline template). Model names are listed at the top of `genai.py`.

## Recruitment notes
- **Hogan and PI are proprietary.** The sample simulates scores on their documented scale names; it does not reproduce their scoring,
  and the assessment-to-outcome relationships are assumptions. With real data, export from the vendor platforms and rename columns to match
  the data dictionary on the *Recruitment Data* page.
- Models use only information available at hiring time. **Gender, age and age band are never model inputs**; they are used only by the
  fairness monitor. The sample data has an age-related disparity deliberately injected in interview scoring so the monitor has something to find.
- Models learn from people who were hired (selection bias) and use only hires with a full 12 months of history. Treat scores as input to a
  structured human review, validate on your own outcomes (the *evidence check* on Candidate Scoring does this), and check adverse impact.
- Random Forest takes about 30 seconds to train; the regularised linear option takes a second or two.
