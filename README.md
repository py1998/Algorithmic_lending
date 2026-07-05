# Algorithmic Lending and Model Risk

It covers an analysis of AI-driven credit scoring. A machine-learning credit model is used as
*evidence* to examine the fairness, transparency, and data-privacy risks that
arise when algorithms decide who receives credit — and to map those risks onto
India's existing regulatory frameworks.

## Central finding

A credit model can be "accurate" and still be unfair. On a public lending
dataset, both a logistic-regression and a random-forest model approved men
above their true creditworthiness (79% approved vs 71% genuinely creditworthy)
while approving women at roughly their true rate. The disparity stayed with both the models, showing it is rooted in the data and task, not
in one model's algorithm — and it **could not be removed by deleting the gender field**,
because ordinary features (housing, dependents, employment) act as proxies for it.

## Repository contents

| File | What it is |
|------|-----------|
| `credit_risk_analysis.ipynb` | The full analysis notebook — data, both models, fairness testing, SHAP explainability, and proxy analysis. |
| `Algorithmic_Lending_Governance_Note.pdf` | A report detailing the findings of this study and linking its results with the current regulatory paradigms|

## Method in brief

- **Data:** Statlog German Credit dataset (1,000 borrowers, ~20 features; 700 "good" / 300 "bad").
- **Models:** logistic regression (simple, transparent) and random forest (more accurate, less interpretable).
- **Risk analysis:** group-wise approval-rate testing, SHAP feature attribution, and a proxy-leakage check.
- **Tools:** Python — pandas, scikit-learn, SHAP, matplotlib.

## Key results

- Logistic regression: 72.4% accuracy (baseline "always good" = 70.0%), ROC–AUC 0.775.
- Random forest: 76.8% accuracy, ROC–AUC 0.799 — but it catches the *same* number of bad loans as the simpler model; its gain comes from approving more good applicants.
- Both models miss over half of true defaulters, underscoring that the model is a decision *aid*, not a gatekeeper.

## Note on data

The German Credit dataset is a public teaching dataset with illustrative group
labels. The figures evidence the *type* of risk that arises in algorithmic
lending — they are not a measurement of any real lender. All results come
directly from the notebook.

---
*Author: Porush Yadav · June, 2026*
