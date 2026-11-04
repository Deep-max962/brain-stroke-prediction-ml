# Brain Stroke Risk Analysis: Livelihood & Health KPIs

> **Completed group project.** Intro to Data Science, M.Sc. programme, SRH University (May 2026).
>
> **Team:** Manohar Anand Zalki · Deepa Narayanaswamy · Nicolas Bonilla

An interpretable machine-learning analysis of how **health indicators** (age, hypertension, heart disease, glucose, BMI, smoking) and **livelihood factors** (work type, residence, marital status) relate to stroke, following the **CRISP-DM** process from business understanding to a simple deployment prototype.

## Dataset

The public **Stroke Prediction Dataset** (`healthcare-dataset-stroke-data.csv`, available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)).

- **5,110 patients**, 12 columns, binary target `stroke`
- **4.87 % stroke cases** (249 of 5,110), so the classes are strongly imbalanced
- 201 missing BMI values; no duplicate rows

The CSV is not included in this repository. Download it and place it next to the notebook.

## Workflow

| CRISP-DM phase | What the notebook does |
|---|---|
| Data understanding | Shape, types, missing values, duplicates, class balance |
| Data preparation | Median imputation of BMI (28.1); 99th-percentile capping of glucose and BMI; one-hot encoding of categorical variables |
| Exploratory analysis | Stroke rate by age, work type, residence and marital status; KDE plots for age, glucose and BMI; hypertension and heart disease; smoking × age-group heatmap; correlation heatmap; summary dashboards |
| Modelling | Logistic regression with `class_weight="balanced"` to handle the imbalance; stratified 80/20 train/test split (`random_state=42`) |
| Evaluation | Classification report, ROC-AUC, confusion matrix |
| Insights | Logistic-regression coefficients and a risk-scoring function for example patient profiles |

## Results (test set, 1,022 patients)

| Metric | Value |
|---|---|
| ROC-AUC | **0.84** |
| Recall, stroke class | **0.80** (40 of 50 stroke cases detected) |
| Precision, stroke class | 0.14 |
| Accuracy | 0.75 |

Class weighting trades precision for recall: the model flags many false positives in order to miss few strokes, which suits a **screening** use case rather than diagnosis.

**Largest coefficients:** hypertension (+0.60), heart disease (+0.21) and current smoking (+0.23) increase predicted risk.

## Reproducibility note

The notebook is published **as submitted, with its original outputs**. It does **not currently run cleanly from top to bottom**:

- The execution counter jumps (e.g. from 17 to 30), showing that some cells were run and later deleted.
- The model's feature list includes `smoking_status_Seasonal`, a column that does not exist in the original dataset and was presumably created by one of those deleted cells. Re-running the notebook from a fresh kernel is therefore expected to fail at the modelling step.

All results reported above are taken from the saved outputs of the original run. No code has been added or changed; the only edits were removing warning messages that contained local file paths and one empty trailing cell.

## Limitations

- Features were **not scaled**, so coefficient sizes are not directly comparable across variables (age appears small for this reason), and scikit-learn reported numerical overflow warnings during fitting.
- Imputation and capping were computed on the full dataset before the train/test split.
- Observational data: the results show associations, not causes.
- Children are included, and "Unknown" smoking status is kept as its own category.

## Repository structure

```
notebooks/brain_stroke_risk_analysis.ipynb   full analysis with outputs
reports/Brain_Stroke_Final_24slides.pptx     final group presentation
```

## Tech

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · SciPy · Jupyter
