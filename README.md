# Athlete Motion Phase Recognition

**CS156: Intro to Artificial Intelligence | San José State University, Spring 2026**

End-to-end supervised ML pipeline that classifies an athlete's motion phase from wearable biosensor data. Covers the full lifecycle: raw data exploration, preprocessing, feature engineering, model selection and tuning, deep learning extension, and final evaluation.

**Tech stack:** Python · scikit-learn · pandas · NumPy · SciPy · Matplotlib · Seaborn

---

## Problem

Given timestamped IMU and heart rate readings from a wearable sensor worn by a college track & field athlete, predict which of six motion phases is occurring:

`acceleration` · `jump takeoff` · `landing` · `sprint mid` · `start run` · `stop`

---

## Pipeline

| Stage | What was done |
|-------|--------------|
| **EDA** | Class balance checks, missing value audits, sampling rate verification, signal visualization per athlete and per motion phase |
| **Preprocessing** | Deduplication, timestamp cleanup, per-athlete z-score normalization across all 7 sensor channels, sliding-window segmentation |
| **Feature engineering** | 57 features per window: time-domain (mean, std, RMS, ZCR) and frequency-domain (dominant frequency, spectral energy, spectral entropy) across 7 channels, plus 8 cross-axis Pearson correlations |
| **Model training** | 7 classifiers (Random Forest, SVM-RBF, Decision Tree, Naive Bayes, AdaBoost, XGBoost, MLP) with `GridSearchCV` hyperparameter tuning |
| **Evaluation** | Stratified 5-fold CV and Leave-One-Athlete-Out (LOSO) CV; full error and confusion matrix analysis |
| **Deep learning** | Feedforward ANN (128 → 64 ReLU, batch norm, dropout, softmax) with early stopping trained on the engineered feature matrix |
| **System integration** | End-to-end inference function: raw CSV → windowing → feature extraction → prediction |

---

## Results

| Evaluation strategy | Best accuracy |
|--------------------|--------------|
| Stratified 5-fold CV | **30.4%** |
| Leave-One-Athlete-Out (LOSO) | **18.1%** |

Chance baseline with 6 classes is approximately 16.7%. The gap between 5-fold and LOSO reflects cross-athlete generalization difficulty, a known challenge in sports analytics with small, heterogeneous cohorts. LOSO is the more realistic estimate for deployment on unseen athletes.

**Model comparison (pre-tuning, 5-fold / LOSO):**

| Model | 5-Fold | LOSO |
|-------|--------|------|
| Random Forest | 28.7% | 16.8% |
| MLP | 27.6% | 19.0% |
| SVM-RBF | 25.9% | 14.5% |
| XGBoost | 24.8% | 23.2% |
| Naive Bayes | 23.1% | 14.0% |
| AdaBoost | 20.7% | 17.0% |
| Decision Tree | 18.7% | 17.5% |

---

## Why the Results Look This Way

The dataset is synthetic, so there is no real discriminative signal structure to learn from (real biosensor pipelines typically reach 70-90% accuracy). Labels also flip nearly every sample: the median consecutive label block is 1 row (0.1 s) and the maximum is only 6 rows (0.6 s), so every window captures multiple mixed phases and gets a noisy majority label. The 5-fold vs LOSO gap exists because 5-fold lets the same athlete appear in both train and test folds; LOSO is the honest number.

---

## Notable Design Choices

- **Leakage prevention:** train/test split strictly by `Athlete_ID`; `StandardScaler` fit on training athletes only and applied to held-out athletes, so no test-set statistics leak into the pipeline at any stage
- **Window size selection:** empirically chosen via a sweep across 10 sizes (0.2–5.0 s); larger windows hurt due to the dataset's short median event block length (1 sample = 0.1 s at 10 Hz)
- **Dual CV strategy:** 5-fold for stable model comparison, LOSO for realistic generalization estimates and error analysis
- **Error analysis:** conducted exclusively on LOSO predictions to surface weaknesses on unseen athletes; most confusions occur between biomechanically adjacent phases (e.g., `start_run` ↔ `stop`)

---

## Repo Layout

```
motionphaserecognition.ipynb   # full pipeline notebook
motionphase-report.pdf         # written report
data/
  biosensor_dataset_with_target.csv
report/
  main.tex                     # LaTeX source
  csvFiles/                    # result tables
  images/                      # figures from notebook
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
# then open motionphaserecognition.ipynb and run all cells
```
