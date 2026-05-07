# Explainable Temporal Learning Analytics for Early Identification of At-Risk Students in Online Higher Education

## Table of Contents

<details open>
<summary>Click to expand/collapse navigation</summary>

- [1. Project Overview](#1-project-overview)
  - [1.1 Research Title](#11-research-title)
  - [1.2 Research Problem](#12-research-problem)
  - [1.3 Research Objective](#13-research-objective)
  - [1.4 Q1-Oriented Contribution Statement](#14-q1-oriented-contribution-statement)
- [2. Dataset](#2-dataset)
  - [2.1 Dataset Source](#21-dataset-source)
  - [2.2 Raw Data Files](#22-raw-data-files)
  - [2.3 Dataset Summary](#23-dataset-summary)
  - [2.4 Target Definition](#24-target-definition)
- [3. Research Design](#3-research-design)
  - [3.1 Prediction Task](#31-prediction-task)
  - [3.2 Observation Windows](#32-observation-windows)
  - [3.3 Feature Groups](#33-feature-groups)
  - [3.4 Evaluation Settings](#34-evaluation-settings)
- [4. Methodology](#4-methodology)
  - [4.1 Overall Pipeline](#41-overall-pipeline)
  - [4.2 Data Preprocessing](#42-data-preprocessing)
  - [4.3 Temporal Feature Engineering](#43-temporal-feature-engineering)
  - [4.4 Modeling Strategy](#44-modeling-strategy)
  - [4.5 Explainability Analysis](#45-explainability-analysis)
  - [4.6 Fairness and Subgroup Analysis](#46-fairness-and-subgroup-analysis)
  - [4.7 Calibration and Threshold Analysis](#47-calibration-and-threshold-analysis)
  - [4.8 Bootstrap Confidence Intervals](#48-bootstrap-confidence-intervals)
- [5. Repository Structure](#5-repository-structure)
- [6. Reproducibility Guide](#6-reproducibility-guide)
  - [6.1 Environment Setup](#61-environment-setup)
  - [6.2 Dataset Placement](#62-dataset-placement)
  - [6.3 Run All Experiments](#63-run-all-experiments)
- [7. Experimental Results](#7-experimental-results)
  - [7.1 Standard Early Prediction](#71-standard-early-prediction)
  - [7.2 Active-at-Window Early Prediction](#72-active-at-window-early-prediction)
  - [7.3 Feature-Group Ablation Study](#73-feature-group-ablation-study)
  - [7.4 SHAP Explainability](#74-shap-explainability)
  - [7.5 Fairness and Subgroup Analysis](#75-fairness-and-subgroup-analysis)
  - [7.6 Cross-Module Validation](#76-cross-module-validation)
  - [7.7 Calibration and Threshold Analysis](#77-calibration-and-threshold-analysis)
  - [7.8 Bootstrap Confidence Interval Analysis](#78-bootstrap-confidence-interval-analysis)
- [8. Generated Tables](#8-generated-tables)
- [9. Generated Figures](#9-generated-figures)
  - [9.1 SHAP Bar Plots](#91-shap-bar-plots)
  - [9.2 SHAP Beeswarm Plots](#92-shap-beeswarm-plots)
  - [9.3 Calibration Curves](#93-calibration-curves)
- [10. Q1 Manuscript Positioning](#10-q1-manuscript-positioning)
  - [10.1 Recommended Paper Title](#101-recommended-paper-title)
  - [10.2 Suggested Abstract](#102-suggested-abstract)
  - [10.3 Suggested Research Questions](#103-suggested-research-questions)
  - [10.4 Suggested Manuscript Structure](#104-suggested-manuscript-structure)
- [11. Key Findings](#11-key-findings)
- [12. Limitations](#12-limitations)
- [13. Future Work](#13-future-work)
- [14. Citation](#14-citation)
- [15. License and Ethical Use](#15-license-and-ethical-use)
- [16. Q1 Result Tables and Rendered Figures](#16-q1-result-tables-and-rendered-figures)
  - [16.1 Supplementary Result Tables](#161-supplementary-result-tables)
  - [16.2 SHAP Feature Importance Figures](#162-shap-feature-importance-figures)
    - [Figure 16.1. SHAP feature importance, Day 14](#figure-161-shap-feature-importance-day-14)
    - [Figure 16.2. SHAP feature importance, Day 28](#figure-162-shap-feature-importance-day-28)
    - [Figure 16.3. SHAP feature importance, Day 56](#figure-163-shap-feature-importance-day-56)
    - [Figure 16.4. SHAP feature importance, Day 84](#figure-164-shap-feature-importance-day-84)
    - [Figure 16.5. SHAP feature importance, Full period](#figure-165-shap-feature-importance-full-period)
  - [16.3 SHAP Beeswarm Figures](#163-shap-beeswarm-figures)
    - [Figure 16.6. SHAP beeswarm, Day 14](#figure-166-shap-beeswarm-day-14)
    - [Figure 16.7. SHAP beeswarm, Day 28](#figure-167-shap-beeswarm-day-28)
    - [Figure 16.8. SHAP beeswarm, Day 56](#figure-168-shap-beeswarm-day-56)
    - [Figure 16.9. SHAP beeswarm, Day 84](#figure-169-shap-beeswarm-day-84)
    - [Figure 16.10. SHAP beeswarm, Full period](#figure-1610-shap-beeswarm-full-period)
  - [16.4 Calibration Curve Figures](#164-calibration-curve-figures)
    - [Figure 16.11. Calibration curve, Day 14](#figure-1611-calibration-curve-day-14)
    - [Figure 16.12. Calibration curve, Day 28](#figure-1612-calibration-curve-day-28)
    - [Figure 16.13. Calibration curve, Day 56](#figure-1613-calibration-curve-day-56)
    - [Figure 16.14. Calibration curve, Day 84](#figure-1614-calibration-curve-day-84)
    - [Figure 16.15. Calibration curve, Full period](#figure-1615-calibration-curve-full-period)

</details>

---


## 1. Project Overview

### 1.1 Research Title

**Explainable Temporal Learning Analytics for Early Identification of At-Risk Students in Online Higher Education**

This repository implements a complete, Q1-journal-oriented experimental framework for early identification of at-risk students using the **Open University Learning Analytics Dataset (OULAD)**. The project integrates temporal feature engineering, early-warning prediction, active-at-window evaluation, feature-group ablation, SHAP-based explainability, subgroup/fairness analysis, cross-module validation, calibration analysis, threshold optimization, and bootstrap confidence intervals.

### 1.2 Research Problem

Online higher education platforms continuously record students' learning activities through Virtual Learning Environment (VLE) logs, assessment submissions, and course interactions. However, these behavioral traces are often underutilized for timely academic-risk detection. A practical early-warning system should identify students at risk of failure or withdrawal before the end of the course, while also providing interpretable, calibrated, and fairness-aware decision-support evidence.

### 1.3 Research Objective

The objective of this project is to develop and evaluate an explainable temporal learning analytics framework that can identify at-risk students across multiple observation windows using demographic, course-context, VLE engagement, and assessment-related indicators.

The primary prediction target is:

```text
At-risk      = Fail + Withdrawn
Non-risk     = Pass + Distinction
```

### 1.4 Q1-Oriented Contribution Statement

This study contributes:

1. A temporal early-warning framework for identifying at-risk students using multiple observation windows: Day 14, Day 28, Day 56, Day 84, and Full period.
2. A strict **active-at-window evaluation** that excludes students who had already withdrawn before each prediction window to avoid post-outcome contamination.
3. A feature-group ablation study quantifying the predictive roles of demographic, course-context, VLE engagement, and assessment-related features.
4. SHAP-based explainability showing a temporal shift from early engagement/contextual signals to later assessment-driven evidence.
5. A subgroup/fairness analysis evaluating consistency across gender, disability, age band, IMD band, highest education, and region.
6. Cross-module validation assessing generalization to unseen course modules.
7. Calibration and threshold analysis for intervention-oriented decision support.
8. Bootstrap confidence intervals to assess statistical stability of the reported performance.

---

## 2. Dataset

### 2.1 Dataset Source

This project uses the **Open University Learning Analytics Dataset (OULAD)**, a widely used benchmark dataset for learning analytics research. The dataset contains anonymized student demographics, course information, assessment submissions, registration records, VLE interaction logs, and VLE activity metadata.

### 2.2 Raw Data Files

The raw dataset is expected to be placed in:

```text
data/raw/anonymisedData/
```

The required files are:

```text
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv
vle.csv
```

### 2.3 Dataset Summary

| Table | Rows | Columns | Main Content |
|---|---:|---:|---|
| `assessments.csv` | 206 | 6 | Assessment metadata, type, date, and weight |
| `courses.csv` | 22 | 3 | Course module, presentation, and module length |
| `studentAssessment.csv` | 173,912 | 5 | Student assessment submissions and scores |
| `studentInfo.csv` | 32,593 | 12 | Student demographics and final outcomes |
| `studentRegistration.csv` | 32,593 | 5 | Registration and unregistration information |
| `studentVle.csv` | 10,655,280 | 6 | Daily VLE interactions and click counts |
| `vle.csv` | 6,364 | 6 | VLE resource metadata and activity type |

### 2.4 Target Definition

The original OULAD final result contains four categories:

| Original `final_result` | Binary Label |
|---|---|
| `Withdrawn` | At-risk = 1 |
| `Fail` | At-risk = 1 |
| `Pass` | Non-risk = 0 |
| `Distinction` | Non-risk = 0 |

The resulting class distribution is:

| Class | Count | Percentage |
|---|---:|---:|
| At-risk | 17,208 | 52.80% |
| Non-risk | 15,385 | 47.20% |

---

## 3. Research Design

### 3.1 Prediction Task

The task is binary classification:

```text
Input:
Student demographic, course-context, VLE engagement, and assessment features.

Output:
Probability that a student belongs to the at-risk group.
```

### 3.2 Observation Windows

The model is evaluated under multiple temporal observation windows:

| Window | Description |
|---|---|
| Day 14 | Early course stage |
| Day 28 | First-month prediction |
| Day 56 | Mid-early prediction |
| Day 84 | Mid-course prediction |
| Full period | Upper-bound complete-course benchmark |

### 3.3 Feature Groups

The framework constructs four main feature groups:

| Feature Group | Examples |
|---|---|
| Demographic | `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability` |
| Course context | `code_module`, `code_presentation`, `module_presentation_length` |
| VLE engagement | `total_clicks`, `active_days`, `last_vle_day`, `unique_sites`, `click_page`, `click_forumng` |
| Assessment | `mean_score`, `max_score`, `weighted_score_ratio`, `submission_delay`, `assessment_count` |

### 3.4 Evaluation Settings

This project evaluates the framework through the following settings:

| Setting | Purpose |
|---|---|
| Standard early prediction | Evaluate model performance using data up to each observation window |
| Active-at-window prediction | Exclude students who had already withdrawn before each observation window |
| Ablation study | Quantify contribution of feature groups |
| SHAP explainability | Explain feature contributions across time |
| Fairness/subgroup analysis | Evaluate subgroup performance consistency |
| Cross-module validation | Test generalization to unseen modules |
| Calibration and threshold analysis | Assess probability reliability and decision thresholds |
| Bootstrap confidence interval | Quantify statistical stability |

---

## 4. Methodology

### 4.1 Overall Pipeline

```mermaid
flowchart TD
    A[Raw OULAD CSV Files] --> B[Dataset Inspection]
    B --> C[Preprocessing and Label Construction]
    C --> D[Temporal Feature Engineering]
    D --> E[Observation Window Feature Tables]
    E --> F[Baseline and Early Prediction Models]
    F --> G[Feature-Group Ablation]
    F --> H[SHAP Explainability]
    F --> I[Fairness/Subgroup Analysis]
    F --> J[Active-at-Window Evaluation]
    F --> K[Cross-Module Validation]
    F --> L[Calibration and Threshold Analysis]
    F --> M[Bootstrap Confidence Intervals]
    G --> N[Q1-Ready Result Tables]
    H --> N
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
```

### 4.2 Data Preprocessing

The preprocessing stage:

1. Loads all OULAD CSV files.
2. Constructs the binary target variable `at_risk`.
3. Merges student information, registration, course metadata, VLE logs, and assessment records.
4. Removes direct leakage variables from predictors, especially `final_result`, `id_student`, and `date_unregistration`.
5. Saves processed feature tables for full-period and early-window experiments.

### 4.3 Temporal Feature Engineering

The framework constructs temporal learning indicators, including:

| Feature Type | Examples |
|---|---|
| Engagement intensity | `total_clicks`, `clicks_per_active_day` |
| Engagement continuity | `first_vle_day`, `last_vle_day`, `vle_span_days` |
| Activity diversity | `unique_sites`, `unique_activity_types` |
| Activity-type interactions | `click_page`, `click_forumng`, `click_oucontent`, `click_resource` |
| Assessment performance | `mean_score`, `max_score`, `min_score`, `weighted_score_ratio` |
| Submission behavior | `first_submission_day`, `last_submission_day`, `mean_submission_delay` |

### 4.4 Modeling Strategy

The following models are evaluated:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting
4. XGBoost

The main model used for interpretability and calibration analysis is:

```text
XGBoost + All features
```

because it consistently achieved the strongest or near-strongest performance across major experiments.

### 4.5 Explainability Analysis

SHAP-style feature contribution analysis is performed using XGBoost feature contributions. The analysis produces:

1. Raw-level feature importance tables.
2. Transformed feature importance tables.
3. SHAP bar plots.
4. SHAP beeswarm plots.

### 4.6 Fairness and Subgroup Analysis

Subgroup performance is evaluated across:

```text
gender
disability
age_band
imd_band
highest_education
region
```

Metrics include:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
PR-AUC
At-risk rate
Predicted at-risk rate
```

### 4.7 Calibration and Threshold Analysis

Calibration analysis evaluates whether predicted probabilities align with observed at-risk rates.

Metrics include:

```text
Brier score
Expected Calibration Error (ECE)
Calibration curve
Threshold-specific precision, recall, specificity, F1-score
```

### 4.8 Bootstrap Confidence Intervals

Non-parametric bootstrap resampling is performed using 1,000 bootstrap iterations on the test predictions.

The reported confidence intervals cover:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
PR-AUC
```

---

## 5. Repository Structure

```text
oulad-explainable-early-warning/
│
├── data/
│   ├── raw/
│   │   └── anonymisedData/
│   │       ├── assessments.csv
│   │       ├── courses.csv
│   │       ├── studentAssessment.csv
│   │       ├── studentInfo.csv
│   │       ├── studentRegistration.csv
│   │       ├── studentVle.csv
│   │       └── vle.csv
│   │
│   └── processed/
│       ├── oulad_features_day14.csv
│       ├── oulad_features_day28.csv
│       ├── oulad_features_day56.csv
│       ├── oulad_features_day84.csv
│       └── oulad_features_full.csv
│
├── outputs/
│   ├── tables/
│   │   ├── baseline_model_results.csv
│   │   ├── early_prediction_results.csv
│   │   ├── early_prediction_best_by_window.csv
│   │   ├── ablation_study_results.csv
│   │   ├── ablation_best_by_group.csv
│   │   ├── ablation_best_per_window.csv
│   │   ├── active_window_prediction_results.csv
│   │   ├── active_window_best_by_window.csv
│   │   ├── cross_module_validation_results.csv
│   │   ├── cross_module_summary_by_window.csv
│   │   ├── cross_module_worst_by_window.csv
│   │   ├── shap/
│   │   ├── fairness/
│   │   ├── calibration/
│   │   └── bootstrap/
│   │
│   ├── figures/
│   │   ├── shap/
│   │   └── calibration/
│   │
│   └── models/
│
├── src/
│   ├── 01_check_dataset.py
│   ├── 02_build_features.py
│   ├── 03_train_baseline_models.py
│   ├── 04_build_early_features.py
│   ├── 05_train_early_prediction.py
│   ├── 06_ablation_study.py
│   ├── 07_shap_explainability.py
│   ├── 08_fairness_subgroup_analysis.py
│   ├── 09_active_window_prediction.py
│   ├── 10_cross_module_validation.py
│   ├── 11_calibration_threshold_analysis.py
│   └── 12_bootstrap_confidence_interval.py
│
├── README.md
└── requirements.txt
```

---

## 6. Reproducibility Guide

### 6.1 Environment Setup

Activate the Python environment:

```bash
conda activate main
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib xgboost shap
```

Optional:

```bash
pip freeze > requirements.txt
```

### 6.2 Dataset Placement

Place the OULAD dataset in:

```text
data/raw/anonymisedData/
```

Verify:

```bash
ls data/raw/anonymisedData
```

Expected files:

```text
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv
vle.csv
```

### 6.3 Run All Experiments

Run the full pipeline:

```bash
python src/01_check_dataset.py
python src/02_build_features.py
python src/03_train_baseline_models.py
python src/04_build_early_features.py
python src/05_train_early_prediction.py
python src/06_ablation_study.py
python src/07_shap_explainability.py
python src/08_fairness_subgroup_analysis.py
python src/09_active_window_prediction.py
python src/10_cross_module_validation.py
python src/11_calibration_threshold_analysis.py
python src/12_bootstrap_confidence_interval.py
```

---

## 7. Experimental Results

### 7.1 Standard Early Prediction

**Table 1. Standard early prediction performance using XGBoost + All features.**

| Window | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Day 14 | 0.7239 | 0.7566 | 0.7032 | 0.7289 | 0.8012 | 0.8368 |
| Day 28 | 0.7719 | 0.8114 | 0.7400 | 0.7741 | 0.8547 | 0.8874 |
| Day 56 | 0.8206 | 0.8626 | 0.7854 | 0.8221 | 0.8994 | 0.9252 |
| Day 84 | 0.8337 | 0.8762 | 0.7978 | 0.8351 | 0.9160 | 0.9399 |
| Full | 0.9552 | 0.9820 | 0.9322 | 0.9565 | 0.9894 | 0.9921 |

Interpretation: performance increases as more temporal learning evidence becomes available. The full-period setting should be interpreted as an upper-bound benchmark rather than a deployable early-warning setting.

### 7.2 Active-at-Window Early Prediction

**Table 2. Active-at-window prediction, excluding students already withdrawn before each observation window.**

| Window | Active Rows | Removed Pre-Window Withdrawn | Best Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Day 14 | 28,119 | 4,474 | Logistic Regression | 0.6866 | 0.6479 | 0.6743 | 0.6609 | 0.7476 | 0.7203 |
| Day 28 | 27,538 | 5,055 | XGBoost | 0.7306 | 0.7273 | 0.6231 | 0.6712 | 0.7947 | 0.7778 |
| Day 56 | 26,522 | 6,071 | Logistic Regression | 0.7771 | 0.7328 | 0.7385 | 0.7356 | 0.8468 | 0.8251 |
| Day 84 | 25,724 | 6,869 | XGBoost | 0.7852 | 0.7812 | 0.6467 | 0.7076 | 0.8574 | 0.8350 |

Interpretation: active-at-window performance is lower than standard early-window performance but provides a stricter and more realistic estimate of deployable early-warning utility.

### 7.3 Feature-Group Ablation Study

**Table 3. Best feature group per observation window.**

| Window | Best Feature Group | Model | Number of Features | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Day 14 | All features | XGBoost | 57 | 0.7239 | 0.7566 | 0.7032 | 0.7289 | 0.8012 | 0.8368 |
| Day 28 | All features | XGBoost | 57 | 0.7719 | 0.8114 | 0.7400 | 0.7741 | 0.8547 | 0.8874 |
| Day 56 | All features | XGBoost | 58 | 0.8206 | 0.8626 | 0.7854 | 0.8221 | 0.8994 | 0.9252 |
| Day 84 | All features | XGBoost | 58 | 0.8337 | 0.8762 | 0.7978 | 0.8351 | 0.9160 | 0.9399 |
| Full | All features | XGBoost | 95 | 0.9552 | 0.9820 | 0.9322 | 0.9565 | 0.9894 | 0.9921 |

Key finding: integrating demographic, course-context, VLE engagement, and assessment-related features provides the strongest overall performance.

### 7.4 SHAP Explainability

**Table 4. Top SHAP-ranked raw features by observation window.**

| Window | Top Features |
|---|---|
| Day 14 | `code_module`, `last_vle_day`, `highest_education`, `imd_band`, `total_clicks`, `studied_credits`, `vle_span_days`, `module_presentation_length`, `click_page`, `region` |
| Day 28 | `code_module`, `last_vle_day`, `mean_score`, `highest_education`, `min_score`, `imd_band`, `mean_submission_delay`, `weighted_score_ratio`, `active_days`, `studied_credits` |
| Day 56 | `last_vle_day`, `code_module`, `mean_score`, `total_weighted_score`, `weighted_score_ratio`, `last_submission_day`, `highest_education`, `max_score`, `mean_submission_delay`, `imd_band` |
| Day 84 | `last_submission_day`, `last_vle_day`, `mean_score`, `max_score`, `code_module`, `mean_submission_delay`, `total_weighted_score`, `highest_education`, `module_presentation_length`, `click_page` |
| Full | `last_vle_day`, `last_submission_day`, `total_weighted_score`, `weighted_score_ratio`, `vle_span_days`, `mean_score`, `code_module`, `total_assessment_weight`, `assessment_span_days`, `assessment_count` |

Key interpretation: early prediction relies more on engagement continuity and course context, while later prediction increasingly depends on assessment performance and submission behavior.

### 7.5 Fairness and Subgroup Analysis

Subgroup analysis evaluates performance across:

```text
gender
disability
age_band
imd_band
highest_education
region
```

The largest early-window gaps were observed in:

| Window | Subgroup | Metric | Gap |
|---|---|---|---:|
| Day 14 | `highest_education` | Recall | 0.4136 |
| Day 14 | `highest_education` | F1-score | 0.4046 |
| Day 14 | `imd_band` | Recall | 0.3189 |
| Day 28 | `imd_band` | Recall | 0.2858 |

Key interpretation: early-warning predictions are useful but should be interpreted cautiously because subgroup performance gaps are more pronounced in the earliest observation windows.

### 7.6 Cross-Module Validation

**Table 5. Cross-module validation summary.**

| Window | Mean Accuracy | Mean Precision | Mean Recall | Mean F1-score | Mean ROC-AUC | Mean PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Day 14 | 0.6568 | 0.6745 | 0.6778 | 0.6432 | 0.7633 | 0.7579 |
| Day 28 | 0.6437 | 0.6783 | 0.7534 | 0.6656 | 0.8018 | 0.8015 |
| Day 56 | 0.7199 | 0.7367 | 0.7933 | 0.7313 | 0.8547 | 0.8502 |
| Day 84 | 0.7413 | 0.7577 | 0.8128 | 0.7546 | 0.8843 | 0.8903 |
| Full | 0.8176 | 0.8101 | 0.9434 | 0.8481 | 0.9724 | 0.9747 |

**Table 6. Worst holdout module per observation window.**

| Window | Worst Holdout Module | Model | Accuracy | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| Day 14 | GGG | XGBoost | 0.4676 | 0.9520 | 0.5901 | 0.6731 | 0.5753 |
| Day 28 | GGG | HistGradientBoosting | 0.4057 | 0.9961 | 0.5743 | 0.6710 | 0.5925 |
| Day 56 | GGG | XGBoost | 0.4025 | 1.0000 | 0.5740 | 0.7263 | 0.6511 |
| Day 84 | GGG | XGBoost | 0.4025 | 1.0000 | 0.5740 | 0.7908 | 0.7717 |
| Full | GGG | XGBoost | 0.4025 | 1.0000 | 0.5740 | 0.9229 | 0.9323 |

Key interpretation: cross-module validation is more challenging than random-split evaluation. Module GGG appears to represent a distribution-shift case in which the model tends to over-identify students as at risk.

### 7.7 Calibration and Threshold Analysis

**Table 7. Calibration and threshold summary.**

| Window | ROC-AUC | PR-AUC | Brier Score | ECE | Best-F1 Threshold | Best F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Day 14 | 0.8012 | 0.8368 | 0.1805 | 0.0129 | 0.35 | 0.7457 | 0.6579 | 0.8605 |
| Day 28 | 0.8547 | 0.8874 | 0.1528 | 0.0220 | 0.40 | 0.7783 | 0.7466 | 0.8129 |
| Day 56 | 0.8994 | 0.9252 | 0.1253 | 0.0116 | 0.45 | 0.8252 | 0.8376 | 0.8133 |
| Day 84 | 0.9160 | 0.9399 | 0.1117 | 0.0150 | 0.55 | 0.8391 | 0.9027 | 0.7838 |
| Full | 0.9894 | 0.9921 | 0.0347 | 0.0070 | 0.50 | 0.9565 | 0.9820 | 0.9322 |

Key interpretation: lower thresholds are preferable in early windows when recall is more important for intervention-oriented screening.

### 7.8 Bootstrap Confidence Interval Analysis

**Table 8. Bootstrap 95% confidence intervals at default threshold 0.50.**

| Window | F1-score, 95% CI | ROC-AUC, 95% CI | PR-AUC, 95% CI |
|---|---:|---:|---:|
| Day 14 | 0.7288 [0.7155, 0.7419] | 0.8012 [0.7891, 0.8138] | 0.8368 [0.8238, 0.8491] |
| Day 28 | 0.7741 [0.7613, 0.7871] | 0.8549 [0.8449, 0.8655] | 0.8876 [0.8784, 0.8969] |
| Day 56 | 0.8218 [0.8109, 0.8335] | 0.8993 [0.8908, 0.9076] | 0.9252 [0.9184, 0.9318] |
| Day 84 | 0.8349 [0.8237, 0.8450] | 0.9159 [0.9084, 0.9230] | 0.9398 [0.9340, 0.9452] |
| Full | 0.9564 [0.9509, 0.9621] | 0.9894 [0.9872, 0.9916] | 0.9921 [0.9905, 0.9937] |

**Table 9. Bootstrap 95% confidence intervals at best-F1 thresholds.**

| Window | Threshold | F1-score, 95% CI | Recall, 95% CI |
|---|---:|---:|---:|
| Day 14 | 0.35 | 0.7457 [0.7332, 0.7582] | 0.8608 [0.8473, 0.8741] |
| Day 28 | 0.40 | 0.7785 [0.7660, 0.7903] | 0.8130 [0.7978, 0.8271] |
| Day 56 | 0.45 | 0.8254 [0.8144, 0.8360] | 0.8135 [0.7989, 0.8271] |
| Day 84 | 0.55 | 0.8391 [0.8282, 0.8500] | 0.7839 [0.7677, 0.7992] |
| Full | 0.50 | 0.9565 [0.9509, 0.9624] | 0.9323 [0.9226, 0.9421] |

---

## 8. Generated Tables

| File | Description |
|---|---|
| `outputs/tables/dataset_summary.csv` | Raw dataset summary |
| `outputs/tables/target_distribution.csv` | Target class distribution |
| `outputs/tables/processed_feature_summary.csv` | Processed feature summary |
| `outputs/tables/baseline_model_results.csv` | Full-period baseline results |
| `outputs/tables/early_prediction_results.csv` | Standard early prediction results |
| `outputs/tables/early_prediction_best_by_window.csv` | Best model by observation window |
| `outputs/tables/ablation_study_results.csv` | Complete ablation results |
| `outputs/tables/ablation_best_by_group.csv` | Best model per feature group |
| `outputs/tables/ablation_best_per_window.csv` | Best feature group per window |
| `outputs/tables/active_window_prediction_results.csv` | Active-at-window prediction results |
| `outputs/tables/active_window_best_by_window.csv` | Best active-window model per window |
| `outputs/tables/cross_module_validation_results.csv` | Complete cross-module validation results |
| `outputs/tables/cross_module_summary_by_window.csv` | Mean cross-module performance by window |
| `outputs/tables/cross_module_worst_by_window.csv` | Worst holdout module per window |
| `outputs/tables/fairness/subgroup_performance_results.csv` | Subgroup performance results |
| `outputs/tables/fairness/subgroup_performance_gaps.csv` | Subgroup gap analysis |
| `outputs/tables/shap/shap_raw_importance_all_windows.csv` | SHAP raw feature importance |
| `outputs/tables/shap/shap_top10_raw_features_all_windows.csv` | Top 10 SHAP features per window |
| `outputs/tables/calibration/calibration_threshold_summary.csv` | Calibration and threshold summary |
| `outputs/tables/calibration/threshold_analysis_all_windows.csv` | Threshold-specific metrics |
| `outputs/tables/bootstrap/bootstrap_confidence_interval_summary.csv` | Bootstrap 95% confidence intervals |

---

## 9. Generated Figures

### 9.1 SHAP Bar Plots

| Figure | Path |
|---|---|
| Figure 1. SHAP bar plot for Day 14 | `outputs/figures/shap/shap_bar_day14.png` |
| Figure 2. SHAP bar plot for Day 28 | `outputs/figures/shap/shap_bar_day28.png` |
| Figure 3. SHAP bar plot for Day 56 | `outputs/figures/shap/shap_bar_day56.png` |
| Figure 4. SHAP bar plot for Day 84 | `outputs/figures/shap/shap_bar_day84.png` |
| Figure 5. SHAP bar plot for Full period | `outputs/figures/shap/shap_bar_full.png` |

### 9.2 SHAP Beeswarm Plots

| Figure | Path |
|---|---|
| Figure 6. SHAP beeswarm plot for Day 14 | `outputs/figures/shap/shap_beeswarm_day14.png` |
| Figure 7. SHAP beeswarm plot for Day 28 | `outputs/figures/shap/shap_beeswarm_day28.png` |
| Figure 8. SHAP beeswarm plot for Day 56 | `outputs/figures/shap/shap_beeswarm_day56.png` |
| Figure 9. SHAP beeswarm plot for Day 84 | `outputs/figures/shap/shap_beeswarm_day84.png` |
| Figure 10. SHAP beeswarm plot for Full period | `outputs/figures/shap/shap_beeswarm_full.png` |

### 9.3 Calibration Curves

| Figure | Path |
|---|---|
| Figure 11. Calibration curve for Day 14 | `outputs/figures/calibration/calibration_curve_day14.png` |
| Figure 12. Calibration curve for Day 28 | `outputs/figures/calibration/calibration_curve_day28.png` |
| Figure 13. Calibration curve for Day 56 | `outputs/figures/calibration/calibration_curve_day56.png` |
| Figure 14. Calibration curve for Day 84 | `outputs/figures/calibration/calibration_curve_day84.png` |
| Figure 15. Calibration curve for Full period | `outputs/figures/calibration/calibration_curve_full.png` |

---

## 10. Q1 Manuscript Positioning

### 10.1 Recommended Paper Title

**Explainable, Calibrated, and Fairness-Aware Temporal Learning Analytics for Early Identification of At-Risk Students in Online Higher Education**

Alternative shorter title:

**Explainable Temporal Learning Analytics for Early Identification of At-Risk Students in Online Higher Education**

### 10.2 Suggested Abstract

Digital learning environments generate rich behavioral traces that can support early identification of students at risk of academic failure or withdrawal. However, many predictive models are evaluated under simplified random-split settings and provide limited evidence regarding temporal validity, interpretability, subgroup reliability, cross-course generalization, and probability calibration. This study proposes an explainable temporal learning analytics framework for early identification of at-risk students using the Open University Learning Analytics Dataset. The framework integrates demographic, course-context, VLE engagement, and assessment-related features across multiple observation windows. Models are evaluated under standard early prediction, active-at-window prediction, feature-group ablation, cross-module validation, subgroup analysis, calibration analysis, and bootstrap confidence intervals. Results show that standard early prediction performance improved from a ROC-AUC of 0.8012 at day 14 to 0.9160 at day 84, while the full-period upper-bound setting achieved 0.9894. Active-at-window evaluation produced more conservative but practically relevant performance estimates, with ROC-AUC increasing from 0.7476 to 0.8574. SHAP analysis revealed a temporal shift from engagement and contextual indicators in early windows toward assessment and submission-related indicators in later windows. Calibration and threshold analysis further showed that stage-specific thresholds improved recall-oriented early-warning utility. These findings demonstrate the importance of temporally constrained, interpretable, calibrated, and fairness-aware evaluation for educational early-warning systems.

### 10.3 Suggested Research Questions

**RQ1.** How accurately can temporal learning analytics features identify at-risk students across different observation windows?

**RQ2.** How do active-at-window constraints affect early-warning performance when students who already withdrew before the prediction window are excluded?

**RQ3.** Which feature groups contribute most to early risk prediction, and how does their contribution change over time?

**RQ4.** Which features explain model predictions across early, mid-course, and full-period settings?

**RQ5.** Does the model maintain consistent performance across demographic and contextual student subgroups?

**RQ6.** How well does the model generalize to unseen course modules?

**RQ7.** Are the predicted probabilities sufficiently calibrated for decision-support use, and what thresholds are suitable for intervention-oriented screening?

### 10.4 Suggested Manuscript Structure

```text
1. Introduction
   1.1 Background and Motivation
   1.2 Problem Statement
   1.3 Research Gaps
   1.4 Contributions

2. Related Work
   2.1 Learning Analytics and Educational Data Mining
   2.2 At-Risk Student Prediction
   2.3 Temporal Learning Behavior Modeling
   2.4 Explainable AI in Education
   2.5 Fairness and Calibration in Educational Prediction

3. Materials and Methods
   3.1 Dataset Description
   3.2 Prediction Task and Target Definition
   3.3 Temporal Observation Windows
   3.4 Feature Engineering
   3.5 Model Development
   3.6 Evaluation Settings
   3.7 Explainability, Fairness, Calibration, and Bootstrap CI

4. Results
   4.1 Dataset Characteristics
   4.2 Standard Early Prediction
   4.3 Active-at-Window Prediction
   4.4 Feature-Group Ablation
   4.5 SHAP Explainability
   4.6 Fairness and Subgroup Analysis
   4.7 Cross-Module Validation
   4.8 Calibration, Threshold, and Bootstrap CI

5. Discussion
   5.1 Temporal Evolution of Predictive Evidence
   5.2 Practical Early-Warning Implications
   5.3 Generalization and Module-Level Distribution Shift
   5.4 Fairness and Responsible Use
   5.5 Calibration and Stage-Specific Thresholds

6. Limitations

7. Conclusion
```

---

## 11. Key Findings

1. Standard early prediction performance improved consistently as the observation window increased.
2. The full-period setting achieved very high performance but should be interpreted only as an upper-bound benchmark.
3. Active-at-window evaluation produced lower but more realistic early-warning estimates.
4. All features consistently outperformed individual feature groups.
5. VLE engagement features were informative in early windows, while assessment-related features became more influential in later windows.
6. SHAP analysis confirmed a temporal shift from engagement/contextual indicators to assessment/submission indicators.
7. Subgroup gaps were largest in early windows, especially across highest education and IMD band.
8. Cross-module validation showed meaningful but variable generalization, with module GGG being the most challenging holdout module.
9. Calibration analysis showed low ECE values, suggesting reasonably calibrated probabilities.
10. Stage-specific thresholds improved early-warning recall and F1-score.
11. Bootstrap confidence intervals indicated stable performance estimates.

---

## 12. Limitations

1. The dataset is observational; therefore, predictive associations should not be interpreted as causal relationships.
2. Some demographic and contextual variables contributed to prediction, but these variables are not directly actionable and require careful ethical interpretation.
3. Cross-module validation showed that model generalization varies across modules, especially for module GGG.
4. The framework does not yet implement fairness-aware optimization or post-hoc bias mitigation.
5. The active-at-window setting is stricter and more realistic, but additional validation on external datasets would be needed for broader generalizability.
6. Full-period results may contain complete-course behavioral evidence and should not be interpreted as deployable early-warning performance.

---

## 13. Future Work

Future work should investigate:

1. Fairness-aware threshold adjustment.
2. Module-specific calibration strategies.
3. External validation on additional online learning datasets.
4. Intervention-oriented evaluation with teacher feedback.
5. Sequential deep learning models under strict temporal and active-at-window constraints.
6. Causal analysis of educational interventions following risk prediction.
7. Deployment-oriented dashboards for instructor decision support.

---

## 14. Citation

If this repository is used in academic work, cite the project as:

```bibtex
@misc{oulad_explainable_early_warning,
  title        = {Explainable Temporal Learning Analytics for Early Identification of At-Risk Students in Online Higher Education},
  author       = {Your Name},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Temporal learning analytics, early-warning prediction, SHAP explainability, fairness analysis, calibration, and bootstrap confidence intervals using OULAD}
}
```

---

## 15. License and Ethical Use

This repository is intended for academic research and educational analytics experimentation. The dataset is anonymized, but student-risk prediction remains a sensitive educational application. Predictions should be used only as decision-support signals and should not be used as deterministic labels for punitive or exclusionary decisions.

Responsible use principles:

1. Use predictions to support students, not to penalize them.
2. Interpret demographic and socioeconomic variables with caution.
3. Monitor subgroup performance before deployment.
4. Prefer human-in-the-loop intervention decisions.
5. Avoid using the model as a fully automated decision-making system.

---

## 16. Q1 Result Tables and Rendered Figures

This section provides repository-accessible supplementary tables and rendered figures for the Q1-oriented experimental framework. Raw and processed datasets are intentionally excluded from the repository to avoid redistributing large educational data files. Users should download the original OULAD dataset separately and run the scripts in `src/` to reproduce the results.

### 16.1 Supplementary Result Tables

| Table File | Description |
|---|---|
| [`dataset_summary.csv`](docs/tables/dataset_summary.csv) | Summary of raw OULAD tables |
| [`target_distribution.csv`](docs/tables/target_distribution.csv) | Distribution of original and binary target labels |
| [`processed_feature_summary.csv`](docs/tables/processed_feature_summary.csv) | Summary of processed features |
| [`early_feature_table_summary.csv`](docs/tables/early_feature_table_summary.csv) | Summary of generated early-window feature tables |
| [`baseline_model_results.csv`](docs/tables/baseline_model_results.csv) | Full-period baseline model results |
| [`baseline_confusion_matrices.csv`](docs/tables/baseline_confusion_matrices.csv) | Baseline confusion matrices |
| [`early_prediction_results.csv`](docs/tables/early_prediction_results.csv) | Standard early prediction results for all models and windows |
| [`early_prediction_best_by_window.csv`](docs/tables/early_prediction_best_by_window.csv) | Best standard early prediction model by observation window |
| [`ablation_study_results.csv`](docs/tables/ablation_study_results.csv) | Complete feature-group ablation results |
| [`ablation_best_by_group.csv`](docs/tables/ablation_best_by_group.csv) | Best ablation result per feature group |
| [`ablation_best_per_window.csv`](docs/tables/ablation_best_per_window.csv) | Best feature group per observation window |
| [`active_window_prediction_results.csv`](docs/tables/active_window_prediction_results.csv) | Active-at-window early prediction results |
| [`active_window_dataset_summary.csv`](docs/tables/active_window_dataset_summary.csv) | Number of active students retained per observation window |
| [`active_window_best_by_window.csv`](docs/tables/active_window_best_by_window.csv) | Best active-at-window model by observation window |
| [`cross_module_validation_results.csv`](docs/tables/cross_module_validation_results.csv) | Complete cross-module validation results |
| [`cross_module_best_by_holdout.csv`](docs/tables/cross_module_best_by_holdout.csv) | Best model for each holdout module |
| [`cross_module_summary_by_window.csv`](docs/tables/cross_module_summary_by_window.csv) | Cross-module validation summary by observation window |
| [`cross_module_worst_by_window.csv`](docs/tables/cross_module_worst_by_window.csv) | Worst holdout module per observation window |
| [`shap_raw_importance_all_windows.csv`](docs/tables/shap_raw_importance_all_windows.csv) | SHAP raw feature importance across all windows |
| [`shap_top10_raw_features_all_windows.csv`](docs/tables/shap_top10_raw_features_all_windows.csv) | Top 10 SHAP-ranked features per observation window |
| [`top_subgroup_gaps.csv`](docs/tables/top_subgroup_gaps.csv) | Largest subgroup performance gaps |
| [`subgroup_performance_results.csv`](docs/tables/subgroup_performance_results.csv) | Full subgroup performance analysis |
| [`subgroup_performance_gaps.csv`](docs/tables/subgroup_performance_gaps.csv) | Subgroup performance-gap metrics |
| [`calibration_threshold_summary.csv`](docs/tables/calibration_threshold_summary.csv) | Calibration and threshold summary |
| [`threshold_analysis_all_windows.csv`](docs/tables/threshold_analysis_all_windows.csv) | Threshold-specific precision, recall, specificity, and F1 |
| [`calibration_bins_all_windows.csv`](docs/tables/calibration_bins_all_windows.csv) | Calibration-bin statistics |
| [`bootstrap_confidence_interval_summary.csv`](docs/tables/bootstrap_confidence_interval_summary.csv) | Bootstrap 95% confidence intervals |
| [`bootstrap_metric_samples_all_windows.csv`](docs/tables/bootstrap_metric_samples_all_windows.csv) | Bootstrap metric samples for reproducibility |

### 16.2 SHAP Feature Importance Figures

#### Figure 16.1. SHAP feature importance, Day 14

![SHAP bar Day 14](docs/figures/shap/shap_bar_day14.png)

#### Figure 16.2. SHAP feature importance, Day 28

![SHAP bar Day 28](docs/figures/shap/shap_bar_day28.png)

#### Figure 16.3. SHAP feature importance, Day 56

![SHAP bar Day 56](docs/figures/shap/shap_bar_day56.png)

#### Figure 16.4. SHAP feature importance, Day 84

![SHAP bar Day 84](docs/figures/shap/shap_bar_day84.png)

#### Figure 16.5. SHAP feature importance, Full period

![SHAP bar Full](docs/figures/shap/shap_bar_full.png)

### 16.3 SHAP Beeswarm Figures

#### Figure 16.6. SHAP beeswarm, Day 14

![SHAP beeswarm Day 14](docs/figures/shap/shap_beeswarm_day14.png)

#### Figure 16.7. SHAP beeswarm, Day 28

![SHAP beeswarm Day 28](docs/figures/shap/shap_beeswarm_day28.png)

#### Figure 16.8. SHAP beeswarm, Day 56

![SHAP beeswarm Day 56](docs/figures/shap/shap_beeswarm_day56.png)

#### Figure 16.9. SHAP beeswarm, Day 84

![SHAP beeswarm Day 84](docs/figures/shap/shap_beeswarm_day84.png)

#### Figure 16.10. SHAP beeswarm, Full period

![SHAP beeswarm Full](docs/figures/shap/shap_beeswarm_full.png)

### 16.4 Calibration Curve Figures

#### Figure 16.11. Calibration curve, Day 14

![Calibration Day 14](docs/figures/calibration/calibration_curve_day14.png)

#### Figure 16.12. Calibration curve, Day 28

![Calibration Day 28](docs/figures/calibration/calibration_curve_day28.png)

#### Figure 16.13. Calibration curve, Day 56

![Calibration Day 56](docs/figures/calibration/calibration_curve_day56.png)

#### Figure 16.14. Calibration curve, Day 84

![Calibration Day 84](docs/figures/calibration/calibration_curve_day84.png)

#### Figure 16.15. Calibration curve, Full period

![Calibration Full](docs/figures/calibration/calibration_curve_full.png)
