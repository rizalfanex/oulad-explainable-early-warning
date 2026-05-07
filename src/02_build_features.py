from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "anonymisedData"
OUT_DIR = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BUILDING OULAD FEATURES")
print("=" * 80)

# ============================================================
# 1. Load data
# ============================================================

student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
student_reg = pd.read_csv(DATA_DIR / "studentRegistration.csv")
student_assess = pd.read_csv(DATA_DIR / "studentAssessment.csv")
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
vle = pd.read_csv(DATA_DIR / "vle.csv")
courses = pd.read_csv(DATA_DIR / "courses.csv")

print("[OK] Data loaded")

# ============================================================
# 2. Create target label
# ============================================================

student_info["at_risk"] = student_info["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

# Unique student-course key
key_cols = ["code_module", "code_presentation", "id_student"]

# ============================================================
# 3. Merge course and registration info
# ============================================================

base = student_info.merge(courses, on=["code_module", "code_presentation"], how="left")
base = base.merge(student_reg, on=key_cols, how="left")

print(f"[OK] Base table shape: {base.shape}")

# ============================================================
# 4. VLE feature engineering
# ============================================================

print("[INFO] Building VLE features...")

student_vle_enriched = student_vle.merge(
    vle[["id_site", "code_module", "code_presentation", "activity_type"]],
    on=["id_site", "code_module", "code_presentation"],
    how="left"
)

# Basic VLE engagement
vle_basic = student_vle_enriched.groupby(key_cols).agg(
    total_clicks=("sum_click", "sum"),
    mean_clicks_per_day_record=("sum_click", "mean"),
    max_clicks_in_day_record=("sum_click", "max"),
    active_days=("date", "nunique"),
    first_vle_day=("date", "min"),
    last_vle_day=("date", "max"),
    unique_sites=("id_site", "nunique"),
    unique_activity_types=("activity_type", "nunique")
).reset_index()

vle_basic["vle_span_days"] = vle_basic["last_vle_day"] - vle_basic["first_vle_day"] + 1
vle_basic["clicks_per_active_day"] = vle_basic["total_clicks"] / vle_basic["active_days"].replace(0, np.nan)
vle_basic["activity_density"] = vle_basic["active_days"] / vle_basic["vle_span_days"].replace(0, np.nan)

# Activity type pivot
vle_activity = student_vle_enriched.pivot_table(
    index=key_cols,
    columns="activity_type",
    values="sum_click",
    aggfunc="sum",
    fill_value=0
).reset_index()

vle_activity.columns = [
    col if isinstance(col, str) else str(col)
    for col in vle_activity.columns
]

activity_cols = [c for c in vle_activity.columns if c not in key_cols]
vle_activity = vle_activity.rename(columns={c: f"click_{c}" for c in activity_cols})

# Early VLE windows
windows = [14, 28, 56, 84]
window_features = []

for w in windows:
    tmp = student_vle_enriched[student_vle_enriched["date"] <= w].groupby(key_cols).agg(
        **{
            f"clicks_w{w}": ("sum_click", "sum"),
            f"active_days_w{w}": ("date", "nunique"),
            f"unique_sites_w{w}": ("id_site", "nunique"),
            f"unique_activity_types_w{w}": ("activity_type", "nunique"),
        }
    ).reset_index()

    tmp[f"clicks_per_active_day_w{w}"] = tmp[f"clicks_w{w}"] / tmp[f"active_days_w{w}"].replace(0, np.nan)
    window_features.append(tmp)

vle_features = vle_basic.merge(vle_activity, on=key_cols, how="left")

for tmp in window_features:
    vle_features = vle_features.merge(tmp, on=key_cols, how="left")

print(f"[OK] VLE features shape: {vle_features.shape}")

# ============================================================
# 5. Assessment feature engineering
# ============================================================

print("[INFO] Building assessment features...")

assess_enriched = student_assess.merge(
    assessments,
    on="id_assessment",
    how="left"
)

# Submission lateness relative to assessment date
assess_enriched["submission_delay"] = assess_enriched["date_submitted"] - assess_enriched["date"]

assessment_features = assess_enriched.groupby(key_cols).agg(
    assessment_count=("id_assessment", "count"),
    assessment_unique_count=("id_assessment", "nunique"),
    mean_score=("score", "mean"),
    min_score=("score", "min"),
    max_score=("score", "max"),
    std_score=("score", "std"),
    first_submission_day=("date_submitted", "min"),
    last_submission_day=("date_submitted", "max"),
    mean_submission_delay=("submission_delay", "mean"),
    late_submission_count=("submission_delay", lambda x: (x > 0).sum()),
    banked_count=("is_banked", "sum")
).reset_index()

assessment_features["assessment_span_days"] = (
    assessment_features["last_submission_day"] - assessment_features["first_submission_day"] + 1
)

assessment_features["late_submission_ratio"] = (
    assessment_features["late_submission_count"] / assessment_features["assessment_count"].replace(0, np.nan)
)

# Weighted score feature
assess_enriched["weighted_score"] = assess_enriched["score"] * assess_enriched["weight"] / 100.0

weighted_score = assess_enriched.groupby(key_cols).agg(
    total_weighted_score=("weighted_score", "sum"),
    total_assessment_weight=("weight", "sum")
).reset_index()

weighted_score["weighted_score_ratio"] = (
    weighted_score["total_weighted_score"] / weighted_score["total_assessment_weight"].replace(0, np.nan)
)

assessment_features = assessment_features.merge(weighted_score, on=key_cols, how="left")

# Early assessment windows
assessment_window_features = []

for w in windows:
    tmp = assess_enriched[assess_enriched["date_submitted"] <= w].groupby(key_cols).agg(
        **{
            f"assessment_count_w{w}": ("id_assessment", "count"),
            f"mean_score_w{w}": ("score", "mean"),
            f"max_score_w{w}": ("score", "max"),
            f"late_submission_count_w{w}": ("submission_delay", lambda x: (x > 0).sum()),
        }
    ).reset_index()

    assessment_window_features.append(tmp)

for tmp in assessment_window_features:
    assessment_features = assessment_features.merge(tmp, on=key_cols, how="left")

print(f"[OK] Assessment features shape: {assessment_features.shape}")

# ============================================================
# 6. Merge all features
# ============================================================

data = base.merge(vle_features, on=key_cols, how="left")
data = data.merge(assessment_features, on=key_cols, how="left")

# Fill missing numeric features
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col not in ["at_risk"]:
        data[col] = data[col].fillna(0)

# Fill missing categorical features
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    data[col] = data[col].fillna("Unknown")

# ============================================================
# 7. Save processed data
# ============================================================

out_path = OUT_DIR / "oulad_features_full.csv"
data.to_csv(out_path, index=False)

print("=" * 80)
print(f"[DONE] Saved processed feature table to: {out_path}")
print(f"[DONE] Final shape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
print("=" * 80)

# Save feature summary
summary = pd.DataFrame({
    "column": data.columns,
    "dtype": [str(data[c].dtype) for c in data.columns],
    "missing": [int(data[c].isna().sum()) for c in data.columns],
    "unique_values": [int(data[c].nunique()) for c in data.columns],
})

summary.to_csv(TABLE_DIR / "processed_feature_summary.csv", index=False)

print("[DONE] Saved processed feature summary")
print("\nTarget distribution:")
print(data["at_risk"].value_counts())
print("\nTarget percentage:")
print((data["at_risk"].value_counts(normalize=True) * 100).round(2))
