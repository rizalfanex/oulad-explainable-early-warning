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
print("BUILDING EARLY-PREDICTION OULAD FEATURES")
print("=" * 80)

# ============================================================
# Load data
# ============================================================

student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
student_reg = pd.read_csv(DATA_DIR / "studentRegistration.csv")
student_assess = pd.read_csv(DATA_DIR / "studentAssessment.csv")
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
vle = pd.read_csv(DATA_DIR / "vle.csv")
courses = pd.read_csv(DATA_DIR / "courses.csv")

key_cols = ["code_module", "code_presentation", "id_student"]

student_info["at_risk"] = student_info["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

base = student_info.merge(courses, on=["code_module", "code_presentation"], how="left")
base = base.merge(student_reg, on=key_cols, how="left")

# Avoid direct leakage from withdrawal date
if "date_unregistration" in base.columns:
    base = base.drop(columns=["date_unregistration"])

student_vle_enriched = student_vle.merge(
    vle[["id_site", "code_module", "code_presentation", "activity_type"]],
    on=["id_site", "code_module", "code_presentation"],
    how="left"
)

assess_enriched = student_assess.merge(
    assessments,
    on="id_assessment",
    how="left"
)

assess_enriched["submission_delay"] = assess_enriched["date_submitted"] - assess_enriched["date"]
assess_enriched["weighted_score"] = assess_enriched["score"] * assess_enriched["weight"] / 100.0

windows = [14, 28, 56, 84]

summary_rows = []

for w in windows:
    print("\n" + "-" * 80)
    print(f"[WINDOW] Building features up to day {w}")
    print("-" * 80)

    # ========================================================
    # VLE features up to day w
    # ========================================================

    vle_w = student_vle_enriched[student_vle_enriched["date"] <= w].copy()

    if len(vle_w) > 0:
        vle_basic = vle_w.groupby(key_cols).agg(
            total_clicks=("sum_click", "sum"),
            mean_clicks_per_record=("sum_click", "mean"),
            max_clicks_per_record=("sum_click", "max"),
            active_days=("date", "nunique"),
            first_vle_day=("date", "min"),
            last_vle_day=("date", "max"),
            unique_sites=("id_site", "nunique"),
            unique_activity_types=("activity_type", "nunique")
        ).reset_index()

        vle_basic["vle_span_days"] = vle_basic["last_vle_day"] - vle_basic["first_vle_day"] + 1
        vle_basic["clicks_per_active_day"] = (
            vle_basic["total_clicks"] / vle_basic["active_days"].replace(0, np.nan)
        )
        vle_basic["activity_density"] = (
            vle_basic["active_days"] / vle_basic["vle_span_days"].replace(0, np.nan)
        )

        vle_activity = vle_w.pivot_table(
            index=key_cols,
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        vle_activity.columns = [str(c) for c in vle_activity.columns]
        activity_cols = [c for c in vle_activity.columns if c not in key_cols]
        vle_activity = vle_activity.rename(columns={c: f"click_{c}" for c in activity_cols})

        vle_features = vle_basic.merge(vle_activity, on=key_cols, how="left")
    else:
        vle_features = base[key_cols].copy()

    # ========================================================
    # Assessment features up to day w
    # ========================================================

    assess_w = assess_enriched[assess_enriched["date_submitted"] <= w].copy()

    if len(assess_w) > 0:
        assessment_features = assess_w.groupby(key_cols).agg(
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
            banked_count=("is_banked", "sum"),
            total_weighted_score=("weighted_score", "sum"),
            total_assessment_weight=("weight", "sum")
        ).reset_index()

        assessment_features["assessment_span_days"] = (
            assessment_features["last_submission_day"] - assessment_features["first_submission_day"] + 1
        )

        assessment_features["late_submission_ratio"] = (
            assessment_features["late_submission_count"] /
            assessment_features["assessment_count"].replace(0, np.nan)
        )

        assessment_features["weighted_score_ratio"] = (
            assessment_features["total_weighted_score"] /
            assessment_features["total_assessment_weight"].replace(0, np.nan)
        )
    else:
        assessment_features = base[key_cols].copy()

    # ========================================================
    # Merge all features
    # ========================================================

    data_w = base.merge(vle_features, on=key_cols, how="left")
    data_w = data_w.merge(assessment_features, on=key_cols, how="left")

    # Fill numeric missing values with 0
    numeric_cols = data_w.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != "at_risk":
            data_w[col] = data_w[col].fillna(0)

    # Fill categorical missing values
    categorical_cols = data_w.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        data_w[col] = data_w[col].fillna("Unknown")

    out_path = OUT_DIR / f"oulad_features_day{w}.csv"
    data_w.to_csv(out_path, index=False)

    print(f"[DONE] Saved: {out_path}")
    print(f"[INFO] Shape: {data_w.shape[0]:,} rows x {data_w.shape[1]:,} columns")
    print("[INFO] Target distribution:")
    print(data_w["at_risk"].value_counts())

    summary_rows.append({
        "window_day": w,
        "rows": data_w.shape[0],
        "columns": data_w.shape[1],
        "at_risk": int(data_w["at_risk"].sum()),
        "non_risk": int((data_w["at_risk"] == 0).sum())
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLE_DIR / "early_feature_table_summary.csv", index=False)

print("\n" + "=" * 80)
print("[DONE] All early feature tables generated")
print("=" * 80)
print(summary_df)
