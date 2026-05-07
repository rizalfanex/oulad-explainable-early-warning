from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "anonymisedData"
OUT_DIR = BASE_DIR / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "assessments": "assessments.csv",
    "courses": "courses.csv",
    "studentAssessment": "studentAssessment.csv",
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentVle": "studentVle.csv",
    "vle": "vle.csv",
}

summary_rows = []

print("=" * 80)
print("OULAD DATASET CHECK")
print("=" * 80)

for name, filename in files.items():
    path = DATA_DIR / filename

    if not path.exists():
        print(f"[MISSING] {filename}")
        continue

    df = pd.read_csv(path)

    print(f"\n[{name}]")
    print(f"Path       : {path}")
    print(f"Shape      : {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Columns    : {list(df.columns)}")
    print("Missing values:")
    print(df.isna().sum())

    summary_rows.append({
        "table": name,
        "filename": filename,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": ", ".join(df.columns),
        "missing_total": int(df.isna().sum().sum())
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "dataset_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print(f"Saved summary to: {summary_path}")
print("=" * 80)

# Check target distribution
student_info_path = DATA_DIR / "studentInfo.csv"
if student_info_path.exists():
    student_info = pd.read_csv(student_info_path)

    print("\nTARGET DISTRIBUTION: final_result")
    if "final_result" in student_info.columns:
        print(student_info["final_result"].value_counts(dropna=False))
        print("\nPercentage:")
        print((student_info["final_result"].value_counts(normalize=True, dropna=False) * 100).round(2))

        target_dist = student_info["final_result"].value_counts(dropna=False).reset_index()
        target_dist.columns = ["final_result", "count"]
        target_dist["percentage"] = (target_dist["count"] / target_dist["count"].sum() * 100).round(2)
        target_dist.to_csv(OUT_DIR / "target_distribution.csv", index=False)
    else:
        print("[WARNING] final_result column not found in studentInfo.csv")
