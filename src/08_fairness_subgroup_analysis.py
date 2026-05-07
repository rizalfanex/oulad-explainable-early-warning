from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables" / "fairness"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FAIRNESS / SUBGROUP ANALYSIS - OULAD")
print("=" * 80)

datasets = {
    "day14": DATA_DIR / "oulad_features_day14.csv",
    "day28": DATA_DIR / "oulad_features_day28.csv",
    "day56": DATA_DIR / "oulad_features_day56.csv",
    "day84": DATA_DIR / "oulad_features_day84.csv",
    "full": DATA_DIR / "oulad_features_full.csv",
}

TARGET_COL = "at_risk"

DROP_COLS = [
    "at_risk",
    "final_result",
    "id_student",
    "date_unregistration",
]

SUBGROUP_COLS = [
    "gender",
    "disability",
    "age_band",
    "imd_band",
    "highest_education",
    "region",
]

MIN_GROUP_SIZE = 50
RANDOM_STATE = 42


def build_pipeline(X_train):
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot_encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def safe_metrics(y_true, y_pred, y_prob):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    # ROC-AUC requires both classes to be present
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
        result["pr_auc"] = average_precision_score(y_true, y_prob)

    return result


all_rows = []
gap_rows = []

for window_name, data_path in datasets.items():
    print("\n" + "=" * 80)
    print(f"[WINDOW] {window_name}")
    print("=" * 80)

    if not data_path.exists():
        print(f"[SKIP] Missing file: {data_path}")
        continue

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {data_path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    y = df[TARGET_COL].astype(int)

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("[INFO] Training XGBoost all-features model...")
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    overall_metrics = safe_metrics(y_test, y_pred, y_prob)
    print(
        f"[OVERALL TEST] "
        f"Acc={overall_metrics['accuracy']:.4f} | "
        f"Recall={overall_metrics['recall']:.4f} | "
        f"F1={overall_metrics['f1']:.4f} | "
        f"ROC-AUC={overall_metrics['roc_auc']:.4f}"
    )

    # Keep subgroup columns from X_test index
    X_test_original = X_test.copy()
    X_test_original["_y_true"] = y_test.values
    X_test_original["_y_pred"] = y_pred
    X_test_original["_y_prob"] = y_prob

    for subgroup_col in SUBGROUP_COLS:
        if subgroup_col not in X_test_original.columns:
            print(f"[WARNING] Subgroup column missing: {subgroup_col}")
            continue

        print(f"\n[SUBGROUP] {subgroup_col}")

        subgroup_results = []

        for group_value, group_df in X_test_original.groupby(subgroup_col):
            n = len(group_df)

            if n < MIN_GROUP_SIZE:
                continue

            y_true_g = group_df["_y_true"].astype(int).values
            y_pred_g = group_df["_y_pred"].astype(int).values
            y_prob_g = group_df["_y_prob"].astype(float).values

            metrics = safe_metrics(y_true_g, y_pred_g, y_prob_g)

            row = {
                "window": window_name,
                "subgroup": subgroup_col,
                "group_value": group_value,
                "n_samples": n,
                "at_risk_rate": float(np.mean(y_true_g)),
                "predicted_at_risk_rate": float(np.mean(y_pred_g)),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }

            subgroup_results.append(row)
            all_rows.append(row)

        subgroup_df = pd.DataFrame(subgroup_results)

        if len(subgroup_df) == 0:
            print("  [SKIP] No group with sufficient sample size")
            continue

        print(
            subgroup_df[
                [
                    "group_value",
                    "n_samples",
                    "at_risk_rate",
                    "accuracy",
                    "recall",
                    "f1",
                    "roc_auc",
                ]
            ].to_string(index=False)
        )

        # Compute gap metrics
        for metric in ["accuracy", "recall", "f1", "roc_auc", "pr_auc"]:
            valid_values = subgroup_df[metric].dropna()
            if len(valid_values) >= 2:
                gap = valid_values.max() - valid_values.min()
                gap_rows.append({
                    "window": window_name,
                    "subgroup": subgroup_col,
                    "metric": metric,
                    "min_value": valid_values.min(),
                    "max_value": valid_values.max(),
                    "gap": gap,
                    "n_groups": len(valid_values),
                })

all_results = pd.DataFrame(all_rows)
gap_results = pd.DataFrame(gap_rows)

out_path = OUT_TABLE_DIR / "subgroup_performance_results.csv"
gap_path = OUT_TABLE_DIR / "subgroup_performance_gaps.csv"

all_results.to_csv(out_path, index=False)
gap_results.to_csv(gap_path, index=False)

print("\n" + "=" * 80)
print("[DONE] Fairness subgroup analysis completed")
print(f"Subgroup results : {out_path}")
print(f"Gap results      : {gap_path}")
print("=" * 80)

if len(gap_results) > 0:
    print("\nTOP 20 LARGEST SUBGROUP PERFORMANCE GAPS")
    top_gaps = gap_results.sort_values("gap", ascending=False).head(20)
    print(top_gaps.to_string(index=False))

    top_gap_path = OUT_TABLE_DIR / "top_subgroup_gaps.csv"
    top_gaps.to_csv(top_gap_path, index=False)
    print(f"\n[DONE] Top gaps saved to: {top_gap_path}")
