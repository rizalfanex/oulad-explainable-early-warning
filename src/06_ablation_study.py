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
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ABLATION STUDY - OULAD EARLY RISK PREDICTION")
print("=" * 80)

datasets = {
    "day14": DATA_DIR / "oulad_features_day14.csv",
    "day28": DATA_DIR / "oulad_features_day28.csv",
    "day56": DATA_DIR / "oulad_features_day56.csv",
    "day84": DATA_DIR / "oulad_features_day84.csv",
    "full": DATA_DIR / "oulad_features_full.csv",
}

TARGET_COL = "at_risk"

BASE_DROP_COLS = [
    "at_risk",
    "final_result",
    "id_student",
    "date_unregistration",
]

# ============================================================
# Feature group definitions
# ============================================================

demographic_cols = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
]

course_cols = [
    "code_module",
    "code_presentation",
    "module_presentation_length",
    "date_registration",
]

# Prefix/pattern based feature selection
vle_patterns = [
    "click",
    "active_days",
    "first_vle_day",
    "last_vle_day",
    "unique_sites",
    "unique_activity_types",
    "vle_span_days",
    "clicks_per_active_day",
    "activity_density",
    "mean_clicks",
    "max_clicks",
    "total_clicks",
]

assessment_patterns = [
    "assessment",
    "score",
    "submission",
    "late",
    "banked",
    "weighted",
]


def select_existing_columns(df, explicit_cols=None, patterns=None):
    selected = []

    explicit_cols = explicit_cols or []
    patterns = patterns or []

    for col in explicit_cols:
        if col in df.columns:
            selected.append(col)

    for col in df.columns:
        for pattern in patterns:
            if pattern.lower() in col.lower():
                selected.append(col)
                break

    # Remove duplicates while preserving order
    selected = list(dict.fromkeys(selected))

    # Remove forbidden columns if accidentally selected
    selected = [c for c in selected if c not in BASE_DROP_COLS]

    return selected


def build_model(model_name):
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        )

    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not available.")
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(X_train, model_name):
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

    model = build_model(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline, numeric_features, categorical_features


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


all_results = []

model_names = ["HistGradientBoosting"]
if XGBOOST_AVAILABLE:
    model_names.append("XGBoost")

for window_name, data_path in datasets.items():
    print("\n" + "=" * 80)
    print(f"[WINDOW] {window_name}")
    print("=" * 80)

    df = pd.read_csv(data_path)
    y = df[TARGET_COL].astype(int)

    # Build feature groups dynamically for each window
    demo = select_existing_columns(df, explicit_cols=demographic_cols)
    course = select_existing_columns(df, explicit_cols=course_cols)
    vle = select_existing_columns(df, patterns=vle_patterns)
    assessment = select_existing_columns(df, patterns=assessment_patterns)

    all_valid_features = [
        c for c in df.columns
        if c not in BASE_DROP_COLS
    ]

    feature_groups = {
        "Demographic only": demo,
        "Course only": course,
        "VLE only": vle,
        "Assessment only": assessment,
        "Demographic + VLE": list(dict.fromkeys(demo + vle)),
        "Demographic + Assessment": list(dict.fromkeys(demo + assessment)),
        "VLE + Assessment": list(dict.fromkeys(vle + assessment)),
        "Demographic + Course + VLE": list(dict.fromkeys(demo + course + vle)),
        "Demographic + Course + Assessment": list(dict.fromkeys(demo + course + assessment)),
        "All features": all_valid_features,
    }

    print("[INFO] Feature group sizes:")
    for group_name, cols in feature_groups.items():
        print(f"  - {group_name:35s}: {len(cols)}")

    for group_name, cols in feature_groups.items():
        if len(cols) == 0:
            print(f"[SKIP] {group_name}: no features")
            continue

        X = df[cols].copy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp,
        )

        for model_name in model_names:
            print("\n" + "-" * 80)
            print(f"[TRAIN] {window_name} | {group_name} | {model_name}")
            print("-" * 80)

            pipeline, num_cols, cat_cols = build_pipeline(X_train, model_name)
            pipeline.fit(X_train, y_train)

            val_metrics = evaluate(pipeline, X_val, y_val)
            test_metrics = evaluate(pipeline, X_test, y_test)

            row = {
                "window": window_name,
                "feature_group": group_name,
                "model": model_name,
                "n_features_raw": len(cols),
                "n_numeric": len(num_cols),
                "n_categorical": len(cat_cols),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_tn": test_metrics["tn"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_tp": test_metrics["tp"],
            }

            all_results.append(row)

            print(
                f"VAL  | Acc={val_metrics['accuracy']:.4f} | "
                f"F1={val_metrics['f1']:.4f} | "
                f"ROC-AUC={val_metrics['roc_auc']:.4f} | "
                f"PR-AUC={val_metrics['pr_auc']:.4f}"
            )

            print(
                f"TEST | Acc={test_metrics['accuracy']:.4f} | "
                f"Prec={test_metrics['precision']:.4f} | "
                f"Recall={test_metrics['recall']:.4f} | "
                f"F1={test_metrics['f1']:.4f} | "
                f"ROC-AUC={test_metrics['roc_auc']:.4f} | "
                f"PR-AUC={test_metrics['pr_auc']:.4f}"
            )

results_df = pd.DataFrame(all_results)

out_path = OUT_TABLE_DIR / "ablation_study_results.csv"
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 80)
print(f"[DONE] Saved ablation results to: {out_path}")
print("=" * 80)

# Best model for each window-feature group based on validation ROC-AUC
best_by_group = (
    results_df.sort_values(["window", "feature_group", "val_roc_auc"], ascending=[True, True, False])
    .groupby(["window", "feature_group"])
    .head(1)
    .reset_index(drop=True)
)

best_by_group_path = OUT_TABLE_DIR / "ablation_best_by_group.csv"
best_by_group.to_csv(best_by_group_path, index=False)

print(f"[DONE] Saved best-by-group results to: {best_by_group_path}")

# Compact summary: all feature groups, best model per group
compact_cols = [
    "window",
    "feature_group",
    "model",
    "n_features_raw",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
]

print("\nBEST ABLATION RESULTS BY WINDOW AND FEATURE GROUP")
print(best_by_group[compact_cols].to_string(index=False))

# Best feature group per window
best_per_window = (
    best_by_group.sort_values(["window", "test_roc_auc"], ascending=[True, False])
    .groupby("window")
    .head(1)
    .reset_index(drop=True)
)

best_per_window_path = OUT_TABLE_DIR / "ablation_best_per_window.csv"
best_per_window.to_csv(best_per_window_path, index=False)

print("\nBEST FEATURE GROUP PER WINDOW")
print(best_per_window[compact_cols].to_string(index=False))

print(f"\n[DONE] Saved best-per-window results to: {best_per_window_path}")
