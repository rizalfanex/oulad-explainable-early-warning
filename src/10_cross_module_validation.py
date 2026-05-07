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

from sklearn.linear_model import LogisticRegression
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

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CROSS-MODULE VALIDATION - OULAD")
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

RANDOM_STATE = 42


def safe_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
        result["pr_auc"] = average_precision_score(y_true, y_prob)

    return result


def build_preprocessor(X_train):
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
        sparse_threshold=0,
    )

    return preprocessor, numeric_features, categorical_features


def build_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


all_rows = []

for window_name, data_path in datasets.items():
    print("\n" + "=" * 80)
    print(f"[WINDOW] {window_name}")
    print("=" * 80)

    if not data_path.exists():
        print(f"[SKIP] Missing file: {data_path}")
        continue

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {data_path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    if "code_module" not in df.columns:
        raise ValueError("code_module column is required for cross-module validation.")

    modules = sorted(df["code_module"].unique().tolist())
    print(f"[INFO] Modules: {modules}")

    for holdout_module in modules:
        print("\n" + "-" * 80)
        print(f"[HOLDOUT MODULE] {holdout_module}")
        print("-" * 80)

        train_df = df[df["code_module"] != holdout_module].copy()
        test_df = df[df["code_module"] == holdout_module].copy()

        print(f"[INFO] Train modules rows : {len(train_df):,}")
        print(f"[INFO] Test module rows   : {len(test_df):,}")

        if len(test_df) < 100:
            print("[SKIP] Test module too small.")
            continue

        y_train_full = train_df[TARGET_COL].astype(int)
        y_test = test_df[TARGET_COL].astype(int)

        if len(np.unique(y_test)) < 2:
            print("[SKIP] Holdout module has only one class.")
            continue

        drop_cols_train = [c for c in DROP_COLS if c in train_df.columns]
        drop_cols_test = [c for c in DROP_COLS if c in test_df.columns]

        X_train_full = train_df.drop(columns=drop_cols_train)
        X_test = test_df.drop(columns=drop_cols_test)

        # Internal validation split from training modules only
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=y_train_full,
        )

        print(f"[INFO] Internal train : {len(X_train):,}")
        print(f"[INFO] Internal val   : {len(X_val):,}")
        print(f"[INFO] Holdout test   : {len(X_test):,}")
        print("[INFO] Holdout target distribution:")
        print(y_test.value_counts())
        print((y_test.value_counts(normalize=True) * 100).round(2))

        preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

        print(f"[INFO] Numeric features     : {len(numeric_features)}")
        print(f"[INFO] Categorical features : {len(categorical_features)}")

        models = build_models()

        for model_name, model in models.items():
            print(f"\n[TRAIN] {window_name} | holdout={holdout_module} | {model_name}")

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            # Validation performance on seen modules
            y_val_pred = pipeline.predict(X_val)
            y_val_prob = pipeline.predict_proba(X_val)[:, 1]
            val_metrics = safe_metrics(y_val, y_val_pred, y_val_prob)

            # Test performance on unseen module
            y_test_pred = pipeline.predict(X_test)
            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
            test_metrics = safe_metrics(y_test, y_test_pred, y_test_prob)

            print(
                f"VAL seen-modules | "
                f"Acc={val_metrics['accuracy']:.4f} | "
                f"F1={val_metrics['f1']:.4f} | "
                f"ROC-AUC={val_metrics['roc_auc']:.4f} | "
                f"PR-AUC={val_metrics['pr_auc']:.4f}"
            )

            print(
                f"TEST unseen-module | "
                f"Acc={test_metrics['accuracy']:.4f} | "
                f"Prec={test_metrics['precision']:.4f} | "
                f"Recall={test_metrics['recall']:.4f} | "
                f"F1={test_metrics['f1']:.4f} | "
                f"ROC-AUC={test_metrics['roc_auc']:.4f} | "
                f"PR-AUC={test_metrics['pr_auc']:.4f}"
            )

            row = {
                "window": window_name,
                "holdout_module": holdout_module,
                "model": model_name,
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "test_at_risk_rate": float(y_test.mean()),
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

            all_rows.append(row)


results_df = pd.DataFrame(all_rows)

results_path = OUT_TABLE_DIR / "cross_module_validation_results.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 80)
print("[DONE] Cross-module validation completed")
print(f"Results: {results_path}")
print("=" * 80)

# Best model per window and holdout module based on validation ROC-AUC
best_by_holdout = (
    results_df
    .sort_values(["window", "holdout_module", "val_roc_auc"], ascending=[True, True, False])
    .groupby(["window", "holdout_module"])
    .head(1)
    .reset_index(drop=True)
)

best_by_holdout_path = OUT_TABLE_DIR / "cross_module_best_by_holdout.csv"
best_by_holdout.to_csv(best_by_holdout_path, index=False)

# Summary across holdout modules
summary = (
    best_by_holdout
    .groupby("window")
    .agg(
        n_holdout_modules=("holdout_module", "nunique"),
        mean_accuracy=("test_accuracy", "mean"),
        std_accuracy=("test_accuracy", "std"),
        mean_precision=("test_precision", "mean"),
        std_precision=("test_precision", "std"),
        mean_recall=("test_recall", "mean"),
        std_recall=("test_recall", "std"),
        mean_f1=("test_f1", "mean"),
        std_f1=("test_f1", "std"),
        mean_roc_auc=("test_roc_auc", "mean"),
        std_roc_auc=("test_roc_auc", "std"),
        mean_pr_auc=("test_pr_auc", "mean"),
        std_pr_auc=("test_pr_auc", "std"),
    )
    .reset_index()
)

summary_path = OUT_TABLE_DIR / "cross_module_summary_by_window.csv"
summary.to_csv(summary_path, index=False)

print(f"[DONE] Best-by-holdout saved: {best_by_holdout_path}")
print(f"[DONE] Summary saved        : {summary_path}")

print("\nCROSS-MODULE SUMMARY BY WINDOW")
print(summary.to_string(index=False))

print("\nBEST MODEL PER HOLDOUT MODULE")
print(
    best_by_holdout[
        [
            "window",
            "holdout_module",
            "model",
            "test_rows",
            "test_at_risk_rate",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
    ].to_string(index=False)
)

# Worst-case module per window
worst_by_window = (
    best_by_holdout
    .sort_values(["window", "test_roc_auc"], ascending=[True, True])
    .groupby("window")
    .head(1)
    .reset_index(drop=True)
)

worst_path = OUT_TABLE_DIR / "cross_module_worst_by_window.csv"
worst_by_window.to_csv(worst_path, index=False)

print("\nWORST HOLDOUT MODULE PER WINDOW")
print(
    worst_by_window[
        [
            "window",
            "holdout_module",
            "model",
            "test_accuracy",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
    ].to_string(index=False)
)

print(f"\n[DONE] Worst-by-window saved: {worst_path}")
