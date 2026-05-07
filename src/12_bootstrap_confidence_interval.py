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

OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables" / "bootstrap"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BOOTSTRAP CONFIDENCE INTERVAL - OULAD")
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
N_BOOTSTRAP = 1000
CI_LOWER = 2.5
CI_UPPER = 97.5


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
        sparse_threshold=0,
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


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
        result["pr_auc"] = average_precision_score(y_true, y_prob)

    return result


def bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    rows = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        y_true_b = y_true[idx]
        y_prob_b = y_prob[idx]

        # Skip bootstrap sample if only one class appears.
        if len(np.unique(y_true_b)) < 2:
            continue

        metrics = compute_metrics(y_true_b, y_prob_b, threshold=threshold)
        metrics["bootstrap_id"] = i
        rows.append(metrics)

    boot_df = pd.DataFrame(rows)

    summary_rows = []

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        values = boot_df[metric].dropna().values

        summary_rows.append({
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci_lower": float(np.percentile(values, CI_LOWER)),
            "ci_upper": float(np.percentile(values, CI_UPPER)),
            "n_bootstrap_valid": int(len(values)),
        })

    summary_df = pd.DataFrame(summary_rows)

    return boot_df, summary_df


# Optional: thresholds from calibration best-F1 result.
# These thresholds are useful for an intervention-oriented supplementary table.
BEST_F1_THRESHOLDS = {
    "day14": 0.35,
    "day28": 0.40,
    "day56": 0.45,
    "day84": 0.55,
    "full": 0.50,
}

all_summary_rows = []
all_bootstrap_rows = []

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

    print(f"[INFO] Train: {len(X_train):,}")
    print(f"[INFO] Val  : {len(X_val):,}")
    print(f"[INFO] Test : {len(X_test):,}")

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # A. Default threshold 0.50 CI
    # --------------------------------------------------------
    default_threshold = 0.50

    print(f"[INFO] Bootstrap CI at default threshold = {default_threshold:.2f}")
    boot_df, summary_df = bootstrap_ci(
        y_true=y_test.values,
        y_prob=y_test_prob,
        threshold=default_threshold,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )

    boot_df.insert(0, "window", window_name)
    boot_df.insert(1, "threshold_setting", "default_0.50")
    boot_df.insert(2, "threshold", default_threshold)

    summary_df.insert(0, "window", window_name)
    summary_df.insert(1, "threshold_setting", "default_0.50")
    summary_df.insert(2, "threshold", default_threshold)

    all_bootstrap_rows.append(boot_df)
    all_summary_rows.append(summary_df)

    # --------------------------------------------------------
    # B. Best-F1 threshold CI
    # --------------------------------------------------------
    best_threshold = BEST_F1_THRESHOLDS[window_name]

    print(f"[INFO] Bootstrap CI at best-F1 threshold = {best_threshold:.2f}")
    boot_df_best, summary_df_best = bootstrap_ci(
        y_true=y_test.values,
        y_prob=y_test_prob,
        threshold=best_threshold,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE + 100,
    )

    boot_df_best.insert(0, "window", window_name)
    boot_df_best.insert(1, "threshold_setting", "best_f1_threshold")
    boot_df_best.insert(2, "threshold", best_threshold)

    summary_df_best.insert(0, "window", window_name)
    summary_df_best.insert(1, "threshold_setting", "best_f1_threshold")
    summary_df_best.insert(2, "threshold", best_threshold)

    all_bootstrap_rows.append(boot_df_best)
    all_summary_rows.append(summary_df_best)

    print("\n[DEFAULT 0.50 THRESHOLD CI]")
    print(summary_df[["metric", "mean", "ci_lower", "ci_upper"]].to_string(index=False))

    print("\n[BEST-F1 THRESHOLD CI]")
    print(summary_df_best[["metric", "mean", "ci_lower", "ci_upper"]].to_string(index=False))


combined_bootstrap = pd.concat(all_bootstrap_rows, ignore_index=True)
combined_summary = pd.concat(all_summary_rows, ignore_index=True)

bootstrap_path = OUT_TABLE_DIR / "bootstrap_metric_samples_all_windows.csv"
summary_path = OUT_TABLE_DIR / "bootstrap_confidence_interval_summary.csv"

combined_bootstrap.to_csv(bootstrap_path, index=False)
combined_summary.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("[DONE] Bootstrap confidence interval analysis completed")
print(f"Bootstrap samples : {bootstrap_path}")
print(f"CI summary        : {summary_path}")
print("=" * 80)

print("\nBOOTSTRAP 95% CI SUMMARY")
print(
    combined_summary[
        [
            "window",
            "threshold_setting",
            "threshold",
            "metric",
            "mean",
            "ci_lower",
            "ci_upper",
            "n_bootstrap_valid",
        ]
    ].to_string(index=False)
)
