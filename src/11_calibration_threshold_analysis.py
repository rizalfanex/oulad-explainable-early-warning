from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables" / "calibration"
OUT_FIG_DIR = BASE_DIR / "outputs" / "figures" / "calibration"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CALIBRATION AND THRESHOLD ANALYSIS - OULAD")
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
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


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


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    rows = []

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        n_bin = int(mask.sum())

        if n_bin == 0:
            rows.append({
                "bin": bin_id,
                "bin_lower": bins[bin_id],
                "bin_upper": bins[bin_id + 1],
                "n_samples": 0,
                "mean_predicted_probability": np.nan,
                "observed_at_risk_rate": np.nan,
                "absolute_gap": np.nan,
            })
            continue

        mean_prob = float(y_prob[mask].mean())
        observed_rate = float(y_true[mask].mean())
        abs_gap = abs(mean_prob - observed_rate)

        ece += (n_bin / len(y_true)) * abs_gap

        rows.append({
            "bin": bin_id,
            "bin_lower": bins[bin_id],
            "bin_upper": bins[bin_id + 1],
            "n_samples": n_bin,
            "mean_predicted_probability": mean_prob,
            "observed_at_risk_rate": observed_rate,
            "absolute_gap": abs_gap,
        })

    return float(ece), pd.DataFrame(rows)


def threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "predicted_at_risk_rate": float(y_pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


all_calibration_rows = []
all_threshold_rows = []
summary_rows = []

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

    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calibration metrics
    test_brier = brier_score_loss(y_test, y_test_prob)
    test_ece, calibration_bins = expected_calibration_error(y_test, y_test_prob, n_bins=10)

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)

    print(
        f"[TEST PROBABILITY] "
        f"ROC-AUC={roc_auc:.4f} | "
        f"PR-AUC={pr_auc:.4f} | "
        f"Brier={test_brier:.4f} | "
        f"ECE={test_ece:.4f}"
    )

    calibration_bins.insert(0, "window", window_name)
    all_calibration_rows.append(calibration_bins)

    calibration_bins.to_csv(
        OUT_TABLE_DIR / f"calibration_bins_{window_name}.csv",
        index=False
    )

    # Plot calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label=f"{window_name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed at-risk rate")
    plt.title(f"Calibration Curve - {window_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / f"calibration_curve_{window_name}.png", dpi=300)
    plt.close()

    # Threshold analysis
    threshold_rows = []

    for threshold in THRESHOLDS:
        row = threshold_metrics(y_test.values, y_test_prob, threshold)
        row["window"] = window_name
        row["roc_auc"] = roc_auc
        row["pr_auc"] = pr_auc
        row["brier_score"] = test_brier
        row["ece"] = test_ece
        threshold_rows.append(row)
        all_threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        OUT_TABLE_DIR / f"threshold_analysis_{window_name}.csv",
        index=False
    )

    # Recommended thresholds
    # 1. Best F1
    best_f1_row = threshold_df.sort_values("f1", ascending=False).iloc[0].to_dict()

    # 2. High recall threshold: recall >= 0.80 with best precision
    high_recall_candidates = threshold_df[threshold_df["recall"] >= 0.80].copy()

    if len(high_recall_candidates) > 0:
        high_recall_row = high_recall_candidates.sort_values(
            ["precision", "f1"],
            ascending=[False, False]
        ).iloc[0].to_dict()
    else:
        high_recall_row = threshold_df.sort_values("recall", ascending=False).iloc[0].to_dict()

    summary_rows.append({
        "window": window_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": test_brier,
        "ece": test_ece,
        "best_f1_threshold": best_f1_row["threshold"],
        "best_f1": best_f1_row["f1"],
        "best_f1_precision": best_f1_row["precision"],
        "best_f1_recall": best_f1_row["recall"],
        "best_f1_predicted_at_risk_rate": best_f1_row["predicted_at_risk_rate"],
        "high_recall_threshold": high_recall_row["threshold"],
        "high_recall_precision": high_recall_row["precision"],
        "high_recall_recall": high_recall_row["recall"],
        "high_recall_f1": high_recall_row["f1"],
        "high_recall_predicted_at_risk_rate": high_recall_row["predicted_at_risk_rate"],
    })

    print("\nTHRESHOLD ANALYSIS")
    print(
        threshold_df[
            [
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "predicted_at_risk_rate",
            ]
        ].to_string(index=False)
    )

    print("\nRECOMMENDED THRESHOLDS")
    print(
        f"Best F1 threshold        : {best_f1_row['threshold']:.2f} | "
        f"F1={best_f1_row['f1']:.4f} | "
        f"Precision={best_f1_row['precision']:.4f} | "
        f"Recall={best_f1_row['recall']:.4f}"
    )
    print(
        f"High recall threshold    : {high_recall_row['threshold']:.2f} | "
        f"F1={high_recall_row['f1']:.4f} | "
        f"Precision={high_recall_row['precision']:.4f} | "
        f"Recall={high_recall_row['recall']:.4f}"
    )


combined_calibration = pd.concat(all_calibration_rows, ignore_index=True)
combined_thresholds = pd.DataFrame(all_threshold_rows)
summary_df = pd.DataFrame(summary_rows)

combined_calibration_path = OUT_TABLE_DIR / "calibration_bins_all_windows.csv"
combined_thresholds_path = OUT_TABLE_DIR / "threshold_analysis_all_windows.csv"
summary_path = OUT_TABLE_DIR / "calibration_threshold_summary.csv"

combined_calibration.to_csv(combined_calibration_path, index=False)
combined_thresholds.to_csv(combined_thresholds_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("[DONE] Calibration and threshold analysis completed")
print(f"Calibration bins : {combined_calibration_path}")
print(f"Thresholds       : {combined_thresholds_path}")
print(f"Summary          : {summary_path}")
print("=" * 80)

print("\nCALIBRATION AND THRESHOLD SUMMARY")
print(summary_df.to_string(index=False))
