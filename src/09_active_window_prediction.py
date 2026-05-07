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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

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
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "anonymisedData"
OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ACTIVE-AT-WINDOW EARLY PREDICTION - OULAD")
print("=" * 80)

registration_path = RAW_DATA_DIR / "studentRegistration.csv"
if not registration_path.exists():
    raise FileNotFoundError(f"studentRegistration.csv not found: {registration_path}")

student_registration = pd.read_csv(registration_path)
registration_cols = ["code_module", "code_presentation", "id_student", "date_unregistration"]
student_registration = student_registration[registration_cols].copy()

print(f"[OK] Loaded registration data for active-window filtering: {student_registration.shape}")

datasets = {
    "day14": {
        "path": DATA_DIR / "oulad_features_day14.csv",
        "window_day": 14,
    },
    "day28": {
        "path": DATA_DIR / "oulad_features_day28.csv",
        "window_day": 28,
    },
    "day56": {
        "path": DATA_DIR / "oulad_features_day56.csv",
        "window_day": 56,
    },
    "day84": {
        "path": DATA_DIR / "oulad_features_day84.csv",
        "window_day": 84,
    },
}

TARGET_COL = "at_risk"

DROP_COLS = [
    "at_risk",
    "final_result",
    "id_student",
    "date_unregistration",  # direct withdrawal information, removed from predictors
]

RANDOM_STATE = 42


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
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
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
            n_estimators=500,
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


def evaluate(model_name, window_name, pipeline, X_split, y_split, split_name):
    y_pred = pipeline.predict(X_split)
    y_prob = pipeline.predict_proba(X_split)[:, 1]

    cm = confusion_matrix(y_split, y_pred)

    return {
        "window": window_name,
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y_split, y_pred),
        "precision": precision_score(y_split, y_pred, zero_division=0),
        "recall": recall_score(y_split, y_pred, zero_division=0),
        "f1": f1_score(y_split, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_split, y_prob),
        "pr_auc": average_precision_score(y_split, y_prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


all_results = []
window_summary_rows = []

for window_name, config in datasets.items():
    data_path = config["path"]
    window_day = config["window_day"]

    print("\n" + "=" * 80)
    print(f"[WINDOW] {window_name} | Active at day {window_day}")
    print("=" * 80)

    if not data_path.exists():
        print(f"[SKIP] Missing file: {data_path}")
        continue

    df = pd.read_csv(data_path)
    original_n = len(df)

    # Merge original date_unregistration from raw registration data.
    # The early feature tables intentionally do not contain date_unregistration
    # because it is a direct withdrawal-related variable and must not be used
    # as a predictor. Here, it is used only to filter students who were still
    # active at the observation window.
    if "date_unregistration" in df.columns:
        df = df.drop(columns=["date_unregistration"])

    df = df.merge(
        student_registration,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    if "date_unregistration" not in df.columns:
        raise ValueError("Failed to merge date_unregistration from studentRegistration.csv.")

    # Active-at-window definition:
    # Keep students who have not withdrawn by the observation window.
    # In original OULAD, missing date_unregistration indicates not withdrawn.
    active_mask = df["date_unregistration"].isna() | (df["date_unregistration"] > window_day)
    df_active = df[active_mask].copy()

    removed_n = original_n - len(df_active)

    print(f"[INFO] Original rows       : {original_n:,}")
    print(f"[INFO] Active rows kept    : {len(df_active):,}")
    print(f"[INFO] Removed rows        : {removed_n:,}")
    print(f"[INFO] Removed percentage  : {removed_n / original_n * 100:.2f}%")

    print("[INFO] Target distribution after active-window filtering:")
    print(df_active["at_risk"].value_counts())
    print((df_active["at_risk"].value_counts(normalize=True) * 100).round(2))

    window_summary_rows.append({
        "window": window_name,
        "window_day": window_day,
        "original_rows": original_n,
        "active_rows_kept": len(df_active),
        "removed_rows": removed_n,
        "removed_percentage": removed_n / original_n * 100,
        "at_risk_count": int(df_active["at_risk"].sum()),
        "non_risk_count": int((df_active["at_risk"] == 0).sum()),
        "at_risk_rate": float(df_active["at_risk"].mean()),
    })

    y = df_active[TARGET_COL].astype(int)

    drop_cols = [c for c in DROP_COLS if c in df_active.columns]
    X = df_active.drop(columns=drop_cols)

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

    print(f"[INFO] Train: {X_train.shape[0]:,}")
    print(f"[INFO] Val  : {X_val.shape[0]:,}")
    print(f"[INFO] Test : {X_test.shape[0]:,}")

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    print(f"[INFO] Numeric features     : {len(numeric_features)}")
    print(f"[INFO] Categorical features : {len(categorical_features)}")
    print(f"[INFO] Dropped columns      : {drop_cols}")

    models = build_models()

    for model_name, model in models.items():
        print("\n" + "-" * 80)
        print(f"[TRAIN] {window_name} | active-window | {model_name}")
        print("-" * 80)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            result = evaluate(
                model_name=model_name,
                window_name=window_name,
                pipeline=pipeline,
                X_split=X_split,
                y_split=y_split,
                split_name=split_name,
            )

            result["setting"] = "active_at_window"
            result["window_day"] = window_day
            result["n_active_total"] = len(df_active)
            result["n_removed_pre_window_withdrawn"] = removed_n

            all_results.append(result)

            print(
                f"{split_name.upper():5s} | "
                f"Acc={result['accuracy']:.4f} | "
                f"Prec={result['precision']:.4f} | "
                f"Recall={result['recall']:.4f} | "
                f"F1={result['f1']:.4f} | "
                f"ROC-AUC={result['roc_auc']:.4f} | "
                f"PR-AUC={result['pr_auc']:.4f}"
            )


results_df = pd.DataFrame(all_results)
summary_df = pd.DataFrame(window_summary_rows)

results_path = OUT_TABLE_DIR / "active_window_prediction_results.csv"
summary_path = OUT_TABLE_DIR / "active_window_dataset_summary.csv"

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("[DONE] Active-window prediction completed")
print(f"Results summary : {results_path}")
print(f"Dataset summary : {summary_path}")
print("=" * 80)

test_results = results_df[results_df["split"] == "test"].copy()

best_by_window = (
    test_results
    .sort_values(["window", "roc_auc"], ascending=[True, False])
    .groupby("window")
    .head(1)
    .reset_index(drop=True)
)

best_path = OUT_TABLE_DIR / "active_window_best_by_window.csv"
best_by_window.to_csv(best_path, index=False)

print("\nBEST ACTIVE-WINDOW TEST MODEL BY WINDOW")
print(
    best_by_window[
        [
            "window",
            "window_day",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "n_active_total",
            "n_removed_pre_window_withdrawn",
        ]
    ].to_string(index=False)
)

print(f"\n[DONE] Best active-window results saved to: {best_path}")
