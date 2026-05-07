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
print("EARLY PREDICTION TRAINING - OULAD")
print("=" * 80)

datasets = {
    "day14": DATA_DIR / "oulad_features_day14.csv",
    "day28": DATA_DIR / "oulad_features_day28.csv",
    "day56": DATA_DIR / "oulad_features_day56.csv",
    "day84": DATA_DIR / "oulad_features_day84.csv",
    "full": DATA_DIR / "oulad_features_full.csv",
}

drop_base = [
    "at_risk",
    "final_result",
    "id_student",
    "date_unregistration",
]

all_results = []
all_confusions = []

def build_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
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

    return models


def evaluate(model_name, window_name, pipeline, X_split, y_split, split_name):
    y_pred = pipeline.predict(X_split)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_split)[:, 1]
    else:
        y_prob = y_pred.astype(float)

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


for window_name, data_path in datasets.items():
    print("\n" + "=" * 80)
    print(f"[WINDOW] {window_name}")
    print("=" * 80)

    if not data_path.exists():
        print(f"[SKIP] File not found: {data_path}")
        continue

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {data_path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    y = df["at_risk"].astype(int)

    drop_cols = [c for c in drop_base if c in df.columns]
    X = df.drop(columns=drop_cols)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] Numeric features     : {len(numeric_features)}")
    print(f"[INFO] Categorical features : {len(categorical_features)}")
    print(f"[INFO] Dropped columns      : {drop_cols}")

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

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = build_models()

    for model_name, model in models.items():
        print("\n" + "-" * 80)
        print(f"[TRAIN] {window_name} | {model_name}")
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
out_path = OUT_TABLE_DIR / "early_prediction_results.csv"
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 80)
print(f"[DONE] Saved early prediction results to: {out_path}")
print("=" * 80)

test_results = results_df[results_df["split"] == "test"].copy()

summary = (
    test_results.sort_values(["window", "roc_auc"], ascending=[True, False])
    .groupby("window")
    .head(1)
    .sort_values("window")
)

summary_path = OUT_TABLE_DIR / "early_prediction_best_by_window.csv"
summary.to_csv(summary_path, index=False)

print("\nBEST TEST MODEL BY WINDOW")
print(summary[["window", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]])

print(f"\n[DONE] Saved best-by-window summary to: {summary_path}")
