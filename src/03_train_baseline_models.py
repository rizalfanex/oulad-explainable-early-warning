from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

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
    classification_report,
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "oulad_features_full.csv"
OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables"
OUT_MODEL_DIR = BASE_DIR / "outputs" / "models"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRAIN BASELINE MODELS - OULAD AT-RISK PREDICTION")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Processed data not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded data: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

# ============================================================
# 1. Target
# ============================================================

target_col = "at_risk"
y = df[target_col].astype(int)

# ============================================================
# 2. Drop leakage / identifier / label columns
# ============================================================

drop_cols = [
    # target and original outcome
    "at_risk",
    "final_result",

    # identifiers
    "id_student",

    # post-outcome / direct withdrawal information
    # Important: date_unregistration is strongly related to Withdrawn.
    # We remove it for a fairer early-warning baseline.
    "date_unregistration",
]

# Keep code_module and code_presentation as categorical context features.
# Remove columns only if they exist.
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)

# ============================================================
# 3. Identify numeric and categorical features
# ============================================================

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"[INFO] Numeric features     : {len(numeric_features)}")
print(f"[INFO] Categorical features : {len(categorical_features)}")
print(f"[INFO] Dropped columns      : {drop_cols}")

# ============================================================
# 4. Train/validation/test split
# ============================================================
# Initial baseline: stratified random split.
# Later we will add temporal/early prediction and cross-module validation.

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

print(f"[INFO] Train: {X_train.shape[0]:,}")
print(f"[INFO] Val  : {X_val.shape[0]:,}")
print(f"[INFO] Test : {X_test.shape[0]:,}")

# ============================================================
# 5. Preprocessor
# ============================================================

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

# ============================================================
# 6. Models
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
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
else:
    print("[WARNING] XGBoost not available. Skipping XGBoost.")

# ============================================================
# 7. Evaluation helper
# ============================================================

def evaluate_model(model_name, pipeline, X_split, y_split, split_name):
    y_pred = pipeline.predict(X_split)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_split)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y_split, y_pred),
        "precision": precision_score(y_split, y_pred, zero_division=0),
        "recall": recall_score(y_split, y_pred, zero_division=0),
        "f1": f1_score(y_split, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_split, y_prob),
        "pr_auc": average_precision_score(y_split, y_prob),
    }

    cm = confusion_matrix(y_split, y_pred)
    return metrics, cm


# ============================================================
# 8. Train and evaluate
# ============================================================

all_results = []
all_confusions = []

best_model_name = None
best_val_auc = -1
best_pipeline = None

for model_name, model in models.items():
    print("\n" + "-" * 80)
    print(f"[TRAIN] {model_name}")
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
        metrics, cm = evaluate_model(model_name, pipeline, X_split, y_split, split_name)
        all_results.append(metrics)

        all_confusions.append({
            "model": model_name,
            "split": split_name,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

        print(
            f"{split_name.upper():5s} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

    val_auc = [r for r in all_results if r["model"] == model_name and r["split"] == "val"][0]["roc_auc"]

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_name = model_name
        best_pipeline = pipeline

    # Save each model
    model_path = OUT_MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[SAVED] {model_path}")

# ============================================================
# 9. Save results
# ============================================================

results_df = pd.DataFrame(all_results)
confusion_df = pd.DataFrame(all_confusions)

results_path = OUT_TABLE_DIR / "baseline_model_results.csv"
confusion_path = OUT_TABLE_DIR / "baseline_confusion_matrices.csv"

results_df.to_csv(results_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

print("\n" + "=" * 80)
print("[DONE] Results saved")
print(f"Results   : {results_path}")
print(f"Confusion : {confusion_path}")
print("=" * 80)

print("\nBEST MODEL BASED ON VALIDATION ROC-AUC")
print(f"Best model : {best_model_name}")
print(f"Val ROC-AUC: {best_val_auc:.4f}")

best_path = OUT_MODEL_DIR / "best_baseline_model.joblib"
joblib.dump(best_pipeline, best_path)
print(f"[SAVED] Best model: {best_path}")

# Print test classification report for best model
print("\n" + "=" * 80)
print("BEST MODEL TEST CLASSIFICATION REPORT")
print("=" * 80)

y_test_pred = best_pipeline.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names=["Non-risk", "At-risk"]))
