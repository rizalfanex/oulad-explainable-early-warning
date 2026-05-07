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

from xgboost import XGBClassifier
import xgboost as xgb
import shap


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

OUT_FIG_DIR = BASE_DIR / "outputs" / "figures" / "shap"
OUT_TABLE_DIR = BASE_DIR / "outputs" / "tables" / "shap"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SHAP EXPLAINABILITY - XGBOOST ALL FEATURES")
print("=" * 80)

datasets = {
    "day14": DATA_DIR / "oulad_features_day14.csv",
    "day28": DATA_DIR / "oulad_features_day28.csv",
    "day56": DATA_DIR / "oulad_features_day56.csv",
    "day84": DATA_DIR / "oulad_features_day84.csv",
    "full": DATA_DIR / "oulad_features_full.csv",
}

DROP_COLS = [
    "at_risk",
    "final_result",
    "id_student",
    "date_unregistration",
]

TARGET_COL = "at_risk"

# SHAP can be slow. Use a representative sample from the test set.
SHAP_SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

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

    return preprocessor, numeric_features, categorical_features


def clean_feature_name(name):
    """
    Convert ColumnTransformer feature names into cleaner names.
    Examples:
    num__total_clicks -> total_clicks
    cat__gender_M -> gender
    cat__region_Scotland -> region
    """
    name = str(name)

    if name.startswith("num__"):
        return name.replace("num__", "")

    if name.startswith("cat__"):
        tmp = name.replace("cat__", "")
        # OneHotEncoder names are usually: column_value
        # We map back to original column by checking known prefixes later.
        return tmp

    return name


def map_ohe_to_raw_feature(feature_name, original_categorical_cols):
    """
    Aggregate one-hot encoded feature names back to their original raw feature.
    """
    f = str(feature_name)

    if f.startswith("num__"):
        return f.replace("num__", "")

    if f.startswith("cat__"):
        tmp = f.replace("cat__", "")
        for col in original_categorical_cols:
            prefix = f"{col}_"
            if tmp.startswith(prefix):
                return col
        return tmp

    return f


def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        names = []
        names.extend([f"num__{c}" for c in numeric_features])

        cat_pipe = preprocessor.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]

        try:
            cat_names = onehot.get_feature_names_out(categorical_features)
            names.extend([f"cat__{c}" for c in cat_names])
        except Exception:
            names.extend([f"cat__{c}" for c in categorical_features])

        return names


all_raw_importance = []

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

    print(f"[INFO] Dropped columns: {drop_cols}")
    print(f"[INFO] Raw features: {X.shape[1]}")

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

    preprocessor, numeric_features, categorical_features = make_preprocessor(X_train)

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

    print("[INFO] Fitting preprocessor...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_transformed_feature_names(
        preprocessor,
        numeric_features,
        categorical_features
    )

    print(f"[INFO] Transformed feature count: {len(feature_names)}")

    print("[INFO] Training XGBoost...")
    model.fit(X_train_trans, y_train)

    # Sample test data for SHAP
    n_sample = min(SHAP_SAMPLE_SIZE, X_test_trans.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(X_test_trans.shape[0], size=n_sample, replace=False)

    X_shap = X_test_trans[sample_idx]
    y_shap = y_test.iloc[sample_idx].reset_index(drop=True)

    print(f"[INFO] SHAP sample size: {n_sample}")

    print("[INFO] Computing SHAP values using XGBoost pred_contribs...")
    booster = model.get_booster()

    # Do not pass feature_names to DMatrix because XGBoost rejects names
    # containing characters such as [, ], or < from one-hot encoded categories.
    # Feature order remains aligned with feature_names because X_shap is produced
    # by the same fitted preprocessor.
    dmatrix = xgb.DMatrix(X_shap)
    shap_contribs = booster.predict(dmatrix, pred_contribs=True)

    # XGBoost pred_contribs returns one extra bias/base-value column at the end.
    shap_values = shap_contribs[:, :-1]
    shap_values = np.array(shap_values)

    # ========================================================
    # 1. Transformed feature importance
    # ========================================================

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    transformed_importance = pd.DataFrame({
        "window": window_name,
        "feature_transformed": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    transformed_path = OUT_TABLE_DIR / f"shap_transformed_importance_{window_name}.csv"
    transformed_importance.to_csv(transformed_path, index=False)

    print(f"[DONE] Saved transformed SHAP importance: {transformed_path}")

    # ========================================================
    # 2. Raw-level feature importance
    # ========================================================

    raw_names = [
        map_ohe_to_raw_feature(f, categorical_features)
        for f in feature_names
    ]

    raw_importance = pd.DataFrame({
        "window": window_name,
        "feature_raw": raw_names,
        "mean_abs_shap": mean_abs_shap,
    })

    raw_importance = (
        raw_importance
        .groupby(["window", "feature_raw"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )

    raw_path = OUT_TABLE_DIR / f"shap_raw_importance_{window_name}.csv"
    raw_importance.to_csv(raw_path, index=False)

    all_raw_importance.append(raw_importance)

    print(f"[DONE] Saved raw SHAP importance: {raw_path}")

    print("\nTOP 15 RAW FEATURES")
    print(raw_importance.head(15).to_string(index=False))

    # ========================================================
    # 3. SHAP bar plot
    # ========================================================

    top_n = 20
    top_plot = raw_importance.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, 7))
    plt.barh(top_plot["feature_raw"], top_plot["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} SHAP Features - {window_name}")
    plt.tight_layout()

    bar_path = OUT_FIG_DIR / f"shap_bar_{window_name}.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()

    print(f"[DONE] Saved SHAP bar plot: {bar_path}")

    # ========================================================
    # 4. SHAP beeswarm summary plot for transformed features
    # ========================================================
    # Limit to top transformed features to keep the figure readable.

    top_transformed_features = transformed_importance.head(25)["feature_transformed"].tolist()
    top_indices = [feature_names.index(f) for f in top_transformed_features]

    X_shap_top = X_shap[:, top_indices]
    shap_top = shap_values[:, top_indices]
    top_feature_names_clean = [clean_feature_name(feature_names[i]) for i in top_indices]

    plt.figure()
    shap.summary_plot(
        shap_top,
        X_shap_top,
        feature_names=top_feature_names_clean,
        show=False,
        max_display=25
    )
    plt.tight_layout()

    beeswarm_path = OUT_FIG_DIR / f"shap_beeswarm_{window_name}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Saved SHAP beeswarm plot: {beeswarm_path}")

# ============================================================
# Save combined raw importance
# ============================================================

if len(all_raw_importance) > 0:
    combined = pd.concat(all_raw_importance, ignore_index=True)
    combined_path = OUT_TABLE_DIR / "shap_raw_importance_all_windows.csv"
    combined.to_csv(combined_path, index=False)

    top10 = (
        combined
        .sort_values(["window", "mean_abs_shap"], ascending=[True, False])
        .groupby("window")
        .head(10)
    )

    top10_path = OUT_TABLE_DIR / "shap_top10_raw_features_all_windows.csv"
    top10.to_csv(top10_path, index=False)

    print("\n" + "=" * 80)
    print("[DONE] Combined SHAP outputs saved")
    print(f"Combined raw importance : {combined_path}")
    print(f"Top 10 by window        : {top10_path}")
    print("=" * 80)

    print("\nTOP 10 RAW FEATURES BY WINDOW")
    print(top10.to_string(index=False))

print("\n[DONE] SHAP explainability completed.")
