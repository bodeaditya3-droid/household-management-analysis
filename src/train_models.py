"""Multi-task Machine Learning training pipeline for Smart Household Food Waste ML."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CLEANED_DATA_PATH,
    CLUSTER_FEATURES,
    CLUSTER_MODEL_PATH,
    HIGH_WASTE_CLF_MODEL_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    SPEND_REG_MODEL_PATH,
    TARGET_HIGH_WASTE,
    TARGET_SPEND_REG,
    TARGET_WASTE_LEVEL,
    TARGET_WASTE_REG,
    TEST_SIZE,
    WASTE_LEVEL_CLF_MODEL_PATH,
    WASTE_LEVEL_ORDER,
    WASTE_REG_MODEL_PATH,
    ensure_directories,
)


def build_preprocessor() -> ColumnTransformer:
    """Create a leak-free preprocessing ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ALL_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _train_regression_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    candidates: Dict[str, object],
) -> Tuple[str, Pipeline, pd.DataFrame]:
    """Train, cross-validate, and select the best regression pipeline."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    trained: Dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        cv_scores = -cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(root_mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mape = float(mean_absolute_percentage_error(y_test, preds) * 100.0)

        rows.append(
            {
                "Model": name,
                "CV_RMSE_Mean": round(float(cv_scores.mean()), 3),
                "CV_RMSE_Std": round(float(cv_scores.std()), 3),
                "Test_RMSE": round(rmse, 3),
                "Test_MAE": round(mae, 3),
                "Test_R2": round(r2, 4),
                "Test_MAPE_Pct": round(mape, 2),
            }
        )
        trained[name] = pipe

    results_df = pd.DataFrame(rows).sort_values("Test_RMSE", ascending=True).reset_index(drop=True)
    best_name = str(results_df.loc[0, "Model"])
    return best_name, trained[best_name], results_df


def _train_binary_classification_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[str, Pipeline, pd.DataFrame, Dict[str, np.ndarray]]:
    """Train and compare binary classifiers for High_Waste_Household."""
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    trained: Dict[str, Pipeline] = {}
    test_probas: Dict[str, np.ndarray] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        cv_auc = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        test_probas[name] = proba

        rows.append(
            {
                "Model": name,
                "CV_ROC_AUC_Mean": round(float(cv_auc.mean()), 4),
                "Test_Accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "Test_Precision": round(float(precision_score(y_test, preds)), 4),
                "Test_Recall": round(float(recall_score(y_test, preds)), 4),
                "Test_F1": round(float(f1_score(y_test, preds)), 4),
                "Test_ROC_AUC": round(float(roc_auc_score(y_test, proba)), 4),
            }
        )
        trained[name] = pipe

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["Test_ROC_AUC", "Test_F1"], ascending=False)
        .reset_index(drop=True)
    )
    best_name = str(results_df.loc[0, "Model"])
    return best_name, trained[best_name], results_df, test_probas


def _train_multiclass_classification_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[str, Pipeline, pd.DataFrame]:
    """Train and compare multiclass classifiers for Food_Waste_Level."""
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    trained: Dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        cv_f1 = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rows.append(
            {
                "Model": name,
                "CV_Macro_F1_Mean": round(float(cv_f1.mean()), 4),
                "Test_Accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "Test_Macro_F1": round(
                    float(f1_score(y_test, preds, average="macro")), 4
                ),
                "Test_Weighted_F1": round(
                    float(f1_score(y_test, preds, average="weighted")), 4
                ),
            }
        )
        trained[name] = pipe

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["Test_Macro_F1", "Test_Accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    best_name = str(results_df.loc[0, "Model"])
    return best_name, trained[best_name], results_df


def _train_household_clustering(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict]:
    """Fit K-Means clustering on standardized household features and build personas."""
    X_cluster = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    k_metrics = []
    best_k = 3
    best_sil = -1.0

    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = float(
            silhouette_score(X_scaled, labels, sample_size=3000, random_state=RANDOM_STATE)
        )
        k_metrics.append(
            {
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette_score": round(sil, 4),
            }
        )
        if sil > best_sil:
            best_sil = sil
            best_k = k

    k_metrics_df = pd.DataFrame(k_metrics)

    cluster_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)),
        ]
    )
    labels = cluster_pipe.fit_predict(X_cluster)

    profile_df = df.copy()
    profile_df["Cluster"] = labels
    summary = (
        profile_df.groupby("Cluster")
        .agg(
            Households=("Household_ID", "count"),
            Avg_Household_Size=("Household_Size", "mean"),
            Avg_Income_INR=("Monthly_Household_Income_INR", "mean"),
            Avg_Weekly_Spend_INR=("Weekly_Grocery_Spend_INR", "mean"),
            Avg_Habit_Score=("Habit_Score", "mean"),
            Avg_Storage_Risk=("Storage_Risk_Index", "mean"),
            Avg_Daily_Waste_Grams=(TARGET_WASTE_REG, "mean"),
            High_Waste_Rate_Pct=(
                TARGET_HIGH_WASTE,
                lambda s: (s == "Yes").mean() * 100.0,
            ),
        )
        .round(2)
        .reset_index()
        .sort_values("Avg_Daily_Waste_Grams", ascending=True)
        .reset_index(drop=True)
    )

    # Assign intuitive persona names ordered from lowest to highest average daily waste
    persona_templates = {
        2: ["Low-Waste Mindful Households", "High-Consumption High-Waste Households"],
        3: [
            "Low-Waste Efficient Households",
            "Moderate-Spend Balanced Households",
            "Large High-Spend / High-Waste Households",
        ],
        4: [
            "Small Low-Waste Households",
            "Habit-Conscious Mid-Size Households",
            "Busy Convenience-Driven Households",
            "Large High-Spend / High-Waste Households",
        ],
        5: [
            "Minimal-Waste Savers",
            "Disciplined Small Households",
            "Balanced Mid-Size Households",
            "Convenience-Heavy Households",
            "Large High-Waste Households",
        ],
        6: [
            "Minimal-Waste Savers",
            "Disciplined Small Households",
            "Moderate Budget Households",
            "Affluent Mindful Households",
            "Busy Convenience Households",
            "Large High-Waste Households",
        ],
    }
    names = persona_templates.get(best_k, [f"Segment {i + 1}" for i in range(best_k)])
    summary["Segment_Name"] = names

    cluster_name_map = {
        int(row["Cluster"]): {
            "name": str(row["Segment_Name"]),
            "avg_daily_waste_g": float(row["Avg_Daily_Waste_Grams"]),
            "avg_household_size": float(row["Avg_Household_Size"]),
            "avg_income_inr": float(row["Avg_Income_INR"]),
            "high_waste_rate_pct": float(row["High_Waste_Rate_Pct"]),
        }
        for _, row in summary.iterrows()
    }

    return cluster_pipe, k_metrics_df, summary, cluster_name_map


def train_all_models(cleaned_path: Path = CLEANED_DATA_PATH) -> dict:
    """Train, select, and serialize all ML models plus training metadata."""
    ensure_directories()
    df = pd.read_csv(cleaned_path)

    X = df[ALL_MODEL_FEATURES].copy()
    y_waste = df[TARGET_WASTE_REG].copy()
    y_spend = df[TARGET_SPEND_REG].copy()
    y_level = df[TARGET_WASTE_LEVEL].copy()
    y_high = (df[TARGET_HIGH_WASTE] == "Yes").astype(int).to_numpy()

    # Single stratified train/test index split shared across all supervised tasks
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_level,
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    # 1. Daily Food Waste Regression
    waste_candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }
    best_waste_name, best_waste_pipe, waste_comp_df = _train_regression_task(
        X_train, X_test, y_waste.iloc[train_idx], y_waste.iloc[test_idx], waste_candidates
    )

    # 2. Monthly Grocery Spend Regression
    spend_candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }
    best_spend_name, best_spend_pipe, spend_comp_df = _train_regression_task(
        X_train, X_test, y_spend.iloc[train_idx], y_spend.iloc[test_idx], spend_candidates
    )

    # 3. High Waste Binary Classification
    best_hw_name, best_hw_pipe, hw_comp_df, hw_test_probas = _train_binary_classification_task(
        X_train, X_test, y_high[train_idx], y_high[test_idx]
    )

    # 4. Food Waste Level Multiclass Classification
    best_lvl_name, best_lvl_pipe, lvl_comp_df = _train_multiclass_classification_task(
        X_train, X_test, y_level.iloc[train_idx], y_level.iloc[test_idx]
    )

    # 5. Household K-Means Clustering
    cluster_pipe, k_metrics_df, cluster_profiles_df, cluster_map = _train_household_clustering(df)

    # Persist best models strictly inside models/ with lossless compression
    joblib.dump(best_waste_pipe, WASTE_REG_MODEL_PATH, compress=3)
    joblib.dump(best_spend_pipe, SPEND_REG_MODEL_PATH, compress=3)
    joblib.dump(best_hw_pipe, HIGH_WASTE_CLF_MODEL_PATH, compress=3)
    joblib.dump(best_lvl_pipe, WASTE_LEVEL_CLF_MODEL_PATH, compress=3)
    joblib.dump(cluster_pipe, CLUSTER_MODEL_PATH, compress=3)

    # Save model comparison tables strictly inside reports/
    waste_comp_df.to_csv(REPORTS_DIR / "waste_regression_comparison.csv", index=False)
    spend_comp_df.to_csv(REPORTS_DIR / "spend_regression_comparison.csv", index=False)
    hw_comp_df.to_csv(REPORTS_DIR / "high_waste_classification_comparison.csv", index=False)
    lvl_comp_df.to_csv(REPORTS_DIR / "waste_level_classification_comparison.csv", index=False)
    k_metrics_df.to_csv(REPORTS_DIR / "kmeans_metrics.csv", index=False)
    cluster_profiles_df.to_csv(REPORTS_DIR / "kmeans_cluster_profiles.csv", index=False)

    # Save binary classifier test probabilities for multi-model ROC visualization
    roc_df = pd.DataFrame({"y_true": y_high[test_idx]})
    for model_name, probs in hw_test_probas.items():
        roc_df[model_name] = np.round(probs, 6)
    roc_df.to_csv(REPORTS_DIR / "binary_test_probabilities.csv", index=False)

    # Persist metadata for evaluation & Streamlit inference (portable relative paths)
    def _rel(path: Path) -> str:
        return path.relative_to(PROJECT_ROOT).as_posix()

    metadata = {
        "training_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "numerical_features": ALL_NUMERICAL_FEATURES,
        "categorical_features": ALL_CATEGORICAL_FEATURES,
        "cluster_features": CLUSTER_FEATURES,
        "waste_level_order": WASTE_LEVEL_ORDER,
        "best_models": {
            "waste_regression": {
                "name": best_waste_name,
                "metrics": waste_comp_df.iloc[0].to_dict(),
                "path": _rel(WASTE_REG_MODEL_PATH),
            },
            "spend_regression": {
                "name": best_spend_name,
                "metrics": spend_comp_df.iloc[0].to_dict(),
                "path": _rel(SPEND_REG_MODEL_PATH),
            },
            "high_waste_classification": {
                "name": best_hw_name,
                "metrics": hw_comp_df.iloc[0].to_dict(),
                "path": _rel(HIGH_WASTE_CLF_MODEL_PATH),
            },
            "waste_level_classification": {
                "name": best_lvl_name,
                "metrics": lvl_comp_df.iloc[0].to_dict(),
                "path": _rel(WASTE_LEVEL_CLF_MODEL_PATH),
            },
            "household_clustering": {
                "optimal_k": int(cluster_pipe.named_steps["kmeans"].n_clusters),
                "best_silhouette": float(k_metrics_df["silhouette_score"].max()),
                "segments": cluster_map,
                "path": _rel(CLUSTER_MODEL_PATH),
            },
        },
        "population_benchmarks": {
            "mean_daily_waste_g": round(float(df[TARGET_WASTE_REG].mean()), 2),
            "median_daily_waste_g": round(float(df[TARGET_WASTE_REG].median()), 2),
            "p25_daily_waste_g": round(float(df[TARGET_WASTE_REG].quantile(0.25)), 2),
            "p75_daily_waste_g": round(float(df[TARGET_WASTE_REG].quantile(0.75)), 2),
            "mean_monthly_spend_inr": round(float(df[TARGET_SPEND_REG].mean()), 2),
        },
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    meta = train_all_models()
    print("Training complete! Best models selected:")
    for task, info in meta["best_models"].items():
        label = info.get("name", f"K={info.get('optimal_k')}")
        print(f"  - {task}: {label}")
