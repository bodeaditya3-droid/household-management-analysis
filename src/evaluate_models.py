"""Model evaluation, permutation importance, and diagnostic visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_MODEL_FEATURES,
    CLEANED_DATA_PATH,
    CLUSTER_FEATURES,
    CLUSTER_MODEL_PATH,
    EVAL_VIZ_DIR,
    HIGH_WASTE_CLF_MODEL_PATH,
    MODEL_METADATA_PATH,
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

sns.set_theme(style="whitegrid", context="notebook")


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    out_path = EVAL_VIZ_DIR / filename
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_waste_regression_eval(
    comp_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, best_name: str
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Daily Food Waste Regression Evaluation (Best: {best_name})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # 1. Model RMSE comparison
    ax = axes[0]
    ordered = comp_df.sort_values("Test_RMSE", ascending=True)
    colors = ["#2a9d8f" if i == 0 else "#457b9d" for i in range(len(ordered))]
    bars = ax.barh(ordered["Model"], ordered["Test_RMSE"], color=colors)
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{row['Test_RMSE']:.2f}g (R2={row['Test_R2']:.3f})",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlim(0, ordered["Test_RMSE"].max() * 1.45)
    ax.set_title("Test RMSE Comparison (Lower is Better)")
    ax.set_xlabel("RMSE (Grams)")

    # 2. Actual vs Predicted
    ax = axes[1]
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#264653", edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#e76f51", linewidth=2, label="Ideal Fit (y = x)")
    ax.set_title("Actual vs Predicted Daily Waste")
    ax.set_xlabel("Actual Daily Food Waste (g)")
    ax.set_ylabel("Predicted Daily Food Waste (g)")
    ax.legend()

    # 3. Residual Distribution
    ax = axes[2]
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=35, kde=True, color="#e9c46a", ax=ax)
    ax.axvline(0, color="#e76f51", linestyle="--", linewidth=2)
    ax.set_title(
        f"Residuals Distribution (Mean={residuals.mean():.2f}g, SD={residuals.std():.2f}g)"
    )
    ax.set_xlabel("Residual (Actual - Predicted, g)")

    fig.tight_layout()
    return _save_fig(fig, "01_waste_regression_evaluation.png")


def _plot_spend_regression_eval(
    comp_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, best_name: str
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Monthly Grocery Spend Regression Evaluation (Best: {best_name})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    ax = axes[0]
    ordered = comp_df.sort_values("Test_RMSE", ascending=True)
    colors = ["#2a9d8f" if i == 0 else "#457b9d" for i in range(len(ordered))]
    bars = ax.barh(ordered["Model"], ordered["Test_RMSE"], color=colors)
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(
            bar.get_width() + 30,
            bar.get_y() + bar.get_height() / 2,
            f"INR {row['Test_RMSE']:,.0f} (R2={row['Test_R2']:.4f})",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlim(0, ordered["Test_RMSE"].max() * 1.48)
    ax.set_title("Test RMSE Comparison")
    ax.set_xlabel("RMSE (INR)")

    ax = axes[1]
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#2a9d8f", edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#e76f51", linewidth=2, label="Ideal Fit (y = x)")
    ax.set_title("Actual vs Predicted Monthly Grocery Spend")
    ax.set_xlabel("Actual Monthly Spend (INR)")
    ax.set_ylabel("Predicted Monthly Spend (INR)")
    ax.legend()

    fig.tight_layout()
    return _save_fig(fig, "02_spend_regression_evaluation.png")


def _plot_binary_classification_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    roc_df: pd.DataFrame,
    best_name: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(
        f"High-Waste Household Binary Classification Evaluation (Best: {best_name})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # 1. Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["Actual: No", "Actual: Yes"],
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix ({best_name})")

    # 2. Multi-model ROC Curves
    ax = axes[1]
    palette = ["#2a9d8f", "#e76f51", "#264653", "#f4a261"]
    model_cols = [c for c in roc_df.columns if c != "y_true"]
    for idx, model_name in enumerate(model_cols):
        fpr, tpr, _ = roc_curve(roc_df["y_true"], roc_df[model_name])
        roc_auc = auc(fpr, tpr)
        lw = 2.6 if model_name == best_name else 1.6
        ax.plot(
            fpr,
            tpr,
            color=palette[idx % len(palette)],
            linewidth=lw,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.set_title("ROC Curves Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    return _save_fig(fig, "03_high_waste_classification_eval.png")


def _plot_multiclass_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, best_name: str
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=WASTE_LEVEL_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        f"Food Waste Level 4-Class Evaluation (Best: {best_name})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=WASTE_LEVEL_ORDER,
        yticklabels=WASTE_LEVEL_ORDER,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Level")
    axes[0].set_ylabel("Actual Level")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlOrRd",
        xticklabels=WASTE_LEVEL_ORDER,
        yticklabels=WASTE_LEVEL_ORDER,
        ax=axes[1],
    )
    axes[1].set_title("Normalized Recall Matrix (Row %)")
    axes[1].set_xlabel("Predicted Level")
    axes[1].set_ylabel("Actual Level")

    fig.tight_layout()
    return _save_fig(fig, "04_waste_level_confusion_matrix.png")


def _plot_feature_importance(
    waste_imp_df: pd.DataFrame, hw_imp_df: pd.DataFrame
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Top Drivers of Household Food Waste (Permutation Feature Importance on Test Set)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    top_w = waste_imp_df.head(12)
    axes[0].barh(top_w["Feature"], top_w["Importance_Mean"], color="#264653")
    axes[0].invert_yaxis()
    axes[0].set_title("Daily Food Waste Regression (Impact on RMSE, g)")
    axes[0].set_xlabel("Permutation Importance (RMSE Increase in Grams)")

    top_hw = hw_imp_df.head(12)
    axes[1].barh(top_hw["Feature"], top_hw["Importance_Mean"], color="#e76f51")
    axes[1].invert_yaxis()
    axes[1].set_title("High-Waste Risk Classification (Impact on ROC-AUC)")
    axes[1].set_xlabel("Permutation Importance (ROC-AUC Drop)")

    fig.tight_layout()
    return _save_fig(fig, "05_feature_importance.png")


def _plot_household_clusters(
    df: pd.DataFrame,
    cluster_pipe,
    k_metrics_df: pd.DataFrame,
    cluster_meta: dict,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Unsupervised Household Segmentation (K-Means Clustering)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # 1. Elbow + Silhouette Score dual-axis plot
    ax1 = axes[0]
    ax2 = ax1.twinx()
    l1 = ax1.plot(
        k_metrics_df["k"],
        k_metrics_df["inertia"],
        "-o",
        color="#264653",
        linewidth=2.2,
        label="Inertia (Elbow)",
    )
    l2 = ax2.plot(
        k_metrics_df["k"],
        k_metrics_df["silhouette_score"],
        "-s",
        color="#e76f51",
        linewidth=2.2,
        label="Silhouette Score",
    )
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia", color="#264653")
    ax2.set_ylabel("Silhouette Score", color="#e76f51")
    ax1.set_title("K Selection: Elbow Curve & Silhouette Score")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    # 2. PCA 2D projection of clusters
    ax = axes[1]
    X_scaled = cluster_pipe.named_steps["scaler"].transform(df[CLUSTER_FEATURES])
    cluster_labels = cluster_pipe.named_steps["kmeans"].predict(X_scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Segment": [cluster_meta[str(c)]["name"] for c in cluster_labels],
        }
    ).sample(n=min(3000, len(df)), random_state=RANDOM_STATE)

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Segment",
        palette="Set2",
        alpha=0.55,
        s=22,
        edgecolor="none",
        ax=ax,
    )
    var_pct = pca.explained_variance_ratio_ * 100
    ax.set_title(
        f"PCA 2D Projection of Household Segments ({var_pct[0]:.1f}% + {var_pct[1]:.1f}% var)"
    )
    ax.legend(title="Segment", fontsize=8)

    fig.tight_layout()
    return _save_fig(fig, "06_household_clusters.png")


def evaluate_all_models(cleaned_path: Path = CLEANED_DATA_PATH) -> dict:
    """Evaluate persisted best models on held-out test split and save plots/reports."""
    ensure_directories()
    df = pd.read_csv(cleaned_path)

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Reconstruct the exact same test split used in training
    X = df[ALL_MODEL_FEATURES].copy()
    y_waste = df[TARGET_WASTE_REG].copy()
    y_spend = df[TARGET_SPEND_REG].copy()
    y_level = df[TARGET_WASTE_LEVEL].copy()
    y_high = (df[TARGET_HIGH_WASTE] == "Yes").astype(int).to_numpy()

    indices = np.arange(len(df))
    _, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_level,
    )

    X_test = X.iloc[test_idx]
    y_waste_test = y_waste.iloc[test_idx].to_numpy()
    y_spend_test = y_spend.iloc[test_idx].to_numpy()
    y_level_test = y_level.iloc[test_idx]
    y_high_test = y_high[test_idx]

    # Load persisted best pipelines
    waste_model = joblib.load(WASTE_REG_MODEL_PATH)
    spend_model = joblib.load(SPEND_REG_MODEL_PATH)
    hw_model = joblib.load(HIGH_WASTE_CLF_MODEL_PATH)
    lvl_model = joblib.load(WASTE_LEVEL_CLF_MODEL_PATH)
    cluster_model = joblib.load(CLUSTER_MODEL_PATH)

    # Generate test predictions
    waste_pred = waste_model.predict(X_test)
    spend_pred = spend_model.predict(X_test)
    hw_pred = hw_model.predict(X_test)
    lvl_pred = lvl_model.predict(X_test)

    # Permutation feature importance on raw + engineered columns (model-agnostic & leak-free)
    waste_perm = permutation_importance(
        waste_model,
        X_test,
        y_waste_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    waste_imp_df = (
        pd.DataFrame(
            {
                "Feature": ALL_MODEL_FEATURES,
                "Importance_Mean": np.round(waste_perm.importances_mean, 4),
                "Importance_Std": np.round(waste_perm.importances_std, 4),
            }
        )
        .sort_values("Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )
    waste_imp_df.to_csv(REPORTS_DIR / "waste_feature_importance.csv", index=False)

    hw_perm = permutation_importance(
        hw_model,
        X_test,
        y_high_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    hw_imp_df = (
        pd.DataFrame(
            {
                "Feature": ALL_MODEL_FEATURES,
                "Importance_Mean": np.round(hw_perm.importances_mean, 4),
                "Importance_Std": np.round(hw_perm.importances_std, 4),
            }
        )
        .sort_values("Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )
    hw_imp_df.to_csv(REPORTS_DIR / "high_waste_feature_importance.csv", index=False)

    # Classification reports
    binary_report = classification_report(
        y_high_test, hw_pred, target_names=["No", "Yes"], output_dict=True
    )
    multiclass_report = classification_report(
        y_level_test, lvl_pred, labels=WASTE_LEVEL_ORDER, output_dict=True
    )

    # Load comparison tables for plotting
    waste_comp = pd.read_csv(REPORTS_DIR / "waste_regression_comparison.csv")
    spend_comp = pd.read_csv(REPORTS_DIR / "spend_regression_comparison.csv")
    roc_df = pd.read_csv(REPORTS_DIR / "binary_test_probabilities.csv")
    k_metrics = pd.read_csv(REPORTS_DIR / "kmeans_metrics.csv")

    charts = [
        _plot_waste_regression_eval(
            waste_comp,
            y_waste_test,
            waste_pred,
            metadata["best_models"]["waste_regression"]["name"],
        ),
        _plot_spend_regression_eval(
            spend_comp,
            y_spend_test,
            spend_pred,
            metadata["best_models"]["spend_regression"]["name"],
        ),
        _plot_binary_classification_eval(
            y_high_test,
            hw_pred,
            roc_df,
            metadata["best_models"]["high_waste_classification"]["name"],
        ),
        _plot_multiclass_confusion_matrix(
            y_level_test,
            lvl_pred,
            metadata["best_models"]["waste_level_classification"]["name"],
        ),
        _plot_feature_importance(waste_imp_df, hw_imp_df),
        _plot_household_clusters(
            df,
            cluster_model,
            k_metrics,
            metadata["best_models"]["household_clustering"]["segments"],
        ),
    ]

    eval_summary = {
        "best_models": metadata["best_models"],
        "top_5_waste_drivers": waste_imp_df.head(5).to_dict(orient="records"),
        "top_5_high_waste_drivers": hw_imp_df.head(5).to_dict(orient="records"),
        "binary_classification_report": binary_report,
        "multiclass_classification_report": multiclass_report,
        "generated_charts": [str(p) for p in charts],
    }

    with open(REPORTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


if __name__ == "__main__":
    summary = evaluate_all_models()
    print(
        f"Evaluation complete: {len(summary['generated_charts'])} charts saved to {EVAL_VIZ_DIR}"
    )
