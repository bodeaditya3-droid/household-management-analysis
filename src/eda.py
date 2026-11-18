"""Exploratory Data Analysis (EDA) statistics and visualization generator."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    ALL_NUMERICAL_FEATURES,
    CLEANED_DATA_PATH,
    EDA_VIZ_DIR,
    RAW_CATEGORICAL_FEATURES,
    REPORTS_DIR,
    TARGET_HIGH_WASTE,
    TARGET_SPEND_REG,
    TARGET_WASTE_LEVEL,
    TARGET_WASTE_REG,
    WASTE_LEVEL_ORDER,
    ensure_directories,
)

# Consistent visual styling
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {
    "Low": "#2a9d8f",
    "Moderate": "#e9c46a",
    "High": "#f4a261",
    "Very_High": "#e76f51",
    "No": "#2a9d8f",
    "Yes": "#e76f51",
}


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    out_path = EDA_VIZ_DIR / filename
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_target_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Target Variable Distributions (N = 10,000 Households)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # 1. Daily Food Waste Grams
    ax = axes[0, 0]
    sns.histplot(
        df[TARGET_WASTE_REG], bins=35, kde=True, color="#264653", ax=ax
    )
    mean_w = df[TARGET_WASTE_REG].mean()
    med_w = df[TARGET_WASTE_REG].median()
    ax.axvline(mean_w, color="#e76f51", linestyle="--", linewidth=2, label=f"Mean: {mean_w:.1f}g")
    ax.axvline(med_w, color="#2a9d8f", linestyle=":", linewidth=2, label=f"Median: {med_w:.1f}g")
    ax.set_title("Daily Food Waste (Grams)")
    ax.set_xlabel("Daily Food Waste (g)")
    ax.legend()

    # 2. Estimated Monthly Grocery Spend
    ax = axes[0, 1]
    sns.histplot(
        df[TARGET_SPEND_REG], bins=35, kde=True, color="#2a9d8f", ax=ax
    )
    mean_s = df[TARGET_SPEND_REG].mean()
    ax.axvline(mean_s, color="#e76f51", linestyle="--", linewidth=2, label=f"Mean: INR {mean_s:,.0f}")
    ax.set_title("Estimated Monthly Grocery Spend (INR)")
    ax.set_xlabel("Monthly Grocery Spend (INR)")
    ax.legend()

    # 3. Food Waste Level counts
    ax = axes[1, 0]
    level_counts = df[TARGET_WASTE_LEVEL].value_counts().reindex(WASTE_LEVEL_ORDER)
    bars = ax.bar(
        level_counts.index,
        level_counts.values,
        color=[PALETTE[k] for k in level_counts.index],
        edgecolor="#222222",
        linewidth=0.6,
    )
    for bar, val in zip(bars, level_counts.values):
        pct = val / len(df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 80,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, level_counts.max() * 1.18)
    ax.set_title("Food Waste Level Distribution")
    ax.set_ylabel("Households")

    # 4. High Waste Household binary balance
    ax = axes[1, 1]
    hw_counts = df[TARGET_HIGH_WASTE].value_counts().reindex(["No", "Yes"])
    bars = ax.bar(
        hw_counts.index,
        hw_counts.values,
        color=[PALETTE["No"], PALETTE["Yes"]],
        width=0.5,
        edgecolor="#222222",
        linewidth=0.6,
    )
    for bar, val in zip(bars, hw_counts.values):
        pct = val / len(df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 120,
            f"{val:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, hw_counts.max() * 1.16)
    ax.set_title("High-Waste Household Class Balance (>= 320g/day)")
    ax.set_ylabel("Households")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, "01_target_distributions.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    cols = ALL_NUMERICAL_FEATURES + [TARGET_WASTE_REG, TARGET_SPEND_REG]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        annot_kws={"size": 7.5},
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        "Feature & Target Correlation Heatmap",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    plt.xticks(rotation=40, ha="right", fontsize=8.5)
    plt.yticks(fontsize=8.5)
    fig.tight_layout()
    return _save_fig(fig, "02_correlation_heatmap.png")


def _plot_behavioral_habits_impact(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Impact of Household Behavioral Habits on Daily Food Waste",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    configs = [
        ("Meal_Planning", ["Never", "Sometimes", "Usually", "Always"], "Meal Planning Habit"),
        ("Leftover_Reuse_Habit", ["Never", "Sometimes", "Often", "Always"], "Leftover Reuse Habit"),
        ("Food_Storage_Quality_1_to_5", [1, 2, 3, 4, 5], "Food Storage Quality (1=Poor, 5=Excellent)"),
        (
            "Grocery_Shopping_Frequency",
            ["Daily", "2-3_times_week", "Weekly", "Fortnightly"],
            "Grocery Shopping Frequency",
        ),
    ]

    for ax, (col, order, title) in zip(axes.flat, configs):
        sns.boxplot(
            data=df,
            x=col,
            y=TARGET_WASTE_REG,
            order=order,
            hue=col,
            hue_order=order,
            palette="YlGnBu_r",
            legend=False,
            showfliers=False,
            ax=ax,
        )
        means = df.groupby(col)[TARGET_WASTE_REG].mean().reindex(order)
        for idx, val in enumerate(means.values):
            ax.text(
                idx,
                val + 5,
                f"{val:.0f}g",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#1d3557",
            )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Daily Food Waste (g)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, "03_behavioral_habits_impact.png")


def _plot_demographics_and_season(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Household Size, Diet, Seasonality & Grocery Spend vs Daily Food Waste",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # 1. Household Size vs Daily Waste
    ax = axes[0, 0]
    size_stats = df.groupby("Household_Size")[TARGET_WASTE_REG].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        size_stats["Household_Size"],
        size_stats["mean"],
        yerr=size_stats["std"],
        fmt="-o",
        color="#264653",
        ecolor="#a8dadc",
        elinewidth=2.5,
        capsize=4,
        linewidth=2.5,
    )
    for _, row in size_stats.iterrows():
        ax.text(
            row["Household_Size"],
            row["mean"] + 12,
            f"{row['mean']:.0f}g",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_title("Daily Food Waste by Household Size (Mean +/- SD)")
    ax.set_xlabel("Household Size")
    ax.set_ylabel("Daily Food Waste (g)")

    # 2. Diet Type vs Daily Waste
    ax = axes[0, 1]
    diet_order = ["Vegan", "Vegetarian", "Mixed"]
    sns.violinplot(
        data=df,
        x="Diet_Type",
        y=TARGET_WASTE_REG,
        order=diet_order,
        hue="Diet_Type",
        hue_order=diet_order,
        palette="Set2",
        legend=False,
        inner="quartile",
        ax=ax,
    )
    ax.set_title("Daily Food Waste by Diet Type")
    ax.set_xlabel("")
    ax.set_ylabel("Daily Food Waste (g)")

    # 3. Season & Day Type grouped bar
    ax = axes[1, 0]
    sns.barplot(
        data=df,
        x="Season",
        y=TARGET_WASTE_REG,
        hue="Day_Type",
        palette=["#457b9d", "#e76f51"],
        errorbar="sd",
        ax=ax,
    )
    ax.set_title("Average Daily Waste by Season & Day Type")
    ax.set_xlabel("")
    ax.set_ylabel("Daily Food Waste (g)")
    ax.legend(title="Day Type")

    # 4. Weekly Grocery Spend vs Daily Waste (sampled for visual clarity)
    ax = axes[1, 1]
    sample_df = df.sample(n=min(2500, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="Weekly_Grocery_Spend_INR",
        y=TARGET_WASTE_REG,
        hue=TARGET_WASTE_LEVEL,
        hue_order=WASTE_LEVEL_ORDER,
        palette=PALETTE,
        alpha=0.55,
        s=22,
        edgecolor="none",
        ax=ax,
    )
    ax.set_title("Weekly Grocery Spend vs Daily Food Waste (2,500 sample)")
    ax.set_xlabel("Weekly Grocery Spend (INR)")
    ax.set_ylabel("Daily Food Waste (g)")
    ax.legend(title="Waste Level", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, "04_household_demographics_and_season.png")


def _plot_storage_and_income_patterns(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Storage Risk Index & Income-Spend Dynamics",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # 1. Days Stored vs Storage Quality Heatmap of Mean Daily Waste
    ax = axes[0]
    pivot = df.pivot_table(
        index="Days_Food_Stored",
        columns="Food_Storage_Quality_1_to_5",
        values=TARGET_WASTE_REG,
        aggfunc="mean",
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Daily Waste (g)"},
        ax=ax,
    )
    ax.set_title("Mean Daily Waste (g): Days Stored vs Storage Quality")
    ax.set_xlabel("Food Storage Quality (1-5)")
    ax.set_ylabel("Days Food Stored")

    # 2. Monthly Income vs Estimated Monthly Grocery Spend
    ax = axes[1]
    sample_df = df.sample(n=min(2500, len(df)), random_state=42)
    sns.regplot(
        data=sample_df,
        x="Monthly_Household_Income_INR",
        y=TARGET_SPEND_REG,
        scatter_kws={"alpha": 0.35, "s": 18, "color": "#2a9d8f"},
        line_kws={"color": "#e76f51", "linewidth": 2.2},
        ax=ax,
    )
    corr_val = df["Monthly_Household_Income_INR"].corr(df[TARGET_SPEND_REG])
    ax.set_title(f"Monthly Income vs Grocery Spend (r = {corr_val:.3f})")
    ax.set_xlabel("Monthly Household Income (INR)")
    ax.set_ylabel("Estimated Monthly Grocery Spend (INR)")

    fig.tight_layout()
    return _save_fig(fig, "05_storage_and_income_patterns.png")


def run_eda(cleaned_path: Path = CLEANED_DATA_PATH) -> dict:
    """Compute EDA statistics and generate all EDA charts."""
    ensure_directories()
    df = pd.read_csv(cleaned_path)

    # 1. Numerical summary table
    num_cols = ALL_NUMERICAL_FEATURES + [TARGET_WASTE_REG, TARGET_SPEND_REG]
    num_summary = df[num_cols].describe().T
    num_summary["skew"] = df[num_cols].skew()
    num_summary.round(3).to_csv(REPORTS_DIR / "eda_numerical_summary.csv")

    # 2. Categorical breakdown vs daily waste
    cat_rows = []
    for col in RAW_CATEGORICAL_FEATURES + [TARGET_WASTE_LEVEL, TARGET_HIGH_WASTE]:
        grp = df.groupby(col)[TARGET_WASTE_REG].agg(["count", "mean", "median", "std"])
        for category, row in grp.iterrows():
            cat_rows.append(
                {
                    "feature": col,
                    "category": str(category),
                    "count": int(row["count"]),
                    "pct": round(float(row["count"] / len(df) * 100.0), 2),
                    "mean_waste_g": round(float(row["mean"]), 2),
                    "median_waste_g": round(float(row["median"]), 2),
                    "std_waste_g": round(float(row["std"]), 2),
                }
            )
    pd.DataFrame(cat_rows).to_csv(
        REPORTS_DIR / "eda_categorical_summary.csv", index=False
    )

    # 3. Target correlations table
    corr_df = pd.DataFrame(
        {
            "corr_with_daily_waste": df[ALL_NUMERICAL_FEATURES].corrwith(
                df[TARGET_WASTE_REG]
            ),
            "corr_with_monthly_spend": df[ALL_NUMERICAL_FEATURES].corrwith(
                df[TARGET_SPEND_REG]
            ),
        }
    ).sort_values("corr_with_daily_waste", ascending=False)
    corr_df.round(4).to_csv(REPORTS_DIR / "eda_target_correlations.csv")

    # 4. Generate all EDA visualizations
    chart_paths = [
        _plot_target_distributions(df),
        _plot_correlation_heatmap(df),
        _plot_behavioral_habits_impact(df),
        _plot_demographics_and_season(df),
        _plot_storage_and_income_patterns(df),
    ]

    # 5. Extract structured key insights
    top_pos = corr_df["corr_with_daily_waste"].head(3).round(3).to_dict()
    top_neg = corr_df["corr_with_daily_waste"].tail(3).round(3).to_dict()
    meal_plan_means = df.groupby("Meal_Planning")[TARGET_WASTE_REG].mean().round(1).to_dict()
    leftover_means = df.groupby("Leftover_Reuse_Habit")[TARGET_WASTE_REG].mean().round(1).to_dict()

    insights = {
        "total_households": int(len(df)),
        "mean_daily_waste_g": round(float(df[TARGET_WASTE_REG].mean()), 2),
        "median_daily_waste_g": round(float(df[TARGET_WASTE_REG].median()), 2),
        "mean_monthly_spend_inr": round(float(df[TARGET_SPEND_REG].mean()), 2),
        "high_waste_share_pct": round(
            float((df[TARGET_HIGH_WASTE] == "Yes").mean() * 100.0), 2
        ),
        "waste_level_distribution_pct": (
            df[TARGET_WASTE_LEVEL].value_counts(normalize=True).mul(100).round(2).to_dict()
        ),
        "top_positive_waste_correlations": top_pos,
        "top_negative_waste_correlations": top_neg,
        "mean_waste_by_meal_planning": meal_plan_means,
        "mean_waste_by_leftover_reuse": leftover_means,
        "generated_charts": [str(p) for p in chart_paths],
    }

    with open(REPORTS_DIR / "eda_key_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    return insights


if __name__ == "__main__":
    summary = run_eda()
    print(
        f"EDA complete: {len(summary['generated_charts'])} charts saved to {EDA_VIZ_DIR}"
    )
