"""Data cleaning, validation, imputation, and feature engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    LEFTOVER_REUSE_MAP,
    MEAL_PLANNING_MAP,
    RAW_CATEGORICAL_FEATURES,
    RAW_DATA_PATH,
    RAW_NUMERICAL_FEATURES,
    REPORTS_DIR,
    TARGET_COLUMNS,
    ensure_directories,
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic domain features from raw household input columns.

    Uses only non-target input columns so this function is completely leak-free
    and can be reused identically at inference time.
    """
    out = df.copy()

    hh_size = out["Household_Size"].clip(lower=1)
    income = out["Monthly_Household_Income_INR"].clip(lower=1)
    storage_quality = out["Food_Storage_Quality_1_to_5"].clip(lower=1)

    out["Income_Per_Capita_INR"] = (income / hh_size).round(2)
    out["Grocery_Spend_Per_Capita_INR"] = (
        out["Weekly_Grocery_Spend_INR"] / hh_size
    ).round(2)
    out["Grocery_Income_Ratio"] = (
        (out["Weekly_Grocery_Spend_INR"] * 4.33) / income
    ).round(4)
    out["Child_Ratio"] = (out["Children"] / hh_size).round(4)
    out["Eating_Out_Total_Per_Week"] = (
        out["Dining_Out_Days_Per_Week"] + out["Food_Delivery_Orders_Per_Week"]
    )
    out["Storage_Risk_Index"] = (out["Days_Food_Stored"] / storage_quality).round(4)

    meal_plan_score = out["Meal_Planning"].map(MEAL_PLANNING_MAP).fillna(1)
    leftover_score = out["Leftover_Reuse_Habit"].map(LEFTOVER_REUSE_MAP).fillna(1)
    # Normalize combined behavioral score to a 0-10 scale (max raw sum = 3 + 3 + 5 = 11, min = 1)
    raw_habit = meal_plan_score + leftover_score + storage_quality
    out["Habit_Score"] = (((raw_habit - 1.0) / 10.0) * 10.0).round(2)

    return out


def _audit_iqr_outliers(df: pd.DataFrame, columns: list[str]) -> Dict[str, dict]:
    """Compute IQR outlier bounds and counts for numerical columns."""
    audit: Dict[str, dict] = {}
    for col in columns:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        audit[col] = {
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3),
            "outlier_count": outlier_count,
            "outlier_pct": round((outlier_count / max(len(series), 1)) * 100.0, 2),
        }
    return audit


def clean_dataset(
    raw_path: Path = RAW_DATA_PATH,
    cleaned_path: Path = CLEANED_DATA_PATH,
) -> Tuple[pd.DataFrame, dict]:
    """Load, audit, clean, feature-engineer, and persist the household dataset."""
    ensure_directories()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    df_raw = pd.read_csv(raw_path)
    initial_rows, initial_cols = df_raw.shape
    missing_before = df_raw.isnull().sum().to_dict()
    duplicate_rows = int(df_raw.duplicated().sum())

    # Build column-level quality audit table before cleaning
    audit_rows = []
    for col in df_raw.columns:
        audit_rows.append(
            {
                "column": col,
                "dtype": str(df_raw[col].dtype),
                "missing_count": int(df_raw[col].isnull().sum()),
                "missing_pct": round(float(df_raw[col].isnull().mean() * 100.0), 3),
                "unique_values": int(df_raw[col].nunique(dropna=True)),
            }
        )
    pd.DataFrame(audit_rows).to_csv(
        REPORTS_DIR / "dataset_quality_audit.csv", index=False
    )

    df = df_raw.copy()

    # 1. Strip stray whitespace in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()

    # 2. Remove exact duplicates if present
    if duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. Validate structural consistency (Household_Size == Adults + Children)
    size_mismatch_count = int(
        (df["Household_Size"] != df["Adults"] + df["Children"]).sum()
    )
    if size_mismatch_count > 0:
        df["Household_Size"] = df["Adults"] + df["Children"]

    # 4. Impute missing categorical values with column mode
    imputation_values: Dict[str, object] = {}
    for col in RAW_CATEGORICAL_FEATURES:
        mode_val = str(df[col].mode(dropna=True).iloc[0])
        imputation_values[col] = mode_val
        if df[col].isnull().any():
            df[col] = df[col].fillna(mode_val)
        df[col] = df[col].astype(str)

    # 5. Impute missing numerical values cleanly
    # Food_Storage_Quality_1_to_5 -> rounded integer median
    storage_median = float(round(df["Food_Storage_Quality_1_to_5"].median()))
    imputation_values["Food_Storage_Quality_1_to_5"] = storage_median
    df["Food_Storage_Quality_1_to_5"] = (
        df["Food_Storage_Quality_1_to_5"]
        .fillna(storage_median)
        .clip(1, 5)
        .round()
        .astype(int)
    )

    # Weekly_Grocery_Spend_INR -> Household_Size group median (fallback to global median)
    global_weekly_median = float(df["Weekly_Grocery_Spend_INR"].median())
    group_medians = (
        df.groupby("Household_Size")["Weekly_Grocery_Spend_INR"].median().to_dict()
    )
    imputation_values["Weekly_Grocery_Spend_INR_global_median"] = round(
        global_weekly_median, 2
    )
    if df["Weekly_Grocery_Spend_INR"].isnull().any():
        df["Weekly_Grocery_Spend_INR"] = df["Weekly_Grocery_Spend_INR"].fillna(
            df["Household_Size"].map(group_medians)
        )
        df["Weekly_Grocery_Spend_INR"] = df["Weekly_Grocery_Spend_INR"].fillna(
            global_weekly_median
        )

    # Defensive imputation for any other numeric column
    for col in RAW_NUMERICAL_FEATURES:
        if df[col].isnull().any():
            med = float(df[col].median())
            imputation_values[col] = round(med, 4)
            df[col] = df[col].fillna(med)

    # 6. Enforce domain validity bounds without truncating legitimate high earners
    df["Local_Market_Purchase_Share"] = df["Local_Market_Purchase_Share"].clip(0.0, 1.0)
    df["Days_Food_Stored"] = df["Days_Food_Stored"].clip(lower=0)
    df["Meals_Cooked_Per_Day"] = df["Meals_Cooked_Per_Day"].clip(lower=0.5)
    df["Weekly_Grocery_Spend_INR"] = df["Weekly_Grocery_Spend_INR"].clip(lower=0.0).round(2)

    # Convert remaining string columns cleanly
    for col in ["Household_ID", "Food_Waste_Level", "High_Waste_Household"]:
        df[col] = df[col].astype(str)

    # 7. IQR outlier audit (logged for transparency; legitimate extremes retained)
    outlier_audit = _audit_iqr_outliers(
        df,
        RAW_NUMERICAL_FEATURES
        + ["Daily_Food_Waste_Grams", "Estimated_Monthly_Grocery_Spend_INR"],
    )

    # 8. Deterministic domain feature engineering
    df = add_engineered_features(df)

    # Final verification
    missing_after = int(df.isnull().sum().sum())
    assert missing_after == 0, f"Unexpected remaining missing values: {missing_after}"

    # Save cleaned dataset
    df.to_csv(cleaned_path, index=False)

    summary = {
        "raw_data_path": str(raw_path),
        "cleaned_data_path": str(cleaned_path),
        "initial_shape": [initial_rows, initial_cols],
        "final_shape": [int(df.shape[0]), int(df.shape[1])],
        "duplicate_rows_removed": duplicate_rows,
        "household_size_mismatches_fixed": size_mismatch_count,
        "total_missing_values_imputed": int(sum(missing_before.values())),
        "missing_before_by_column": {
            k: int(v) for k, v in missing_before.items() if v > 0
        },
        "missing_after": missing_after,
        "imputation_values": imputation_values,
        "engineered_features_added": [
            c for c in df.columns if c not in df_raw.columns
        ],
        "outlier_audit": outlier_audit,
        "target_columns": TARGET_COLUMNS,
    }

    with open(REPORTS_DIR / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, summary


if __name__ == "__main__":
    cleaned_df, report = clean_dataset()
    print(
        f"Cleaning complete: {report['initial_shape']} -> {report['final_shape']} | "
        f"Imputed {report['total_missing_values_imputed']} missing values."
    )
