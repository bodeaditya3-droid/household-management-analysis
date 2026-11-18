"""End-to-end orchestrator for Smart Household Food Waste ML."""

from __future__ import annotations

import time

from src.data_cleaning import clean_dataset
from src.eda import run_eda
from src.evaluate_models import evaluate_all_models
from src.predict import FoodWastePredictor
from src.train_models import train_all_models


def main() -> None:
    start = time.perf_counter()
    print("=" * 72)
    print("SMART HOUSEHOLD FOOD WASTE ML - END-TO-END PIPELINE")
    print("=" * 72)

    print("\n[1/5] Data Cleaning & Feature Engineering...")
    _, clean_summary = clean_dataset()
    print(
        f"      Rows/Cols: {clean_summary['initial_shape']} -> {clean_summary['final_shape']} | "
        f"Missing imputed: {clean_summary['total_missing_values_imputed']}"
    )

    print("\n[2/5] Exploratory Data Analysis & Visualizations...")
    eda_summary = run_eda()
    print(
        f"      Mean daily waste: {eda_summary['mean_daily_waste_g']}g | "
        f"High-waste households: {eda_summary['high_waste_share_pct']}% | "
        f"Charts created: {len(eda_summary['generated_charts'])}"
    )

    print("\n[3/5] Training & Selecting ML Models...")
    meta = train_all_models()
    for task_key, info in meta["best_models"].items():
        if "name" in info:
            print(f"      {task_key}: {info['name']} -> {info['metrics']}")
        else:
            print(
                f"      {task_key}: K={info['optimal_k']} "
                f"(silhouette={info['best_silhouette']})"
            )

    print("\n[4/5] Evaluating Models & Generating Diagnostic Plots...")
    eval_summary = evaluate_all_models()
    print(
        f"      Evaluation charts created: {len(eval_summary['generated_charts'])}"
    )
    top_drivers = [d["Feature"] for d in eval_summary["top_5_waste_drivers"]]
    print(f"      Top 5 waste drivers: {', '.join(top_drivers)}")

    print("\n[5/5] Verifying Inference Engine on Sample Household...")
    predictor = FoodWastePredictor()
    sample_household = {
        "Household_Size": 4,
        "Adults": 2,
        "Children": 2,
        "Monthly_Household_Income_INR": 65000,
        "Work_From_Home": "Sometimes",
        "Meals_Cooked_Per_Day": 2.5,
        "Meal_Planning": "Sometimes",
        "Grocery_Shopping_Frequency": "Weekly",
        "Food_Storage_Quality_1_to_5": 3,
        "Leftover_Reuse_Habit": "Sometimes",
        "Diet_Type": "Mixed",
        "Dining_Out_Days_Per_Week": 3,
        "Food_Delivery_Orders_Per_Week": 4,
        "Weekly_Grocery_Spend_INR": 8500.0,
        "Days_Food_Stored": 5,
        "Local_Market_Purchase_Share": 0.45,
        "Season": "Monsoon",
        "Day_Type": "Weekend",
    }
    pred = predictor.predict_single(sample_household)
    print(
        f"      Predicted daily waste: {pred['predicted_daily_waste_grams']}g/day "
        f"({pred['predicted_annual_waste_kg']} kg/year)"
    )
    print(
        f"      Waste level: {pred['predicted_waste_level']} | "
        f"High-waste risk: {pred['high_waste_household']} ({pred['high_waste_probability'] * 100:.1f}%)"
    )
    print(
        f"      Est. monthly grocery spend: INR {pred['predicted_monthly_spend_inr']:,.0f} | "
        f"Segment: {pred['household_segment']['name']}"
    )

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
