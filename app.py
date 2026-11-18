"""Minimal Streamlit frontend for Smart Household Food Waste ML."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    CLEANED_DATA_PATH,
    EDA_VIZ_DIR,
    EVAL_VIZ_DIR,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
)
from src.predict import FoodWastePredictor

st.set_page_config(
    page_title="Smart Household Food Waste ML",
    page_icon="🥗",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> FoodWastePredictor:
    return FoodWastePredictor()


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def render_prediction_tab(predictor: FoodWastePredictor) -> None:
    st.subheader("Predict Household Food Waste & Monthly Grocery Spend")

    with st.form("household_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Household Profile**")
            adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=2)
            income = st.number_input(
                "Monthly Income (INR)",
                min_value=5000,
                max_value=300000,
                value=55000,
                step=1000,
            )
            weekly_spend = st.number_input(
                "Weekly Grocery Spend (INR)",
                min_value=500.0,
                max_value=30000.0,
                value=7500.0,
                step=250.0,
            )
            diet_type = st.selectbox("Diet Type", ["Vegetarian", "Mixed", "Vegan"])
            wfh = st.selectbox("Work From Home", ["Never", "Sometimes", "Often"])

        with col2:
            st.markdown("**Habits & Storage**")
            meal_planning = st.selectbox(
                "Meal Planning", ["Never", "Sometimes", "Usually", "Always"], index=1
            )
            leftover_reuse = st.selectbox(
                "Leftover Reuse Habit",
                ["Never", "Sometimes", "Often", "Always"],
                index=1,
            )
            storage_quality = st.slider(
                "Food Storage Quality (1 = Poor, 5 = Excellent)", 1, 5, 3
            )
            days_stored = st.slider("Days Food Stored", 0, 10, 4)
            shopping_freq = st.selectbox(
                "Grocery Shopping Frequency",
                ["Daily", "2-3_times_week", "Weekly", "Fortnightly"],
                index=2,
            )
            local_share = st.slider(
                "Local Market Purchase Share", 0.0, 1.0, 0.50, step=0.05
            )

        with col3:
            st.markdown("**Meals & Timing**")
            meals_cooked = st.slider("Meals Cooked Per Day", 1.0, 5.0, 2.5, step=0.1)
            dining_out = st.slider("Dining Out Days / Week", 0, 7, 3)
            delivery_orders = st.slider("Food Delivery Orders / Week", 0, 14, 4)
            season = st.selectbox("Season", ["Monsoon", "Winter", "Summer"])
            day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])

        submitted = st.form_submit_button("Predict Food Waste", type="primary", width="stretch")

    household_input = {
        "Household_Size": int(adults + children),
        "Adults": int(adults),
        "Children": int(children),
        "Monthly_Household_Income_INR": int(income),
        "Work_From_Home": wfh,
        "Meals_Cooked_Per_Day": float(meals_cooked),
        "Meal_Planning": meal_planning,
        "Grocery_Shopping_Frequency": shopping_freq,
        "Food_Storage_Quality_1_to_5": int(storage_quality),
        "Leftover_Reuse_Habit": leftover_reuse,
        "Diet_Type": diet_type,
        "Dining_Out_Days_Per_Week": int(dining_out),
        "Food_Delivery_Orders_Per_Week": int(delivery_orders),
        "Weekly_Grocery_Spend_INR": float(weekly_spend),
        "Days_Food_Stored": int(days_stored),
        "Local_Market_Purchase_Share": float(local_share),
        "Season": season,
        "Day_Type": day_type,
    }

    if submitted or "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = predictor.predict_single(household_input)

    result = st.session_state["last_prediction"]

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    delta_pct = result["comparison_vs_average_pct"]
    m1.metric(
        "Predicted Daily Waste",
        f"{result['predicted_daily_waste_grams']:.1f} g/day",
        delta=f"{delta_pct:+.1f}% vs avg",
        delta_color="inverse",
    )
    m2.metric(
        "Food Waste Level",
        result["predicted_waste_level"].replace("_", " "),
        delta=f"{result['predicted_annual_waste_kg']:.1f} kg/year",
        delta_color="off",
    )
    m3.metric(
        "High-Waste Risk",
        result["high_waste_household"],
        delta=f"{result['high_waste_probability'] * 100:.1f}% probability",
        delta_color="off",
    )
    m4.metric(
        "Est. Monthly Grocery Spend",
        f"INR {result['predicted_monthly_spend_inr']:,.0f}",
        delta=f"Habit Score: {result['engineered_metrics']['Habit_Score']:.1f}/10",
        delta_color="off",
    )

    seg = result["household_segment"]
    st.info(
        f"**Household Segment:** {seg['name']} "
        f"(Segment avg waste: {seg['avg_daily_waste_g']:.1f} g/day | "
        f"High-waste rate: {seg['high_waste_rate_pct']:.1f}% | "
        f"Storage Risk Index: {result['engineered_metrics']['Storage_Risk_Index']:.2f})"
    )

    left, right = st.columns([1.0, 1.1])
    with left:
        st.markdown("#### Waste Level Probability Breakdown")
        prob_df = pd.DataFrame(
            {
                "Waste Level": [
                    k.replace("_", " ") for k in result["waste_level_probabilities"].keys()
                ],
                "Probability (%)": [
                    round(v * 100.0, 1) for v in result["waste_level_probabilities"].values()
                ],
            }
        )
        fig = px.bar(
            prob_df,
            x="Probability (%)",
            y="Waste Level",
            orientation="h",
            text="Probability (%)",
            color="Waste Level",
            color_discrete_map={
                "Low": "#2a9d8f",
                "Moderate": "#e9c46a",
                "High": "#f4a261",
                "Very High": "#e76f51",
            },
        )
        fig.update_layout(
            showlegend=False,
            xaxis_range=[0, 105],
            height=270,
            margin=dict(l=10, r=20, t=10, b=10),
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Personalized Waste-Reduction Recommendations")
        for idx, rec in enumerate(result["recommendations"], start=1):
            st.markdown(f"{idx}. {rec}")


def render_eda_tab() -> None:
    st.subheader("Dataset Overview & Exploratory Data Analysis")

    insights_path = REPORTS_DIR / "eda_key_insights.json"
    cleaning_path = REPORTS_DIR / "cleaning_summary.json"
    if insights_path.exists():
        insights = load_json(insights_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Households Analyzed", f"{insights['total_households']:,}")
        c2.metric("Mean Daily Waste", f"{insights['mean_daily_waste_g']:.1f} g")
        c3.metric(
            "Mean Monthly Spend", f"INR {insights['mean_monthly_spend_inr']:,.0f}"
        )
        c4.metric("High-Waste Households", f"{insights['high_waste_share_pct']:.1f}%")

    if cleaning_path.exists():
        clean = load_json(cleaning_path)
        with st.expander("Data Cleaning & Feature Engineering Summary"):
            st.write(
                f"Raw shape: **{clean['initial_shape'][0]:,} x {clean['initial_shape'][1]}** -> "
                f"Cleaned shape: **{clean['final_shape'][0]:,} x {clean['final_shape'][1]}** | "
                f"Missing values imputed: **{clean['total_missing_values_imputed']}** | "
                f"Duplicates removed: **{clean['duplicate_rows_removed']}**"
            )
            st.write(
                "**Engineered features added:** "
                + ", ".join(f"`{c}`" for c in clean["engineered_features_added"])
            )
            if CLEANED_DATA_PATH.exists():
                st.dataframe(load_csv(CLEANED_DATA_PATH).head(20), width="stretch")

    eda_charts = [
        ("01_target_distributions.png", "Target Variable Distributions"),
        ("02_correlation_heatmap.png", "Feature & Target Correlation Heatmap"),
        ("03_behavioral_habits_impact.png", "Behavioral Habits vs Daily Food Waste"),
        ("04_household_demographics_and_season.png", "Demographics, Diet, Seasonality & Spend"),
        ("05_storage_and_income_patterns.png", "Storage Risk & Income-Spend Dynamics"),
    ]
    for filename, caption in eda_charts:
        img_path = EDA_VIZ_DIR / filename
        if img_path.exists():
            st.markdown(f"#### {caption}")
            st.image(str(img_path), width="stretch")


def render_model_performance_tab() -> None:
    st.subheader("Model Comparison & Evaluation Diagnostics")

    if MODEL_METADATA_PATH.exists():
        meta = load_json(MODEL_METADATA_PATH)
        best = meta["best_models"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Waste Regressor",
            best["waste_regression"]["name"],
            delta=f"R² = {best['waste_regression']['metrics']['Test_R2']:.4f} | RMSE = {best['waste_regression']['metrics']['Test_RMSE']:.2f}g",
            delta_color="off",
        )
        c2.metric(
            "Spend Regressor",
            best["spend_regression"]["name"],
            delta=f"R² = {best['spend_regression']['metrics']['Test_R2']:.4f} | RMSE = INR {best['spend_regression']['metrics']['Test_RMSE']:,.0f}",
            delta_color="off",
        )
        c3.metric(
            "High-Waste Classifier",
            best["high_waste_classification"]["name"],
            delta=f"ROC-AUC = {best['high_waste_classification']['metrics']['Test_ROC_AUC']:.4f} | F1 = {best['high_waste_classification']['metrics']['Test_F1']:.4f}",
            delta_color="off",
        )
        c4.metric(
            "Waste-Level Classifier",
            best["waste_level_classification"]["name"],
            delta=f"Acc = {best['waste_level_classification']['metrics']['Test_Accuracy']:.4f} | Macro-F1 = {best['waste_level_classification']['metrics']['Test_Macro_F1']:.4f}",
            delta_color="off",
        )

    tables = [
        ("Daily Food Waste Regression Comparison", "waste_regression_comparison.csv"),
        ("Monthly Grocery Spend Regression Comparison", "spend_regression_comparison.csv"),
        ("High-Waste Binary Classification Comparison", "high_waste_classification_comparison.csv"),
        ("Food Waste Level Multiclass Classification Comparison", "waste_level_classification_comparison.csv"),
        ("K-Means Household Segment Profiles", "kmeans_cluster_profiles.csv"),
    ]
    for title, csv_name in tables:
        csv_path = REPORTS_DIR / csv_name
        if csv_path.exists():
            with st.expander(title, expanded=(csv_name == "waste_regression_comparison.csv")):
                st.dataframe(load_csv(csv_path), width="stretch", hide_index=True)

    eval_charts = [
        ("01_waste_regression_evaluation.png", "Daily Food Waste Regression Diagnostics"),
        ("02_spend_regression_evaluation.png", "Monthly Grocery Spend Regression Diagnostics"),
        ("03_high_waste_classification_eval.png", "High-Waste Binary Classification (Confusion Matrix & ROC)"),
        ("04_waste_level_confusion_matrix.png", "Food Waste Level Multiclass Confusion Matrices"),
        ("05_feature_importance.png", "Permutation Feature Importance"),
        ("06_household_clusters.png", "Household K-Means Segmentation"),
    ]
    for filename, caption in eval_charts:
        img_path = EVAL_VIZ_DIR / filename
        if img_path.exists():
            st.markdown(f"#### {caption}")
            st.image(str(img_path), width="stretch")


def main() -> None:
    st.title("Smart Household Food Waste ML")
    st.caption(
        "End-to-end machine learning system for predicting household daily food waste, "
        "monthly grocery spend, risk classification, and personalized waste reduction."
    )

    if not MODEL_METADATA_PATH.exists():
        st.error(
            "Trained models were not found in `models/`. "
            "Please run `python run_pipeline.py` first to clean the dataset, train models, "
            "and generate visualizations."
        )
        return

    predictor = load_predictor()

    tab_predict, tab_eda, tab_eval = st.tabs(
        ["Predict Household Waste", "Dataset & EDA", "Model Performance"]
    )
    with tab_predict:
        render_prediction_tab(predictor)
    with tab_eda:
        render_eda_tab()
    with tab_eval:
        render_model_performance_tab()


if __name__ == "__main__":
    main()
