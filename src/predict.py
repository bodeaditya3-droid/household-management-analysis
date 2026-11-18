"""Inference and personalized recommendation module for Smart Household Food Waste ML."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_MODEL_FEATURES,
    CLUSTER_FEATURES,
    CLUSTER_MODEL_PATH,
    HIGH_WASTE_CLF_MODEL_PATH,
    MODEL_METADATA_PATH,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    SPEND_REG_MODEL_PATH,
    WASTE_LEVEL_CLF_MODEL_PATH,
    WASTE_LEVEL_ORDER,
    WASTE_REG_MODEL_PATH,
)
from src.data_cleaning import add_engineered_features


class FoodWastePredictor:
    """Loads persisted ML pipelines and generates household predictions & advice."""

    def __init__(self) -> None:
        required = [
            WASTE_REG_MODEL_PATH,
            SPEND_REG_MODEL_PATH,
            HIGH_WASTE_CLF_MODEL_PATH,
            WASTE_LEVEL_CLF_MODEL_PATH,
            CLUSTER_MODEL_PATH,
            MODEL_METADATA_PATH,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Trained model artifacts not found. Run `python run_pipeline.py` first. "
                f"Missing: {missing}"
            )

        self.waste_model = joblib.load(WASTE_REG_MODEL_PATH)
        self.spend_model = joblib.load(SPEND_REG_MODEL_PATH)
        self.high_waste_model = joblib.load(HIGH_WASTE_CLF_MODEL_PATH)
        self.waste_level_model = joblib.load(WASTE_LEVEL_CLF_MODEL_PATH)
        self.cluster_model = joblib.load(CLUSTER_MODEL_PATH)

        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _prepare_dataframe(self, records: pd.DataFrame) -> pd.DataFrame:
        df = records.copy()
        expected_raw = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES
        missing_cols = [c for c in expected_raw if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required input columns: {missing_cols}")

        # Keep Household_Size consistent with Adults + Children
        df["Household_Size"] = (df["Adults"] + df["Children"]).clip(lower=1)
        df = add_engineered_features(df)
        return df

    @staticmethod
    def _generate_recommendations(
        row: pd.Series, predicted_waste_g: float, high_waste_prob: float
    ) -> List[str]:
        """Generate tailored, actionable waste-reduction recommendations."""
        recs: List[str] = []

        if row["Food_Storage_Quality_1_to_5"] <= 3:
            recs.append(
                "Upgrade food storage practices (airtight containers, fridge temperature zones, "
                "and FIFO labeling) — poor storage quality is one of the strongest modifiable waste drivers."
            )

        if row["Meal_Planning"] in ("Never", "Sometimes"):
            recs.append(
                "Adopt a simple weekly meal plan before shopping to match grocery quantities "
                "with actual meals cooked at home."
            )

        if row["Leftover_Reuse_Habit"] in ("Never", "Sometimes"):
            recs.append(
                "Designate 1-2 'use-it-up' meals each week to repurpose cooked leftovers "
                "and perishable produce before spoilage."
            )

        if row["Days_Food_Stored"] >= 5:
            recs.append(
                "Reduce average food storage duration (currently "
                f"{int(row['Days_Food_Stored'])} days) by buying perishables in smaller, "
                "more frequent batches or freezing surplus portions early."
            )

        if row["Dining_Out_Days_Per_Week"] + row["Food_Delivery_Orders_Per_Week"] >= 7:
            recs.append(
                "Coordinate grocery quantities with frequent dining out / delivery orders "
                f"({int(row['Dining_Out_Days_Per_Week'] + row['Food_Delivery_Orders_Per_Week'])}/week) "
                "so fresh groceries do not go unused."
            )

        if row["Grocery_Shopping_Frequency"] == "Fortnightly" and row["Food_Storage_Quality_1_to_5"] <= 3:
            recs.append(
                "Shift perishables from fortnightly bulk runs to weekly or 2-3x/week purchases "
                "to minimize spoilage."
            )

        if row["Children"] >= 2 and predicted_waste_g >= 300:
            recs.append(
                "Serve smaller initial portions for children with optional seconds to cut plate waste."
            )

        if not recs:
            if high_waste_prob >= 0.5:
                recs.append(
                    "Your habits are already strong, but household scale drives higher volume — "
                    "track top 3 discarded items for 2 weeks to pinpoint portion adjustments."
                )
            else:
                recs.append(
                    "Great job! Maintain your meal planning, storage discipline, and leftover "
                    "reuse habits to keep food waste consistently low."
                )

        return recs

    def predict_single(self, household_input: Dict[str, Any]) -> Dict[str, Any]:
        """Predict waste, monthly grocery spend, risk tier, segment, and recommendations."""
        df = self._prepare_dataframe(pd.DataFrame([household_input]))
        X = df[ALL_MODEL_FEATURES]

        waste_g = float(np.clip(self.waste_model.predict(X)[0], 50.0, 1200.0))
        spend_inr = float(max(0.0, self.spend_model.predict(X)[0]))

        hw_proba = float(self.high_waste_model.predict_proba(X)[0, 1])
        hw_label = "Yes" if hw_proba >= 0.5 else "No"

        lvl_pred = str(self.waste_level_model.predict(X)[0])
        lvl_classes = list(self.waste_level_model.classes_)
        lvl_probs_raw = self.waste_level_model.predict_proba(X)[0]
        lvl_probs = {
            cls: round(float(lvl_probs_raw[lvl_classes.index(cls)]), 4)
            for cls in WASTE_LEVEL_ORDER
            if cls in lvl_classes
        }

        cluster_id = int(self.cluster_model.predict(df[CLUSTER_FEATURES])[0])
        segment_info = self.metadata["best_models"]["household_clustering"]["segments"][
            str(cluster_id)
        ]

        pop_mean = float(self.metadata["population_benchmarks"]["mean_daily_waste_g"])
        diff_vs_avg_pct = ((waste_g - pop_mean) / pop_mean) * 100.0

        row = df.iloc[0]
        recommendations = self._generate_recommendations(row, waste_g, hw_proba)

        return {
            "predicted_daily_waste_grams": round(waste_g, 1),
            "predicted_monthly_waste_kg": round((waste_g * 30.0) / 1000.0, 2),
            "predicted_annual_waste_kg": round((waste_g * 365.0) / 1000.0, 1),
            "predicted_monthly_spend_inr": round(spend_inr, 0),
            "predicted_waste_level": lvl_pred,
            "waste_level_probabilities": lvl_probs,
            "high_waste_household": hw_label,
            "high_waste_probability": round(hw_proba, 4),
            "comparison_vs_average_pct": round(diff_vs_avg_pct, 1),
            "population_mean_waste_g": pop_mean,
            "household_segment": {
                "cluster_id": cluster_id,
                **segment_info,
            },
            "engineered_metrics": {
                "Habit_Score": float(row["Habit_Score"]),
                "Storage_Risk_Index": float(row["Storage_Risk_Index"]),
                "Income_Per_Capita_INR": float(row["Income_Per_Capita_INR"]),
                "Grocery_Spend_Per_Capita_INR": float(row["Grocery_Spend_Per_Capita_INR"]),
                "Grocery_Income_Ratio": float(row["Grocery_Income_Ratio"]),
            },
            "recommendations": recommendations,
        }

    def predict_batch(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions over a DataFrame of households."""
        df = self._prepare_dataframe(records_df)
        X = df[ALL_MODEL_FEATURES]

        out = records_df.copy()
        out["Predicted_Daily_Waste_Grams"] = np.round(
            np.clip(self.waste_model.predict(X), 50.0, 1200.0), 1
        )
        out["Predicted_Monthly_Spend_INR"] = np.round(
            np.clip(self.spend_model.predict(X), 0.0, None), 0
        )
        out["Predicted_Waste_Level"] = self.waste_level_model.predict(X)
        out["High_Waste_Probability"] = np.round(
            self.high_waste_model.predict_proba(X)[:, 1], 4
        )
        out["Predicted_High_Waste"] = np.where(
            out["High_Waste_Probability"] >= 0.5, "Yes", "No"
        )
        cluster_ids = self.cluster_model.predict(df[CLUSTER_FEATURES])
        seg_map = self.metadata["best_models"]["household_clustering"]["segments"]
        out["Household_Segment"] = [seg_map[str(int(c))]["name"] for c in cluster_ids]
        return out
