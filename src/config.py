"""Centralized configuration, paths, and feature definitions."""

from pathlib import Path

# Project root and directories (all artifacts stay strictly inside this project folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VIZ_DIR = PROJECT_ROOT / "visualizations"
EDA_VIZ_DIR = VIZ_DIR / "eda"
EVAL_VIZ_DIR = VIZ_DIR / "evaluation"

# Dataset files
RAW_DATA_PATH = DATASET_DIR / "smart_household_consumption_ml_dataset_10000.csv"
CLEANED_DATA_PATH = DATASET_DIR / "cleaned_household_food_waste_10000.csv"

# Model files
WASTE_REG_MODEL_PATH = MODELS_DIR / "waste_regressor.joblib"
SPEND_REG_MODEL_PATH = MODELS_DIR / "spend_regressor.joblib"
HIGH_WASTE_CLF_MODEL_PATH = MODELS_DIR / "high_waste_classifier.joblib"
WASTE_LEVEL_CLF_MODEL_PATH = MODELS_DIR / "waste_level_classifier.joblib"
CLUSTER_MODEL_PATH = MODELS_DIR / "household_clusterer.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Column definitions
ID_COL = "Household_ID"

RAW_NUMERICAL_FEATURES = [
    "Household_Size",
    "Adults",
    "Children",
    "Monthly_Household_Income_INR",
    "Meals_Cooked_Per_Day",
    "Food_Storage_Quality_1_to_5",
    "Dining_Out_Days_Per_Week",
    "Food_Delivery_Orders_Per_Week",
    "Weekly_Grocery_Spend_INR",
    "Days_Food_Stored",
    "Local_Market_Purchase_Share",
]

RAW_CATEGORICAL_FEATURES = [
    "Work_From_Home",
    "Meal_Planning",
    "Grocery_Shopping_Frequency",
    "Leftover_Reuse_Habit",
    "Diet_Type",
    "Season",
    "Day_Type",
]

ENGINEERED_FEATURES = [
    "Income_Per_Capita_INR",
    "Grocery_Spend_Per_Capita_INR",
    "Grocery_Income_Ratio",
    "Child_Ratio",
    "Eating_Out_Total_Per_Week",
    "Storage_Risk_Index",
    "Habit_Score",
]

ALL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_FEATURES
ALL_CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES
ALL_MODEL_FEATURES = ALL_NUMERICAL_FEATURES + ALL_CATEGORICAL_FEATURES

# Targets
TARGET_WASTE_REG = "Daily_Food_Waste_Grams"
TARGET_SPEND_REG = "Estimated_Monthly_Grocery_Spend_INR"
TARGET_WASTE_LEVEL = "Food_Waste_Level"
TARGET_HIGH_WASTE = "High_Waste_Household"

TARGET_COLUMNS = [
    TARGET_WASTE_REG,
    TARGET_SPEND_REG,
    TARGET_WASTE_LEVEL,
    TARGET_HIGH_WASTE,
]

WASTE_LEVEL_ORDER = ["Low", "Moderate", "High", "Very_High"]

# Ordinal encodings used for deterministic domain feature engineering
MEAL_PLANNING_MAP = {"Never": 0, "Sometimes": 1, "Usually": 2, "Always": 3}
LEFTOVER_REUSE_MAP = {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3}

# Clustering features (behavioral + household structure)
CLUSTER_FEATURES = [
    "Household_Size",
    "Monthly_Household_Income_INR",
    "Weekly_Grocery_Spend_INR",
    "Meals_Cooked_Per_Day",
    "Food_Storage_Quality_1_to_5",
    "Days_Food_Stored",
    "Eating_Out_Total_Per_Week",
    "Habit_Score",
    "Storage_Risk_Index",
]

# Reproducibility and split settings
RANDOM_STATE = 42
TEST_SIZE = 0.20


def ensure_directories() -> None:
    """Create all project output directories if they do not exist."""
    for directory in (
        DATASET_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        VIZ_DIR,
        EDA_VIZ_DIR,
        EVAL_VIZ_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
