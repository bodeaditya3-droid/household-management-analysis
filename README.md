# Smart Household Food Waste ML

An end-to-end, self-contained machine learning system and minimal Streamlit web application for analyzing, predicting, and reducing household food waste across **10,000 households**.

## Project Structure

```text
Smart_Household_Food_Waste_ML/
├── app.py                          # Minimal Streamlit frontend (Prediction, EDA, Model Performance)
├── run_pipeline.py                 # End-to-end pipeline orchestrator
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── dataset/
│   ├── smart_household_consumption_ml_dataset_10000.csv   # Raw 10,000-row dataset
│   └── cleaned_household_food_waste_10000.csv             # Cleaned + feature-engineered dataset
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized paths, features, targets, and constants
│   ├── data_cleaning.py            # Quality audit, imputation, validation, and feature engineering
│   ├── eda.py                      # Exploratory data analysis statistics & visualizations
│   ├── train_models.py             # Multi-task ML training, CV, selection & serialization
│   ├── evaluate_models.py          # Held-out test evaluation, permutation importance & plots
│   └── predict.py                  # Inference engine + personalized waste-reduction recommendations
├── models/                         # Persisted best scikit-learn pipelines (.joblib) & metadata
├── reports/                        # Generated cleaning, EDA, and model evaluation CSV/JSON reports
└── visualizations/
    ├── eda/                        # 5 EDA charts (distributions, correlations, habits, demographics, storage)
    └── evaluation/                 # 6 diagnostic charts (regression, ROC, confusion matrices, importance, clusters)
```

## Dataset & Cleaning Summary

- **Source:** `dataset/smart_household_consumption_ml_dataset_10000.csv` (**10,000 rows × 23 columns**)
- **Missing Values Imputed:** **94** total (`Leftover_Reuse_Habit`: 25, `Food_Storage_Quality_1_to_5`: 23, `Meal_Planning`: 16, `Work_From_Home`: 15, `Weekly_Grocery_Spend_INR`: 15)
  - Categorical columns imputed with mode; `Food_Storage_Quality_1_to_5` imputed with integer median; `Weekly_Grocery_Spend_INR` imputed with `Household_Size` group median.
- **Duplicates / Structural Mismatches:** 0 duplicate rows; verified `Household_Size == Adults + Children` across all 10,000 rows.
- **Engineered Leak-Free Features (7 added -> 30 total columns):**
  - `Income_Per_Capita_INR`, `Grocery_Spend_Per_Capita_INR`, `Grocery_Income_Ratio`, `Child_Ratio`, `Eating_Out_Total_Per_Week`, `Storage_Risk_Index`, and `Habit_Score`.

## Model Performance (80/20 Stratified Holdout + 5-Fold CV)

| Task | Target | Best Model | Key Test Metrics |
| --- | --- | --- | --- |
| Daily Food Waste Regression | `Daily_Food_Waste_Grams` | **Extra Trees** | RMSE = **2.26 g**, MAE = **1.64 g**, R² = **0.9990**, MAPE = **0.61%** |
| Monthly Grocery Spend Regression | `Estimated_Monthly_Grocery_Spend_INR` | **HistGradientBoosting** | RMSE = **INR 861**, MAE = **INR 553**, R² = **0.9948**, MAPE = **2.00%** |
| High-Waste Risk (Binary) | `High_Waste_Household` | **HistGradientBoosting** | Accuracy = **99.15%**, F1 = **0.9880**, ROC-AUC = **0.9990** |
| Food Waste Level (4-Class) | `Food_Waste_Level` | **Random Forest** | Accuracy = **98.65%**, Macro-F1 = **0.9862**, Weighted-F1 = **0.9865** |
| Household Segmentation | Unsupervised (`CLUSTER_FEATURES`) | **K-Means (K = 2)** | Silhouette = **0.1631** (Low-Waste Mindful vs High-Consumption High-Waste) |

### Top Drivers of Daily Food Waste (Permutation Importance)
1. `Household_Size`
2. `Habit_Score` (engineered composite of meal planning, leftover reuse, and storage quality)
3. `Meal_Planning`
4. `Children`
5. `Leftover_Reuse_Habit`

## Quick Start

### 1. Install Dependencies
```bash
cd Smart_Household_Food_Waste_ML
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
Runs data cleaning, EDA, model training, evaluation, and an inference verification check:
```bash
python run_pipeline.py
```

### 3. Launch the Streamlit Application
```bash
streamlit run app.py
```

### 4. Streamlit Community Cloud Deployment
- **Python version:** Select **Python 3.12** (or **3.11**) in Streamlit Community Cloud **Advanced settings** (matching the prebuilt Linux wheels for the pinned scientific stack).
- **Dependencies:** `requirements.txt` pins the exact versions used during training (`scikit-learn==1.7.2`, `numpy==2.2.6`, `scipy==1.15.3`, `pandas==2.3.3`, `joblib==1.6.0`) so serialized `HistGradientBoosting*` and tree pipelines deserialize cleanly without version skew.

The Streamlit interface includes three tabs:
1. **Predict Household Waste**: Enter household attributes to get daily/annual food waste estimates, monthly grocery spend, waste risk tier probabilities, household segment, and personalized reduction recommendations.
2. **Dataset & EDA**: View dataset KPIs, cleaning summary, cleaned data preview, and all generated EDA charts.
3. **Model Performance**: Inspect model comparison tables, cluster profiles, and evaluation diagnostic plots.
