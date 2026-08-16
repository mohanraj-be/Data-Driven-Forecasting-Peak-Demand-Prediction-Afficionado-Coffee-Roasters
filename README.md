# ☕ Afficionado Coffee Roasters: Data-Driven Demand Forecasting & Peak Rush Prediction

An enterprise-grade Machine Learning solution for **Afficionado Coffee Roasters** to forecast hourly transaction volume, revenue, product category demand, and predict operational peak rush hours across all NYC retail stores (*Hell's Kitchen*, *Lower Manhattan*, *Astoria*).

---

## 🌟 Key Features

1. **Continuous Hourly Demand & Revenue Forecasting**:
   - Machine learning regressors (**XGBoost**, **LightGBM**, **Random Forest**) evaluated against chronological forward test data.
   - Outperforms historical moving average and linear baselines, achieving **$R^2 = 0.704$** and **29.8% WAPE**.

2. **Peak Demand & Rush Hour Surge Detection**:
   - Multi-tier & binary peak classification models predicting the probability of an operational surge ($P(\text{Peak})$).
   - Achieves **82.1% Accuracy**, **78.3% Precision / Recall**, and **0.907 ROC-AUC**.

3. **Smart Labor Allocation & Batch Prep Optimizer**:
   - Translates hourly volume forecasts into real-time staffing recommendations (Barista & Cashier headcounts).
   - Automated kitchen guidance for batch brewers, pre-grinding, and milk staging during morning rush intervals.

4. **Product Category-Level Decomposition**:
   - Forecasts quantity demand across all 9 product categories (Coffee, Tea, Bakery, Drinking Chocolate, Flavours, Coffee beans, Loose Tea, etc.) with **$R^2 = 0.760$**.

5. **Multi-Channel Deployment & Access**:
   - **Streamlit Web Application (`app.py`)**: Interactive manager dashboard and scenario simulator.
   - **FastAPI REST Service (`api.py`)**: High-performance API with OpenAPI / Swagger UI for POS integration.
   - **Forward 31-Day Excel Report Generator (`generate_forecast_report.py`)**: Automated monthly forecasting reports.
   - **Python CLI Tool (`predict.py`)**: Command-line interface for scripts and pipelines.

---

## 📁 Repository Structure

```
├── Afficionado Coffee Roasters.xlsx    # Raw dataset (149,116 transactions)
├── app.py                              # Interactive Streamlit Web Application
├── api.py                              # FastAPI REST microservice API
├── predict.py                          # CLI inference interface
├── generate_forecast_report.py         # Forward 31-day forecast report generator
├── requirements.txt                    # Project dependencies
├── src/
│   ├── data_pipeline.py                # Time-series resampling, lag & rolling feature engine
│   ├── train_models.py                 # Multi-model training and chronological cross-validation
│   ├── evaluate.py                     # Evaluation suite & diagnostic chart generator
│   └── predictor.py                    # Inference engine & decision logic class
├── data/
│   ├── store_hourly_features.csv       # Resampled continuous hourly dataset (8,145 slots)
│   ├── category_hourly_features.csv    # Store-Category hourly dataset (73,305 slots)
│   └── test_predictions.csv            # Test set predictions & actuals
├── models/
│   ├── xgb_hourly_qty_model.joblib     # XGBoost Quantity Regressor
│   ├── lgb_hourly_qty_model.joblib     # LightGBM Quantity Regressor
│   ├── lgb_hourly_rev_model.joblib     # LightGBM Revenue Regressor
│   ├── xgb_peak_classifier.joblib      # XGBoost Peak Demand Classifier
│   ├── lgb_tier_classifier.joblib      # LightGBM Multi-tier Classifier
│   ├── lgb_category_qty_model.joblib   # Category Demand Model
│   ├── encoders_and_scalers.joblib     # Serialized encoders and metadata
│   ├── feature_importance.csv          # Feature importance rankings
│   └── training_metrics.json           # Evaluation metrics benchmark
└── reports/
    ├── July_2025_Demand_Forecast_Report.xlsx # 31-Day Executive Forward Forecast
    └── figures/                        # Generated evaluation charts
        ├── model_benchmark_comparison.png
        ├── actual_vs_forecast_timeseries.png
        ├── peak_demand_classification_diagnostics.png
        ├── feature_importance_ranking.png
        └── store_peak_heatmap.png
```

---

## 🚀 Quick Start Guide

### 1. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```

### 2. Start the FastAPI REST Microservice
```bash
python api.py
# Access Swagger UI at: http://127.0.0.1:8000/docs
```

### 3. Generate Forward 31-Day Forecast Report
```bash
python generate_forecast_report.py
```

### 4. Command-Line Predictions
```bash
# Predict single hour for Hell's Kitchen
python predict.py --store "Hell's Kitchen" --date "2025-07-01" --hour 9 --mode hour

# Generate full daily shift schedule
python predict.py --store "Lower Manhattan" --date "2025-07-01" --mode day_schedule

# Generate category inventory breakdown
python predict.py --store "Astoria" --date "2025-07-01" --hour 8 --mode category_breakdown
```

---

## 📊 Model Benchmark Summary

| Model / Baseline | Task | RMSE | MAE | $R^2$ | WAPE | Accuracy / AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Yesterday Lag Baseline** | Qty Forecaster | 22.08 | 14.08 | 0.430 | 37.31% | — |
| **Ridge Linear Model** | Qty Forecaster | 18.02 | 12.87 | 0.620 | 34.11% | — |
| **Random Forest Regressor** | Qty Forecaster | 16.37 | 11.53 | 0.686 | 30.54% | — |
| **LightGBM Regressor** | Qty Forecaster | 16.15 | 11.26 | 0.695 | 29.85% | — |
| **XGBoost Regressor (Best)**| Qty Forecaster | **15.91** | **11.27** | **0.704** | **29.87%** | — |
| **LightGBM Revenue Model** | Rev Forecaster ($) | $55.93 | $38.12 | 0.681 | 30.91% | — |
| **XGBoost Peak Classifier**| Rush Detection | — | — | — | — | **82.1% Acc / 0.907 AUC** |
| **LightGBM Category Model**| Category Breakdown | 3.87 | 1.89 | **0.760** | 44.99% | — |
