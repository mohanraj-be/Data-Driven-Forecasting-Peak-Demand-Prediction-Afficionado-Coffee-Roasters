import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def wape_metric(y_true, y_pred):
    """Calculates Weighted Absolute Percentage Error."""
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0

def train_and_evaluate_all():
    print("=" * 70)
    print("STARTING AFFICIONADO COFFEE ROASTERS ML MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load processed hourly datasets
    store_df = pd.read_csv(os.path.join(DATA_DIR, 'store_hourly_features.csv'))
    store_df['datetime_hour'] = pd.to_datetime(store_df['datetime_hour'])
    
    cat_df = pd.read_csv(os.path.join(DATA_DIR, 'category_hourly_features.csv'))
    cat_df['datetime_hour'] = pd.to_datetime(cat_df['datetime_hour'])
    
    # -------------------------------------------------------------
    # 1. ENCODING CATEGORICAL FEATURES
    # -------------------------------------------------------------
    le_store = LabelEncoder()
    store_df['store_encoded'] = le_store.fit_transform(store_df['store_location'])
    cat_df['store_encoded'] = le_store.transform(cat_df['store_location'])
    
    le_cat = LabelEncoder()
    cat_df['category_encoded'] = le_cat.fit_transform(cat_df['product_category'])
    
    # Chronological Train / Validation / Test Split
    # Train: Jan 1 to Apr 30 (Months 1-4, ~66% of days)
    # Val: May 1 to May 31 (Month 5, ~17% of days)
    # Test: Jun 1 to Jun 30 (Month 6, ~17% of days)
    
    train_mask = store_df['month'] <= 4
    val_mask = store_df['month'] == 5
    test_mask = store_df['month'] == 6
    
    # Feature columns for Store Hourly models
    feature_cols = [
        'store_encoded', 'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'lag_1h_qty', 'lag_2h_qty', 'lag_15h_qty', 'lag_105h_qty',
        'rolling_mean_3h_qty', 'rolling_mean_15h_qty', 'rolling_mean_7d_qty', 'rolling_std_15h_qty',
        'lag_1h_rev', 'lag_15h_rev'
    ]
    
    X_train = store_df.loc[train_mask, feature_cols]
    y_train_qty = store_df.loc[train_mask, 'quantity_demanded']
    y_train_rev = store_df.loc[train_mask, 'total_revenue']
    y_train_peak = store_df.loc[train_mask, 'is_peak_demand']
    y_train_tier = store_df.loc[train_mask, 'demand_tier']
    
    X_val = store_df.loc[val_mask, feature_cols]
    y_val_qty = store_df.loc[val_mask, 'quantity_demanded']
    y_val_rev = store_df.loc[val_mask, 'total_revenue']
    y_val_peak = store_df.loc[val_mask, 'is_peak_demand']
    y_val_tier = store_df.loc[val_mask, 'demand_tier']
    
    X_test = store_df.loc[test_mask, feature_cols]
    y_test_qty = store_df.loc[test_mask, 'quantity_demanded']
    y_test_rev = store_df.loc[test_mask, 'total_revenue']
    y_test_peak = store_df.loc[test_mask, 'is_peak_demand']
    y_test_tier = store_df.loc[test_mask, 'demand_tier']
    
    print(f"Store Dataset Split: Train={len(X_train)} rows, Val={len(X_val)} rows, Test={len(X_test)} rows")
    
    metrics_summary = {}
    
    # -------------------------------------------------------------
    # 2. BENCHMARK BASELINES & MODELS: HOURLY QUANTITY FORECASTING
    # -------------------------------------------------------------
    print("\n--- 1. Training Hourly Quantity Demand Models ---")
    
    # Baseline 1: Moving Average / Lag Baseline
    y_pred_baseline = X_test['lag_15h_qty'] # Same hour yesterday
    base_rmse = float(np.sqrt(mean_squared_error(y_test_qty, y_pred_baseline)))
    base_mae = float(mean_absolute_error(y_test_qty, y_pred_baseline))
    base_r2 = float(r2_score(y_test_qty, y_pred_baseline))
    base_wape = float(wape_metric(y_test_qty.values, y_pred_baseline.values))
    print(f"Baseline (Yesterday Same Hour) -> RMSE: {base_rmse:.3f}, MAE: {base_mae:.3f}, R2: {base_r2:.3f}, WAPE: {base_wape:.2f}%")
    
    # Baseline 2: Ridge Linear Model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train_qty)
    y_pred_ridge = ridge.predict(X_test_scaled)
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test_qty, y_pred_ridge)))
    ridge_mae = float(mean_absolute_error(y_test_qty, y_pred_ridge))
    ridge_r2 = float(r2_score(y_test_qty, y_pred_ridge))
    ridge_wape = float(wape_metric(y_test_qty.values, y_pred_ridge))
    print(f"Ridge Regressor -> RMSE: {ridge_rmse:.3f}, MAE: {ridge_mae:.3f}, R2: {ridge_r2:.3f}, WAPE: {ridge_wape:.2f}%")
    
    # Model 1A: LightGBM Regressor (Quantity)
    lgb_qty = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1
    )
    lgb_qty.fit(
        X_train, y_train_qty,
        eval_set=[(X_val, y_val_qty)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    y_pred_lgb_qty = lgb_qty.predict(X_test)
    lgb_qty_rmse = float(np.sqrt(mean_squared_error(y_test_qty, y_pred_lgb_qty)))
    lgb_qty_mae = float(mean_absolute_error(y_test_qty, y_pred_lgb_qty))
    lgb_qty_r2 = float(r2_score(y_test_qty, y_pred_lgb_qty))
    lgb_qty_wape = float(wape_metric(y_test_qty.values, y_pred_lgb_qty))
    print(f"LightGBM Qty Regressor -> RMSE: {lgb_qty_rmse:.3f}, MAE: {lgb_qty_mae:.3f}, R2: {lgb_qty_r2:.3f}, WAPE: {lgb_qty_wape:.2f}%")
    
    # Model 1B: XGBoost Regressor (Quantity)
    xgb_qty = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50
    )
    xgb_qty.fit(
        X_train, y_train_qty,
        eval_set=[(X_val, y_val_qty)],
        verbose=False
    )
    y_pred_xgb_qty = xgb_qty.predict(X_test)
    xgb_qty_rmse = float(np.sqrt(mean_squared_error(y_test_qty, y_pred_xgb_qty)))
    xgb_qty_mae = float(mean_absolute_error(y_test_qty, y_pred_xgb_qty))
    xgb_qty_r2 = float(r2_score(y_test_qty, y_pred_xgb_qty))
    xgb_qty_wape = float(wape_metric(y_test_qty.values, y_pred_xgb_qty))
    print(f"XGBoost Qty Regressor -> RMSE: {xgb_qty_rmse:.3f}, MAE: {xgb_qty_mae:.3f}, R2: {xgb_qty_r2:.3f}, WAPE: {xgb_qty_wape:.2f}%")
    
    # Model 1C: Random Forest Regressor (Quantity)
    rf_qty = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_qty.fit(X_train, y_train_qty)
    y_pred_rf_qty = rf_qty.predict(X_test)
    rf_qty_rmse = float(np.sqrt(mean_squared_error(y_test_qty, y_pred_rf_qty)))
    rf_qty_mae = float(mean_absolute_error(y_test_qty, y_pred_rf_qty))
    rf_qty_r2 = float(r2_score(y_test_qty, y_pred_rf_qty))
    rf_qty_wape = float(wape_metric(y_test_qty.values, y_pred_rf_qty))
    print(f"Random Forest Qty Regressor -> RMSE: {rf_qty_rmse:.3f}, MAE: {rf_qty_mae:.3f}, R2: {rf_qty_r2:.3f}, WAPE: {rf_qty_wape:.2f}%")
    
    metrics_summary['hourly_quantity_forecasting'] = {
        'Baseline_Lag15h': {'RMSE': base_rmse, 'MAE': base_mae, 'R2': base_r2, 'WAPE': base_wape},
        'Ridge': {'RMSE': ridge_rmse, 'MAE': ridge_mae, 'R2': ridge_r2, 'WAPE': ridge_wape},
        'LightGBM': {'RMSE': lgb_qty_rmse, 'MAE': lgb_qty_mae, 'R2': lgb_qty_r2, 'WAPE': lgb_qty_wape},
        'XGBoost': {'RMSE': xgb_qty_rmse, 'MAE': xgb_qty_mae, 'R2': xgb_qty_r2, 'WAPE': xgb_qty_wape},
        'RandomForest': {'RMSE': rf_qty_rmse, 'MAE': rf_qty_mae, 'R2': rf_qty_r2, 'WAPE': rf_qty_wape}
    }
    
    # -------------------------------------------------------------
    # 3. HOURLY REVENUE FORECASTING MODEL
    # -------------------------------------------------------------
    print("\n--- 2. Training Hourly Revenue Forecasting Models ---")
    lgb_rev = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1
    )
    lgb_rev.fit(
        X_train, y_train_rev,
        eval_set=[(X_val, y_val_rev)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    y_pred_lgb_rev = lgb_rev.predict(X_test)
    lgb_rev_rmse = float(np.sqrt(mean_squared_error(y_test_rev, y_pred_lgb_rev)))
    lgb_rev_mae = float(mean_absolute_error(y_test_rev, y_pred_lgb_rev))
    lgb_rev_r2 = float(r2_score(y_test_rev, y_pred_lgb_rev))
    lgb_rev_wape = float(wape_metric(y_test_rev.values, y_pred_lgb_rev))
    print(f"LightGBM Revenue Regressor -> RMSE: ${lgb_rev_rmse:.2f}, MAE: ${lgb_rev_mae:.2f}, R2: {lgb_rev_r2:.3f}, WAPE: {lgb_rev_wape:.2f}%")
    
    metrics_summary['hourly_revenue_forecasting'] = {
        'LightGBM': {'RMSE': lgb_rev_rmse, 'MAE': lgb_rev_mae, 'R2': lgb_rev_r2, 'WAPE': lgb_rev_wape}
    }
    
    # -------------------------------------------------------------
    # 4. PEAK DEMAND & RUSH SURGE CLASSIFIERS
    # -------------------------------------------------------------
    print("\n--- 3. Training Peak Demand & Rush Surge Classifiers ---")
    
    # Binary Peak Rush Classifier
    scale_pos_weight = (len(y_train_peak) - sum(y_train_peak)) / max(sum(y_train_peak), 1)
    xgb_peak = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=40
    )
    xgb_peak.fit(
        X_train, y_train_peak,
        eval_set=[(X_val, y_val_peak)],
        verbose=False
    )
    y_pred_peak = xgb_peak.predict(X_test)
    y_prob_peak = xgb_peak.predict_proba(X_test)[:, 1]
    
    peak_acc = float(accuracy_score(y_test_peak, y_pred_peak))
    peak_prec = float(precision_score(y_test_peak, y_pred_peak, zero_division=0))
    peak_rec = float(recall_score(y_test_peak, y_pred_peak, zero_division=0))
    peak_f1 = float(f1_score(y_test_peak, y_pred_peak, zero_division=0))
    peak_auc = float(roc_auc_score(y_test_peak, y_prob_peak))
    print(f"XGBoost Peak Classifier -> Accuracy: {peak_acc*100:.1f}%, Precision: {peak_prec*100:.1f}%, Recall: {peak_rec*100:.1f}%, F1: {peak_f1:.3f}, ROC-AUC: {peak_auc:.3f}")
    
    # Multi-class Demand Tier Classifier (0: Low, 1: Normal, 2: Rush Peak)
    lgb_tier = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_tier.fit(
        X_train, y_train_tier,
        eval_set=[(X_val, y_val_tier)],
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    y_pred_tier = lgb_tier.predict(X_test)
    tier_acc = float(accuracy_score(y_test_tier, y_pred_tier))
    tier_f1_macro = float(f1_score(y_test_tier, y_pred_tier, average='macro'))
    print(f"LightGBM Demand Tier Classifier -> Accuracy: {tier_acc*100:.1f}%, Macro F1: {tier_f1_macro:.3f}")
    
    metrics_summary['peak_demand_classification'] = {
        'Binary_Peak': {'Accuracy': peak_acc, 'Precision': peak_prec, 'Recall': peak_rec, 'F1': peak_f1, 'ROC_AUC': peak_auc},
        'Multi_Tier': {'Accuracy': tier_acc, 'Macro_F1': tier_f1_macro}
    }
    
    # -------------------------------------------------------------
    # 5. CATEGORY-LEVEL DEMAND FORECASTING MODEL
    # -------------------------------------------------------------
    print("\n--- 4. Training Category-Level Demand Forecasting Model ---")
    cat_train_mask = cat_df['month'] <= 4
    cat_val_mask = cat_df['month'] == 5
    cat_test_mask = cat_df['month'] == 6
    
    cat_feature_cols = [
        'store_encoded', 'category_encoded', 'hour', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'lag_1h_qty', 'lag_15h_qty', 'lag_105h_qty',
        'rolling_mean_3h_qty', 'rolling_mean_15h_qty'
    ]
    
    X_cat_train = cat_df.loc[cat_train_mask, cat_feature_cols]
    y_cat_train = cat_df.loc[cat_train_mask, 'quantity_demanded']
    X_cat_val = cat_df.loc[cat_val_mask, cat_feature_cols]
    y_cat_val = cat_df.loc[cat_val_mask, 'quantity_demanded']
    X_cat_test = cat_df.loc[cat_test_mask, cat_feature_cols]
    y_cat_test = cat_df.loc[cat_test_mask, 'quantity_demanded']
    
    lgb_cat = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1
    )
    lgb_cat.fit(
        X_cat_train, y_cat_train,
        eval_set=[(X_cat_val, y_cat_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    y_pred_cat = lgb_cat.predict(X_cat_test)
    cat_rmse = float(np.sqrt(mean_squared_error(y_cat_test, y_pred_cat)))
    cat_mae = float(mean_absolute_error(y_cat_test, y_pred_cat))
    cat_r2 = float(r2_score(y_cat_test, y_pred_cat))
    cat_wape = float(wape_metric(y_cat_test.values, y_pred_cat))
    print(f"Category LightGBM Regressor -> RMSE: {cat_rmse:.3f}, MAE: {cat_mae:.3f}, R2: {cat_r2:.3f}, WAPE: {cat_wape:.2f}%")
    
    metrics_summary['category_demand_forecasting'] = {
        'LightGBM': {'RMSE': cat_rmse, 'MAE': cat_mae, 'R2': cat_r2, 'WAPE': cat_wape}
    }
    
    # -------------------------------------------------------------
    # 6. FEATURE IMPORTANCE RANKINGS
    # -------------------------------------------------------------
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_lgb_qty': lgb_qty.feature_importances_,
        'importance_xgb_peak': xgb_peak.feature_importances_
    }).sort_values('importance_lgb_qty', ascending=False)
    
    # -------------------------------------------------------------
    # 7. SAVE MODELS AND ARTIFACTS
    # -------------------------------------------------------------
    print("\n--- 5. Saving Models and Metadata ---")
    joblib.dump(lgb_qty, os.path.join(MODELS_DIR, 'lgb_hourly_qty_model.joblib'))
    joblib.dump(xgb_qty, os.path.join(MODELS_DIR, 'xgb_hourly_qty_model.joblib'))
    joblib.dump(rf_qty, os.path.join(MODELS_DIR, 'rf_hourly_qty_model.joblib'))
    joblib.dump(lgb_rev, os.path.join(MODELS_DIR, 'lgb_hourly_rev_model.joblib'))
    joblib.dump(xgb_peak, os.path.join(MODELS_DIR, 'xgb_peak_classifier.joblib'))
    joblib.dump(lgb_tier, os.path.join(MODELS_DIR, 'lgb_tier_classifier.joblib'))
    joblib.dump(lgb_cat, os.path.join(MODELS_DIR, 'lgb_category_qty_model.joblib'))
    joblib.dump({
        'le_store': le_store,
        'le_cat': le_cat,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'cat_feature_cols': cat_feature_cols
    }, os.path.join(MODELS_DIR, 'encoders_and_scalers.joblib'))
    
    with open(os.path.join(MODELS_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    fi_df.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)
    
    # Save test predictions for evaluation visualization
    test_eval_df = store_df.loc[test_mask].copy()
    test_eval_df['pred_lgb_qty'] = y_pred_lgb_qty
    test_eval_df['pred_xgb_qty'] = y_pred_xgb_qty
    test_eval_df['pred_rf_qty'] = y_pred_rf_qty
    test_eval_df['pred_lgb_rev'] = y_pred_lgb_rev
    test_eval_df['pred_peak_prob'] = y_prob_peak
    test_eval_df['pred_peak_label'] = y_pred_peak
    test_eval_df['pred_tier'] = y_pred_tier
    test_eval_df.to_csv(os.path.join(DATA_DIR, 'test_predictions.csv'), index=False)
    
    print(f"All models, encoders, and evaluation predictions successfully saved to {MODELS_DIR} and {DATA_DIR}!")
    return metrics_summary

if __name__ == '__main__':
    train_and_evaluate_all()
