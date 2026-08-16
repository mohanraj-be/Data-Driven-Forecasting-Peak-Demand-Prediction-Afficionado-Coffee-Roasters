import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
FIGS_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def generate_evaluation_visualizations():
    print("Generating comprehensive model evaluation charts...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_predictions.csv'))
    test_df['datetime_hour'] = pd.to_datetime(test_df['datetime_hour'])
    
    with open(os.path.join(MODELS_DIR, 'training_metrics.json'), 'r') as f:
        metrics = json.load(f)
        
    fi_df = pd.read_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'))
    
    # -------------------------------------------------------------
    # 1. Model Benchmark Comparison Bar Chart
    # -------------------------------------------------------------
    reg_metrics = metrics['hourly_quantity_forecasting']
    models = list(reg_metrics.keys())
    rmse_vals = [reg_metrics[m]['RMSE'] for m in models]
    r2_vals = [reg_metrics[m]['R2'] for m in models]
    wape_vals = [reg_metrics[m]['WAPE'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # RMSE
    sns.barplot(x=models, y=rmse_vals, ax=axes[0], palette='Blues_r')
    axes[0].set_title('Root Mean Squared Error (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE (Units/Hour)')
    axes[0].tick_params(axis='x', rotation=30)
    for i, v in enumerate(rmse_vals):
        axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold', fontsize=9)
        
    # R2 Score
    sns.barplot(x=models, y=r2_vals, ax=axes[1], palette='Greens')
    axes[1].set_title('R² Variance Explained (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(r2_vals):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=9)
        
    # WAPE
    sns.barplot(x=models, y=wape_vals, ax=axes[2], palette='Oranges_r')
    axes[2].set_title('Weighted Abs % Error (Lower is Better)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('WAPE (%)')
    axes[2].tick_params(axis='x', rotation=30)
    for i, v in enumerate(wape_vals):
        axes[2].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    bench_path = os.path.join(FIGS_DIR, 'model_benchmark_comparison.png')
    plt.savefig(bench_path)
    plt.close()
    print(f"Saved {bench_path}")
    
    # -------------------------------------------------------------
    # 2. Time-Series Actual vs Forecast (1-Week Zoom for All 3 Stores)
    # -------------------------------------------------------------
    # Take first 7 days of test set (June 1 - June 7, 2025)
    sample_week = test_df[(test_df['datetime_hour'] >= '2025-06-01') & (test_df['datetime_hour'] <= '2025-06-07')]
    stores = test_df['store_location'].unique()
    
    fig, axes = plt.subplots(len(stores), 1, figsize=(16, 10), sharex=True)
    for i, store in enumerate(stores):
        store_sample = sample_week[sample_week['store_location'] == store]
        axes[i].plot(store_sample['datetime_hour'], store_sample['quantity_demanded'], label='Actual Demand', color='#2b2d42', linewidth=2)
        axes[i].plot(store_sample['datetime_hour'], store_sample['pred_xgb_qty'], label='XGBoost Forecast', color='#e63946', linestyle='--', linewidth=2)
        axes[i].plot(store_sample['datetime_hour'], store_sample['pred_lgb_qty'], label='LightGBM Forecast', color='#457b9d', linestyle=':', linewidth=1.5)
        axes[i].set_title(f'Store: {store} (Hourly Demand: Actual vs Forecast)', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Qty / Hour')
        axes[i].legend(loc='upper right')
        
    plt.xlabel('Date & Hour', fontsize=11, fontweight='bold')
    plt.tight_layout()
    ts_path = os.path.join(FIGS_DIR, 'actual_vs_forecast_timeseries.png')
    plt.savefig(ts_path)
    plt.close()
    print(f"Saved {ts_path}")
    
    # -------------------------------------------------------------
    # 3. Peak Demand Confusion Matrix & ROC Curve
    # -------------------------------------------------------------
    y_true_peak = test_df['is_peak_demand']
    y_pred_peak = test_df['pred_peak_label']
    y_prob_peak = test_df['pred_peak_prob']
    
    cm = confusion_matrix(y_true_peak, y_pred_peak)
    fpr, tpr, _ = roc_curve(y_true_peak, y_prob_peak)
    roc_auc = auc(fpr, tpr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal / Off-Peak', 'Rush Peak'],
                yticklabels=['Normal / Off-Peak', 'Rush Peak'])
    axes[0].set_title('Peak Demand Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Class', fontweight='bold')
    axes[0].set_ylabel('Actual Class', fontweight='bold')
    
    # ROC Curve
    axes[1].plot(fpr, tpr, color='#e63946', lw=2.5, label=f'XGBoost ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='#6c757d', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontweight='bold')
    axes[1].set_ylabel('True Positive Rate (Recall)', fontweight='bold')
    axes[1].set_title('Peak Rush Detection ROC Curve', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    roc_path = os.path.join(FIGS_DIR, 'peak_demand_classification_diagnostics.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved {roc_path}")
    
    # -------------------------------------------------------------
    # 4. Feature Importance Plot
    # -------------------------------------------------------------
    top_fi = fi_df.sort_values('importance_lgb_qty', ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_fi['feature'], top_fi['importance_lgb_qty'], color='#2a9d8f')
    ax.set_title('Top Feature Importances (LightGBM Demand Forecaster)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Importance (Split Weight)', fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(FIGS_DIR, 'feature_importance_ranking.png')
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved {fi_path}")
    
    # -------------------------------------------------------------
    # 5. Store Peak Hour Heatmap
    # -------------------------------------------------------------
    store_hour_peak = test_df.groupby(['store_location', 'hour'])['pred_peak_prob'].mean().unstack(level=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(store_hour_peak, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Peak Rush Probability'})
    ax.set_title('Store Hourly Peak Rush Probability Heatmap', fontsize=13, fontweight='bold')
    ax.set_xlabel('Store Location', fontweight='bold')
    ax.set_ylabel('Hour of Day (24-Hour)', fontweight='bold')
    plt.tight_layout()
    hm_path = os.path.join(FIGS_DIR, 'store_peak_heatmap.png')
    plt.savefig(hm_path)
    plt.close()
    print(f"Saved {hm_path}")
    
    print("All evaluation visualizations generated successfully!")

if __name__ == '__main__':
    generate_evaluation_visualizations()
