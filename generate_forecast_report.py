import os
import datetime
import pandas as pd
import numpy as np
from src.predictor import AfficionadoDemandPredictor

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_forward_july_projections():
    print("Generating comprehensive July 2025 operational forecast report...")
    predictor = AfficionadoDemandPredictor()
    
    stores = ["Hell's Kitchen", "Lower Manhattan", "Astoria"]
    july_days = pd.date_range("2025-07-01", "2025-07-31", freq='D')
    operating_hours = list(range(6, 21)) # 6 AM to 8 PM
    
    hourly_records = []
    category_records = []
    
    for d in july_days:
        for store in stores:
            for h in operating_hours:
                dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h)
                res = predictor.predict_hour(store, dt)
                hourly_records.append({
                    'Date': d.strftime('%Y-%m-%d'),
                    'Day_of_Week': d.strftime('%A'),
                    'Hour': h,
                    'Store_Location': store,
                    'Predicted_Units': res['predicted_quantity'],
                    'Predicted_Revenue': res['predicted_revenue'],
                    'Peak_Rush_Prob': res['peak_rush_probability'],
                    'Demand_Tier': res['demand_tier'],
                    'Recommended_Baristas': res['recommended_baristas'],
                    'Recommended_Cashiers': res['recommended_cashiers'],
                    'Total_Staff_Needed': res['total_recommended_staff'],
                    'Kitchen_Guidance': res['operational_guidance']
                })
                
                # Category breakdown sample (at daily lunch and peak hours)
                if h in [8, 9, 10, 14]:
                    cat_res = predictor.predict_category_breakdown(store, dt)
                    for _, row in cat_res.iterrows():
                        category_records.append({
                            'Date': d.strftime('%Y-%m-%d'),
                            'Store_Location': store,
                            'Hour': h,
                            'Product_Category': row['product_category'],
                            'Predicted_Units': row['predicted_units']
                        })
                        
    hourly_df = pd.DataFrame(hourly_records)
    cat_df = pd.DataFrame(category_records)
    
    # Summary by Store
    store_summary = hourly_df.groupby('Store_Location').agg(
        Total_Forecasted_Units=('Predicted_Units', 'sum'),
        Total_Forecasted_Revenue=('Predicted_Revenue', 'sum'),
        Avg_Daily_Revenue=('Predicted_Revenue', lambda x: x.sum() / 31.0),
        Peak_Rush_Hours=('Demand_Tier', lambda x: (x == 'Rush Surge (High)').sum()),
        Avg_Hourly_Staff=('Total_Staff_Needed', 'mean')
    ).reset_index()
    
    # Category Purchasing Guidance (Monthly Estimate)
    cat_monthly = cat_df.groupby(['Store_Location', 'Product_Category'])['Predicted_Units'].sum().reset_index()
    
    # Save to Excel Multi-sheet Report
    out_excel = os.path.join(REPORTS_DIR, 'July_2025_Demand_Forecast_Report.xlsx')
    with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
        store_summary.to_excel(writer, sheet_name='Store Executive Summary', index=False)
        hourly_df.head(5000).to_excel(writer, sheet_name='Hourly Shift Forecasts', index=False)
        cat_monthly.to_excel(writer, sheet_name='Category Inventory Needs', index=False)
        
    print(f"Successfully generated forward forecast report: {out_excel}")
    print("\n--- July 2025 Store Summary ---")
    print(store_summary.to_string(index=False))
    return out_excel

if __name__ == '__main__':
    generate_forward_july_projections()
