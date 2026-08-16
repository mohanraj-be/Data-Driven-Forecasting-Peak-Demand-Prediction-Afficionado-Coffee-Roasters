import argparse
import json
import pandas as pd
from src.predictor import AfficionadoDemandPredictor

def main():
    parser = argparse.ArgumentParser(
        description="Afficionado Coffee Roasters ML Demand & Peak Rush Forecaster"
    )
    parser.add_argument(
        '--store', type=str, default="Hell's Kitchen",
        choices=["Hell's Kitchen", "Lower Manhattan", "Astoria"],
        help="Store location"
    )
    parser.add_argument(
        '--date', type=str, default="2025-07-01",
        help="Date for prediction (YYYY-MM-DD)"
    )
    parser.add_argument(
        '--hour', type=int, default=9,
        help="Hour of the day (6 to 20)"
    )
    parser.add_argument(
        '--mode', type=str, default="hour",
        choices=["hour", "day_schedule", "category_breakdown"],
        help="Prediction mode"
    )
    
    args = parser.parse_args()
    predictor = AfficionadoDemandPredictor()
    
    dt_str = f"{args.date} {args.hour:02d}:00:00"
    
    print("=" * 70)
    print(f"AFFICIONADO COFFEE ROASTERS - ML FORECASTING ENGINE")
    print(f"Store: {args.store} | Date: {args.date} | Hour: {args.hour}:00")
    print("=" * 70)
    
    if args.mode == 'hour':
        result = predictor.predict_hour(args.store, dt_str)
        print("\n--- Hourly Demand & Rush Prediction ---")
        for k, v in result.items():
            print(f"  • {k:25s}: {v}")
            
    elif args.mode == 'day_schedule':
        schedule = predictor.predict_day_schedule(args.store, args.date)
        print("\n--- Full Day Operational Schedule ---")
        print(schedule[[
            'hour', 'predicted_quantity', 'predicted_revenue', 
            'peak_rush_probability', 'demand_tier', 'total_recommended_staff'
        ]].to_string(index=False))
        
    elif args.mode == 'category_breakdown':
        breakdown = predictor.predict_category_breakdown(args.store, dt_str)
        print("\n--- Product Category Demand Breakdown ---")
        print(breakdown.to_string(index=False))

if __name__ == '__main__':
    main()
