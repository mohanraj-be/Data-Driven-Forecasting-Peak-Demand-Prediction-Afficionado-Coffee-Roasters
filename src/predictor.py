import os
import datetime
import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

class AfficionadoDemandPredictor:
    """
    Inference and decision support engine for Afficionado Coffee Roasters.
    Predicts demand volume, revenue, peak rush probability, staffing requirements,
    and category-level inventory needs.
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = MODELS_DIR
            
        self.models_dir = models_dir
        self.lgb_qty_model = joblib.load(os.path.join(models_dir, 'lgb_hourly_qty_model.joblib'))
        self.xgb_qty_model = joblib.load(os.path.join(models_dir, 'xgb_hourly_qty_model.joblib'))
        self.lgb_rev_model = joblib.load(os.path.join(models_dir, 'lgb_hourly_rev_model.joblib'))
        self.xgb_peak_model = joblib.load(os.path.join(models_dir, 'xgb_peak_classifier.joblib'))
        self.lgb_tier_model = joblib.load(os.path.join(models_dir, 'lgb_tier_classifier.joblib'))
        self.lgb_cat_model = joblib.load(os.path.join(models_dir, 'lgb_category_qty_model.joblib'))
        
        metadata = joblib.load(os.path.join(models_dir, 'encoders_and_scalers.joblib'))
        self.le_store = metadata['le_store']
        self.le_cat = metadata['le_cat']
        self.feature_cols = metadata['feature_cols']
        self.cat_feature_cols = metadata['cat_feature_cols']
        
        # Load historical baselines for fallback lags
        self.store_history = pd.read_csv(os.path.join(DATA_DIR, 'store_hourly_features.csv'))
        self.store_history['datetime_hour'] = pd.to_datetime(self.store_history['datetime_hour'])
        
        self.cat_history = pd.read_csv(os.path.join(DATA_DIR, 'category_hourly_features.csv'))
        self.cat_history['datetime_hour'] = pd.to_datetime(self.cat_history['datetime_hour'])
        
        # Historical store-hour average benchmarks
        self.store_hour_stats = self.store_history.groupby(['store_location', 'hour'])[['quantity_demanded', 'total_revenue']].mean().reset_index()

    def _build_features_for_slot(self, store_location, dt):
        """Constructs input feature vector for a specific store and datetime slot."""
        store_encoded = self.le_store.transform([store_location])[0]
        hour = dt.hour
        day_of_week = dt.dayofweek
        day_of_month = dt.day
        month = dt.month
        is_weekend = int(day_of_week >= 5)
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)
        
        # Retrieve historical pattern for lags
        hist_match = self.store_hour_stats[
            (self.store_hour_stats['store_location'] == store_location) & 
            (self.store_hour_stats['hour'] == hour)
        ]
        if not hist_match.empty:
            hist_qty = hist_match['quantity_demanded'].values[0]
            hist_rev = hist_match['total_revenue'].values[0]
        else:
            hist_qty = 30.0
            hist_rev = 100.0
            
        lag_1h_qty = hist_qty
        lag_2h_qty = hist_qty
        lag_15h_qty = hist_qty
        lag_105h_qty = hist_qty
        rolling_mean_3h_qty = hist_qty
        rolling_mean_15h_qty = hist_qty
        rolling_mean_7d_qty = hist_qty
        rolling_std_15h_qty = hist_qty * 0.25
        lag_1h_rev = hist_rev
        lag_15h_rev = hist_rev
        
        feat_dict = {
            'store_encoded': store_encoded,
            'hour': hour,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'is_weekend': is_weekend,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'lag_1h_qty': lag_1h_qty,
            'lag_2h_qty': lag_2h_qty,
            'lag_15h_qty': lag_15h_qty,
            'lag_105h_qty': lag_105h_qty,
            'rolling_mean_3h_qty': rolling_mean_3h_qty,
            'rolling_mean_15h_qty': rolling_mean_15h_qty,
            'rolling_mean_7d_qty': rolling_mean_7d_qty,
            'rolling_std_15h_qty': rolling_std_15h_qty,
            'lag_1h_rev': lag_1h_rev,
            'lag_15h_rev': lag_15h_rev
        }
        return pd.DataFrame([feat_dict])[self.feature_cols]

    def predict_hour(self, store_location, dt):
        """
        Predicts demand, revenue, rush probability, and optimal staffing recommendations
        for a single store and datetime.
        """
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
            
        X = self._build_features_for_slot(store_location, dt)
        
        pred_qty_lgb = float(max(0, self.lgb_qty_model.predict(X)[0]))
        pred_qty_xgb = float(max(0, self.xgb_qty_model.predict(X)[0]))
        ensemble_qty = (pred_qty_lgb + pred_qty_xgb) / 2.0
        
        pred_rev = float(max(0, self.lgb_rev_model.predict(X)[0]))
        
        peak_prob = float(self.xgb_peak_model.predict_proba(X)[0][1])
        is_peak = bool(peak_prob >= 0.50)
        
        tier_pred = int(self.lgb_tier_model.predict(X)[0])
        tier_names = {0: "Off-Peak (Low)", 1: "Normal (Moderate)", 2: "Rush Surge (High)"}
        demand_tier_label = tier_names.get(tier_pred, "Normal")
        
        # Staffing & Batch Brew recommendation logic:
        # Standard capacity: 1 barista handles ~25-30 drinks/hour
        # 1 cashier handles ~40 orders/hour
        recommended_baristas = max(1, int(np.ceil(ensemble_qty / 28.0)))
        recommended_cashiers = max(1, int(np.ceil(ensemble_qty / 45.0)))
        total_staff = recommended_baristas + recommended_cashiers
        
        if ensemble_qty >= 60:
            prep_advice = "Heavy Morning Rush: Pre-grind espresso beans, prep 2 extra batch brewers, stage milk pitchers."
        elif ensemble_qty >= 35:
            prep_advice = "Moderate Demand: Maintain standard 1-batch brew rotation, restock cups and lids."
        else:
            prep_advice = "Light Baseline: On-demand brewing, conduct cleaning/inventory restock."
            
        return {
            'store_location': store_location,
            'datetime': dt.strftime('%Y-%m-%d %H:%M'),
            'hour': dt.hour,
            'predicted_quantity': round(ensemble_qty, 1),
            'predicted_revenue': round(pred_rev, 2),
            'peak_rush_probability': round(peak_prob, 3),
            'is_peak_demand': is_peak,
            'demand_tier': demand_tier_label,
            'recommended_baristas': recommended_baristas,
            'recommended_cashiers': recommended_cashiers,
            'total_recommended_staff': total_staff,
            'operational_guidance': prep_advice
        }

    def predict_day_schedule(self, store_location, target_date):
        """
        Generates complete daily 6:00 AM - 8:00 PM operational demand & staffing schedule.
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        elif isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()
            
        operating_hours = list(range(6, 21))
        schedule = []
        
        for h in operating_hours:
            dt = pd.Timestamp(year=target_date.year, month=target_date.month, day=target_date.day, hour=h)
            res = self.predict_hour(store_location, dt)
            schedule.append(res)
            
        schedule_df = pd.DataFrame(schedule)
        return schedule_df

    def predict_category_breakdown(self, store_location, dt):
        """
        Predicts item quantity demand across all 9 product categories for a specific hour.
        """
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
            
        categories = self.le_cat.classes_
        store_encoded = self.le_store.transform([store_location])[0]
        
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month
        is_weekend = int(day_of_week >= 5)
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)
        
        cat_rows = []
        for cat in categories:
            cat_encoded = self.le_cat.transform([cat])[0]
            
            # Historical fallback
            hist_match = self.cat_history[
                (self.cat_history['store_location'] == store_location) &
                (self.cat_history['product_category'] == cat) &
                (self.cat_history['hour'] == hour)
            ]
            hist_qty = hist_match['quantity_demanded'].mean() if not hist_match.empty else 5.0
            if np.isnan(hist_qty):
                hist_qty = 5.0
                
            cat_rows.append({
                'store_encoded': store_encoded,
                'category_encoded': cat_encoded,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': is_weekend,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'day_sin': day_sin,
                'day_cos': day_cos,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'lag_1h_qty': hist_qty,
                'lag_15h_qty': hist_qty,
                'lag_105h_qty': hist_qty,
                'rolling_mean_3h_qty': hist_qty,
                'rolling_mean_15h_qty': hist_qty
            })
            
        X_cat = pd.DataFrame(cat_rows)[self.cat_feature_cols]
        preds = self.lgb_cat_model.predict(X_cat)
        
        breakdown = []
        for cat, pred_qty in zip(categories, preds):
            breakdown.append({
                'product_category': cat,
                'predicted_units': max(0.0, round(float(pred_qty), 1))
            })
            
        df_breakdown = pd.DataFrame(breakdown).sort_values('predicted_units', ascending=False)
        return df_breakdown
