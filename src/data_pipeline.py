import os
import datetime
import numpy as np
import pandas as pd
import joblib

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def load_and_preprocess_raw(excel_path=None):
    """
    Loads the raw Excel transactions and reconstructs exact timestamps, dates, and revenue.
    """
    if excel_path is None:
        excel_path = os.path.join(PROJECT_DIR, 'Afficionado Coffee Roasters.xlsx')
        
    print(f"Loading raw transaction data from {excel_path}...")
    df = pd.read_excel(excel_path)
    
    # Extract hour and minute
    df['hour'] = df['transaction_time'].apply(
        lambda t: t.hour if isinstance(t, datetime.time) else int(str(t).split(':')[0])
    )
    df['minute'] = df['transaction_time'].apply(
        lambda t: t.minute if isinstance(t, datetime.time) else int(str(t).split(':')[1])
    )
    df['time_sec'] = df['hour'] * 3600 + df['minute'] * 60

    # Detect day boundaries by overnight time resets
    time_diff = df['time_sec'].diff()
    resets = df[time_diff < -1800] # Reset when time jumps backward by >= 30 mins

    day_nums = np.zeros(len(df), dtype=int)
    current_day = 1
    for i in range(len(df)):
        if i in resets.index:
            current_day += 1
        day_nums[i] = current_day
    df['day_num'] = day_nums

    start_date = pd.Timestamp('2025-01-01')
    df['transaction_date'] = df['day_num'].apply(lambda d: start_date + pd.Timedelta(days=d-1))
    
    # Construct exact hour timestamp
    df['datetime_hour'] = df.apply(
        lambda row: pd.Timestamp(
            year=row['transaction_date'].year,
            month=row['transaction_date'].month,
            day=row['transaction_date'].day,
            hour=row['hour']
        ),
        axis=1
    )
    
    df['revenue'] = df['transaction_qty'] * df['unit_price']
    print(f"Loaded {len(df)} transactions spanning {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()} ({current_day} days).")
    return df

def build_store_hourly_grid(df):
    """
    Aggregates transactions by [store_location, datetime_hour] on a complete continuous hourly grid
    and computes lag features, rolling statistics, and peak demand targets.
    """
    print("Building Store-Level Hourly Time-Series Grid...")
    stores = df['store_location'].unique()
    store_ids = df[['store_location', 'store_id']].drop_duplicates().set_index('store_location')['store_id'].to_dict()
    
    min_date = df['transaction_date'].min()
    max_date = df['transaction_date'].max()
    
    # Generate complete operating hours grid: 06:00 to 20:00 daily
    all_days = pd.date_range(min_date, max_date, freq='D')
    operating_hours = list(range(6, 21)) # 6 AM to 8 PM
    
    grid_records = []
    for d in all_days:
        for h in operating_hours:
            dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h)
            for store in stores:
                grid_records.append({
                    'datetime_hour': dt,
                    'store_location': store,
                    'store_id': store_ids[store]
                })
                
    grid_df = pd.DataFrame(grid_records)
    
    # Aggregate actual transactions
    agg_df = df.groupby(['store_location', 'datetime_hour']).agg(
        transaction_count=('transaction_id', 'count'),
        quantity_demanded=('transaction_qty', 'sum'),
        total_revenue=('revenue', 'sum'),
        avg_unit_price=('unit_price', 'mean')
    ).reset_index()
    
    # Merge with complete grid and fill zeros
    merged = pd.merge(grid_df, agg_df, on=['store_location', 'datetime_hour'], how='left')
    merged['transaction_count'] = merged['transaction_count'].fillna(0).astype(int)
    merged['quantity_demanded'] = merged['quantity_demanded'].fillna(0).astype(int)
    merged['total_revenue'] = merged['total_revenue'].fillna(0.0)
    merged['avg_unit_price'] = merged.groupby('store_location')['avg_unit_price'].transform(
        lambda s: s.fillna(s.mean())
    )
    
    # Sort chronologically by store and time
    merged = merged.sort_values(by=['store_location', 'datetime_hour']).reset_index(drop=True)
    
    # Temporal features
    merged['hour'] = merged['datetime_hour'].dt.hour
    merged['day_of_week'] = merged['datetime_hour'].dt.dayofweek
    merged['day_of_month'] = merged['datetime_hour'].dt.day
    merged['month'] = merged['datetime_hour'].dt.month
    merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
    
    # Cyclical encodings
    merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24.0)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24.0)
    merged['day_sin'] = np.sin(2 * np.pi * merged['day_of_week'] / 7.0)
    merged['day_cos'] = np.cos(2 * np.pi * merged['day_of_week'] / 7.0)
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12.0)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12.0)
    
    # Lag and Rolling features per store
    store_dfs = []
    for store, s_df in merged.groupby('store_location'):
        s_df = s_df.copy().sort_values('datetime_hour').reset_index(drop=True)
        
        # Operational hourly lags (in sequence of operating slots: 15 hours per day)
        s_df['lag_1h_qty'] = s_df['quantity_demanded'].shift(1)
        s_df['lag_2h_qty'] = s_df['quantity_demanded'].shift(2)
        s_df['lag_15h_qty'] = s_df['quantity_demanded'].shift(15)  # same hour yesterday
        s_df['lag_105h_qty'] = s_df['quantity_demanded'].shift(105) # same hour last week (7 * 15)
        
        s_df['lag_1h_rev'] = s_df['total_revenue'].shift(1)
        s_df['lag_15h_rev'] = s_df['total_revenue'].shift(15)
        
        # Rolling statistics (excluding current hour to avoid data leakage)
        shifted_qty = s_df['quantity_demanded'].shift(1)
        s_df['rolling_mean_3h_qty'] = shifted_qty.rolling(3, min_periods=1).mean()
        s_df['rolling_mean_15h_qty'] = shifted_qty.rolling(15, min_periods=1).mean() # past 1 day
        s_df['rolling_mean_7d_qty'] = shifted_qty.rolling(105, min_periods=1).mean() # past 7 days
        s_df['rolling_std_15h_qty'] = shifted_qty.rolling(15, min_periods=1).std().fillna(0)
        
        # Target for Peak Demand Classification:
        # Peak threshold: 75th percentile of non-zero quantity for that store
        p75 = s_df[s_df['quantity_demanded'] > 0]['quantity_demanded'].quantile(0.75)
        p50 = s_df[s_df['quantity_demanded'] > 0]['quantity_demanded'].quantile(0.50)
        
        # Binary Peak Indicator (Rush Hour Surge)
        s_df['is_peak_demand'] = (s_df['quantity_demanded'] >= p75).astype(int)
        
        # Multi-class Demand Tier: 0: Low (Off-Peak), 1: Moderate (Normal), 2: High (Peak Surge)
        s_df['demand_tier'] = 0
        s_df.loc[s_df['quantity_demanded'] >= p50, 'demand_tier'] = 1
        s_df.loc[s_df['quantity_demanded'] >= p75, 'demand_tier'] = 2
        
        # Fill remaining NA lags with backfill/forward-fill
        s_df = s_df.bfill().ffill()
        store_dfs.append(s_df)
        
    final_store_df = pd.concat(store_dfs, ignore_index=True)
    out_path = os.path.join(DATA_DIR, 'store_hourly_features.csv')
    final_store_df.to_csv(out_path, index=False)
    print(f"Saved store hourly feature dataset to {out_path} ({final_store_df.shape})")
    return final_store_df

def build_category_hourly_grid(df):
    """
    Aggregates transactions by [store_location, product_category, datetime_hour]
    for category-level demand forecasting.
    """
    print("Building Category-Level Hourly Time-Series Grid...")
    stores = df['store_location'].unique()
    categories = df['product_category'].unique()
    min_date = df['transaction_date'].min()
    max_date = df['transaction_date'].max()
    
    all_days = pd.date_range(min_date, max_date, freq='D')
    operating_hours = list(range(6, 21))
    
    grid_records = []
    for d in all_days:
        for h in operating_hours:
            dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h)
            for store in stores:
                for cat in categories:
                    grid_records.append({
                        'datetime_hour': dt,
                        'store_location': store,
                        'product_category': cat
                    })
    grid_df = pd.DataFrame(grid_records)
    
    agg_df = df.groupby(['store_location', 'product_category', 'datetime_hour']).agg(
        quantity_demanded=('transaction_qty', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()
    
    merged = pd.merge(grid_df, agg_df, on=['store_location', 'product_category', 'datetime_hour'], how='left')
    merged['quantity_demanded'] = merged['quantity_demanded'].fillna(0).astype(int)
    merged['total_revenue'] = merged['total_revenue'].fillna(0.0)
    
    # Sort chronologically
    merged = merged.sort_values(by=['store_location', 'product_category', 'datetime_hour']).reset_index(drop=True)
    
    merged['hour'] = merged['datetime_hour'].dt.hour
    merged['day_of_week'] = merged['datetime_hour'].dt.dayofweek
    merged['month'] = merged['datetime_hour'].dt.month
    merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
    
    merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24.0)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24.0)
    merged['day_sin'] = np.sin(2 * np.pi * merged['day_of_week'] / 7.0)
    merged['day_cos'] = np.cos(2 * np.pi * merged['day_of_week'] / 7.0)
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12.0)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12.0)
    
    cat_dfs = []
    for (store, cat), sc_df in merged.groupby(['store_location', 'product_category']):
        sc_df = sc_df.copy().sort_values('datetime_hour').reset_index(drop=True)
        shifted_qty = sc_df['quantity_demanded'].shift(1)
        sc_df['lag_1h_qty'] = shifted_qty
        sc_df['lag_15h_qty'] = sc_df['quantity_demanded'].shift(15)
        sc_df['lag_105h_qty'] = sc_df['quantity_demanded'].shift(105)
        sc_df['rolling_mean_3h_qty'] = shifted_qty.rolling(3, min_periods=1).mean()
        sc_df['rolling_mean_15h_qty'] = shifted_qty.rolling(15, min_periods=1).mean()
        sc_df = sc_df.bfill().ffill()
        cat_dfs.append(sc_df)
        
    final_cat_df = pd.concat(cat_dfs, ignore_index=True)
    out_path = os.path.join(DATA_DIR, 'category_hourly_features.csv')
    final_cat_df.to_csv(out_path, index=False)
    print(f"Saved category hourly feature dataset to {out_path} ({final_cat_df.shape})")
    return final_cat_df

if __name__ == '__main__':
    raw_df = load_and_preprocess_raw()
    store_features = build_store_hourly_grid(raw_df)
    cat_features = build_category_hourly_grid(raw_df)
    print("Data processing & feature engineering completed successfully!")
