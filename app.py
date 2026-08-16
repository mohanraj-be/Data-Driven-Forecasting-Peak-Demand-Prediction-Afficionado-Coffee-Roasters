import os
import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.predictor import AfficionadoDemandPredictor

# Page Configuration
st.set_page_config(
    page_title="Afficionado Coffee Roasters | Demand & Peak Rush ML Platform",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3b2314;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f5539;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #fbf8f5;
        border: 1px solid #ede0d4;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .rush-badge {
        background-color: #e63946;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .normal-badge {
        background-color: #2a9d8f;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .offpeak-badge {
        background-color: #6c757d;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
FIGS_DIR = os.path.join(PROJECT_DIR, 'reports', 'figures')

@st.cache_resource
def load_predictor():
    return AfficionadoDemandPredictor(models_dir=MODELS_DIR)

@st.cache_data
def load_datasets():
    store_df = pd.read_csv(os.path.join(DATA_DIR, 'store_hourly_features.csv'))
    store_df['datetime_hour'] = pd.to_datetime(store_df['datetime_hour'])
    
    with open(os.path.join(MODELS_DIR, 'training_metrics.json'), 'r') as f:
        metrics = json.load(f)
        
    fi_df = pd.read_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'))
    return store_df, metrics, fi_df

try:
    predictor = load_predictor()
    store_df, metrics, fi_df = load_datasets()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=300&q=80", use_container_width=True)
st.sidebar.title("☕ Afficionado Roasters")
st.sidebar.markdown("**Data-Driven ML Demand Forecaster**")

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive KPI Overview",
        "🔮 Real-Time Demand Forecaster",
        "⚡ Peak Demand & Staffing Optimizer",
        "📦 Category Demand Breakdown",
        "🧪 Model Benchmark & Diagnostics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏪 Quick Filter")
selected_store = st.sidebar.selectbox(
    "Select Retail Store",
    ["Hell's Kitchen", "Lower Manhattan", "Astoria"]
)

# -------------------------------------------------------------------------------------------------
# 1. EXECUTIVE KPI OVERVIEW
# -------------------------------------------------------------------------------------------------
if nav_page == "📊 Executive KPI Overview":
    st.markdown('<div class="main-header">📊 Executive Performance & Demand Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Historical insights and trend patterns across NYC retail operations (Jan - Jun 2025)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    total_rev = store_df['total_revenue'].sum()
    total_qty = store_df['quantity_demanded'].sum()
    total_tx = store_df['transaction_count'].sum()
    avg_order_val = total_rev / max(total_tx, 1)
    
    with col1:
        st.metric("Total System Revenue", f"${total_rev:,.2f}", "+104% (Jan vs Jun)")
    with col2:
        st.metric("Total Items Demanded", f"{total_qty:,} units", "+105% Volume Growth")
    with col3:
        st.metric("Total Transactions", f"{total_tx:,} orders", "823 orders/day")
    with col4:
        st.metric("Avg Transaction Value", f"${avg_order_val:.2f}", "3 retail locations")
        
    st.markdown("---")
    
    # Monthly Growth Trend
    st.subheader("📈 Monthly Revenue & Volume Expansion")
    monthly_trend = store_df.groupby(['month', 'store_location'])[['total_revenue', 'quantity_demanded']].sum().reset_index()
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun'}
    monthly_trend['Month'] = monthly_trend['month'].map(month_names)
    
    chart_monthly = alt.Chart(monthly_trend).mark_bar().encode(
        x=alt.X('Month:N', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], title="Month (2025)"),
        y=alt.Y('total_revenue:Q', title="Total Revenue ($)"),
        color=alt.Color('store_location:N', scale=alt.Scale(scheme='tableau10'), title="Store Location"),
        tooltip=['Month', 'store_location', alt.Tooltip('total_revenue:Q', format='$,.2f'), 'quantity_demanded:Q']
    ).properties(height=350)
    st.altair_chart(chart_monthly, use_container_width=True)
    
    # Hourly Profile per Store
    st.subheader("⏰ Store Operating Hours & Hourly Demand Rhythm")
    hourly_store = store_df.groupby(['store_location', 'hour'])['quantity_demanded'].mean().reset_index()
    
    chart_hourly = alt.Chart(hourly_store).mark_line(point=True).encode(
        x=alt.X('hour:O', title="Operating Hour (24h)"),
        y=alt.Y('quantity_demanded:Q', title="Average Hourly Units Demanded"),
        color=alt.Color('store_location:N', title="Store Location"),
        tooltip=['store_location', 'hour', alt.Tooltip('quantity_demanded:Q', format='.1f')]
    ).properties(height=350)
    st.altair_chart(chart_hourly, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# 2. REAL-TIME DEMAND FORECASTER
# -------------------------------------------------------------------------------------------------
elif nav_page == "🔮 Real-Time Demand Forecaster":
    st.markdown('<div class="main-header">🔮 Real-Time Demand & Revenue Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive ML simulation engine powered by trained LightGBM and XGBoost ensembles</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sim_date = st.date_input("Target Forecast Date", datetime.date(2025, 7, 1))
    with c2:
        sim_hour = st.slider("Operating Hour", min_value=6, max_value=20, value=9, format="%d:00")
    with c3:
        target_store = st.selectbox("Store Location", ["Hell's Kitchen", "Lower Manhattan", "Astoria"], index=0)
        
    dt_target = pd.Timestamp(year=sim_date.year, month=sim_date.month, day=sim_date.day, hour=sim_hour)
    res = predictor.predict_hour(target_store, dt_target)
    
    st.markdown("---")
    st.subheader(f"📍 Prediction for {target_store} on {sim_date.strftime('%A, %B %d, %Y')} at {sim_hour:02d}:00")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Predicted Demand Volume", f"{res['predicted_quantity']} units", "Hourly Total")
    with k2:
        st.metric("Expected Revenue", f"${res['predicted_revenue']:.2f}", f"Avg ${(res['predicted_revenue']/max(res['predicted_quantity'], 1)):.2f}/unit")
    with k3:
        rush_pct = res['peak_rush_probability'] * 100
        st.metric("Peak Rush Probability", f"{rush_pct:.1f}%", "Rush Alert" if res['is_peak_demand'] else "Normal")
    with k4:
        tier_label = res['demand_tier']
        st.metric("Demand Classification", tier_label, f"Staff: {res['total_recommended_staff']} baristas/staff")
        
    st.info(f"💡 **Operational Guidance:** {res['operational_guidance']}")
    
    # 24-Hour Forecast Curve for the day
    st.markdown("---")
    st.subheader("📅 Full Day 24-Hour Continuous Trajectory")
    day_sched = predictor.predict_day_schedule(target_store, sim_date)
    
    chart_day = alt.Chart(day_sched).mark_area(
        color='#457b9d',
        opacity=0.3,
        line={'color': '#1d3557', 'width': 2.5}
    ).encode(
        x=alt.X('hour:O', title="Hour of Day"),
        y=alt.Y('predicted_quantity:Q', title="Predicted Units Sold"),
        tooltip=['hour', 'predicted_quantity', 'predicted_revenue', 'peak_rush_probability', 'demand_tier']
    ).properties(height=320)
    
    # Highlight current selected hour
    sel_point = day_sched[day_sched['hour'] == sim_hour]
    chart_point = alt.Chart(sel_point).mark_circle(size=150, color='#e63946').encode(
        x='hour:O',
        y='predicted_quantity:Q',
        tooltip=['hour', 'predicted_quantity', 'predicted_revenue']
    )
    
    st.altair_chart(chart_day + chart_point, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# 3. PEAK DEMAND & STAFFING OPTIMIZER
# -------------------------------------------------------------------------------------------------
elif nav_page == "⚡ Peak Demand & Staffing Optimizer":
    st.markdown('<div class="main-header">⚡ Peak Demand & Smart Staffing Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predictive rush-hour alerts, labor allocation, and batch prep optimization</div>', unsafe_allow_html=True)
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        opt_store = st.selectbox("Select Store for Staffing Plan", ["Hell's Kitchen", "Lower Manhattan", "Astoria"], index=0)
    with opt_col2:
        opt_date = st.date_input("Shift Date", datetime.date(2025, 7, 1), key="staff_date")
        
    schedule = predictor.predict_day_schedule(opt_store, opt_date)
    
    # Staffing Curve
    st.subheader(f"👥 Recommended Hourly Labor Allocation – {opt_store}")
    staff_chart_df = schedule[['hour', 'recommended_baristas', 'recommended_cashiers', 'total_recommended_staff']].melt(
        id_vars=['hour'],
        value_vars=['recommended_baristas', 'recommended_cashiers'],
        var_name='Role',
        value_name='Staff_Count'
    )
    staff_chart_df['Role'] = staff_chart_df['Role'].map({'recommended_baristas': 'Baristas', 'recommended_cashiers': 'Cashiers/Support'})
    
    staff_bar = alt.Chart(staff_chart_df).mark_bar().encode(
        x=alt.X('hour:O', title="Hour of Day"),
        y=alt.Y('Staff_Count:Q', title="Recommended Headcount"),
        color=alt.Color('Role:N', scale=alt.Scale(range=['#2a9d8f', '#e76f51'])),
        tooltip=['hour', 'Role', 'Staff_Count']
    ).properties(height=300)
    
    st.altair_chart(staff_bar, use_container_width=True)
    
    # Schedule Table
    st.subheader("📋 Operational Shift & Batch Prep Schedule")
    
    def style_tier(val):
        if 'Rush' in str(val):
            return 'background-color: #ffcccc; font-weight: bold; color: #900;'
        elif 'Normal' in str(val):
            return 'background-color: #d8f3dc; color: #1b4332;'
        else:
            return 'background-color: #f0f0f0; color: #555;'
            
    display_sched = schedule[[
        'hour', 'predicted_quantity', 'predicted_revenue', 
        'peak_rush_probability', 'demand_tier', 'recommended_baristas', 
        'recommended_cashiers', 'total_recommended_staff', 'operational_guidance'
    ]].rename(columns={
        'hour': 'Hour',
        'predicted_quantity': 'Pred Units',
        'predicted_revenue': 'Pred Rev ($)',
        'peak_rush_probability': 'Rush Prob',
        'demand_tier': 'Demand Tier',
        'recommended_baristas': 'Baristas',
        'recommended_cashiers': 'Cashiers',
        'total_recommended_staff': 'Total Staff',
        'operational_guidance': 'Kitchen Guidance'
    })
    
    st.dataframe(
        display_sched.style.applymap(style_tier, subset=['Demand Tier']).format({
            'Pred Rev ($)': '${:.2f}',
            'Rush Prob': '{:.1%}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Cross-Store Heatmap
    st.markdown("---")
    st.subheader("🗺️ Store-Wide Rush Hour Probability Matrix")
    if os.path.exists(os.path.join(FIGS_DIR, 'store_peak_heatmap.png')):
        st.image(os.path.join(FIGS_DIR, 'store_peak_heatmap.png'), caption="Multi-Store Hourly Peak Demand Probability Heatmap", use_container_width=True)

# -------------------------------------------------------------------------------------------------
# 4. CATEGORY DEMAND BREAKDOWN
# -------------------------------------------------------------------------------------------------
elif nav_page == "📦 Category Demand Breakdown":
    st.markdown('<div class="main-header">📦 Product Category Demand Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fine-grained machine learning predictions across 9 product categories for inventory planning</div>', unsafe_allow_html=True)
    
    cat_c1, cat_c2, cat_c3 = st.columns(3)
    with cat_c1:
        c_store = st.selectbox("Store", ["Hell's Kitchen", "Lower Manhattan", "Astoria"], index=0, key="c_store")
    with cat_c2:
        c_date = st.date_input("Date", datetime.date(2025, 7, 1), key="c_date")
    with cat_c3:
        c_hour = st.slider("Hour", 6, 20, 9, key="c_hour", format="%d:00")
        
    dt_cat = pd.Timestamp(year=c_date.year, month=c_date.month, day=c_date.day, hour=c_hour)
    breakdown_df = predictor.predict_category_breakdown(c_store, dt_cat)
    
    col_chart, col_tbl = st.columns([3, 2])
    with col_chart:
        st.subheader("📊 Forecasted Units by Product Category")
        cat_chart = alt.Chart(breakdown_df).mark_bar().encode(
            x=alt.X('predicted_units:Q', title="Predicted Quantity (Units)"),
            y=alt.Y('product_category:N', sort='-x', title="Product Category"),
            color=alt.Color('predicted_units:Q', scale=alt.Scale(scheme='warmgreys')),
            tooltip=['product_category', 'predicted_units']
        ).properties(height=350)
        st.altair_chart(cat_chart, use_container_width=True)
        
    with col_tbl:
        st.subheader("📋 Inventory Prep List")
        st.dataframe(
            breakdown_df.rename(columns={'product_category': 'Category', 'predicted_units': 'Predicted Demand (Units)'}),
            use_container_width=True,
            hide_index=True
        )

# -------------------------------------------------------------------------------------------------
# 5. MODEL BENCHMARK & DIAGNOSTICS
# -------------------------------------------------------------------------------------------------
elif nav_page == "🧪 Model Benchmark & Diagnostics":
    st.markdown('<div class="main-header">🧪 Model Performance & Benchmarking Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics on chronological forward test set (June 2025) across regression and classification models</div>', unsafe_allow_html=True)
    
    st.subheader("🏆 Model Leaderboard: Hourly Quantity Demand Forecaster")
    
    reg_summary = pd.DataFrame(metrics['hourly_quantity_forecasting']).T.reset_index().rename(columns={'index': 'Model'})
    reg_summary['RMSE'] = reg_summary['RMSE'].map('{:.3f}'.format)
    reg_summary['MAE'] = reg_summary['MAE'].map('{:.3f}'.format)
    reg_summary['R2'] = reg_summary['R2'].map('{:.3f}'.format)
    reg_summary['WAPE'] = reg_summary['WAPE'].map('{:.2f}%'.format)
    
    st.dataframe(reg_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("⚡ Peak Demand Classifier Performance (XGBoost)")
    peak_metrics = metrics['peak_demand_classification']['Binary_Peak']
    
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Test Accuracy", f"{peak_metrics['Accuracy']*100:.1f}%")
    with p2:
        st.metric("Precision", f"{peak_metrics['Precision']*100:.1f}%")
    with p3:
        st.metric("Recall (Sensitivity)", f"{peak_metrics['Recall']*100:.1f}%")
    with p4:
        st.metric("F1-Score", f"{peak_metrics['F1']:.3f}")
    with p5:
        st.metric("ROC-AUC Score", f"{peak_metrics['ROC_AUC']:.3f}")
        
    st.markdown("---")
    st.subheader("📈 Diagnostic Figures & Time-Series Evaluation")
    
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        if os.path.exists(os.path.join(FIGS_DIR, 'model_benchmark_comparison.png')):
            st.image(os.path.join(FIGS_DIR, 'model_benchmark_comparison.png'), caption="Model Benchmark Comparison", use_container_width=True)
        if os.path.exists(os.path.join(FIGS_DIR, 'feature_importance_ranking.png')):
            st.image(os.path.join(FIGS_DIR, 'feature_importance_ranking.png'), caption="Feature Importance Ranking", use_container_width=True)
            
    with fig_col2:
        if os.path.exists(os.path.join(FIGS_DIR, 'peak_demand_classification_diagnostics.png')):
            st.image(os.path.join(FIGS_DIR, 'peak_demand_classification_diagnostics.png'), caption="Peak Demand Diagnostics & ROC Curve", use_container_width=True)
        if os.path.exists(os.path.join(FIGS_DIR, 'actual_vs_forecast_timeseries.png')):
            st.image(os.path.join(FIGS_DIR, 'actual_vs_forecast_timeseries.png'), caption="Actual vs Forecast Time Series (Test Period)", use_container_width=True)
