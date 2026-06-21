
# app.py
# Alishamart Supply Chain Hybrid AI Framework - Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import zipfile
import io
import requests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Alishamart Supply Chain AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1E3A5F, #2E5A8A);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .insight-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E3A5F;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FROM GITHUB ZIP FILE
# ============================================================================

@st.cache_data
def load_data_from_github():
    """
    Load data from GitHub ZIP file
    This function downloads the dataset from GitHub repository
    """
    try:
        # GitHub repository URL
        # ⚠️ Replace with your actual GitHub username and repository name
        github_username = "mdsalauddin-lab"  # Replace with your username
        repo_name = "supply-chain-framework-with-streamlit-dashboard"  # Replace with your repo name
        zip_filename = "alishamart_supply_chain_dataset.zip"
        
        # Construct the raw URL
        zip_url = f"https://raw.githubusercontent.com/mdsalauddin-lab/supply-chain-framework-with-streamlit-dashboard/main/alishamart_supply_chain_dataset.zip"
        
        st.info("📥 Downloading dataset from GitHub...")
        
        # Download the ZIP file
        response = requests.get(zip_url, timeout=30)
        
        if response.status_code == 200:
            # Extract ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                # Get the first CSV file from ZIP
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    st.error("❌ No CSV file found in the ZIP archive.")
                    return generate_sample_data()
                
                # Read the first CSV file
                with zip_ref.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)
                    
                    # Verify required columns exist
                    required_cols = ['order_date', 'quantity_ordered', 'phi_prediction']
                    if not all(col in df.columns for col in required_cols):
                        st.warning("⚠️ Required columns not found. Using sample data.")
                        return generate_sample_data()
                    
                    st.success("✅ Dataset loaded successfully from GitHub!")
                    return df
        else:
            st.error(f"❌ Failed to download from GitHub. Status code: {response.status_code}")
            st.info("ℹ️ Using sample data instead...")
            return generate_sample_data()
            
    except requests.exceptions.Timeout:
        st.error("❌ Connection timeout. Please check your internet connection.")
        return generate_sample_data()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
        return generate_sample_data()
    except zipfile.BadZipFile:
        st.error("❌ Invalid ZIP file. The file may be corrupted.")
        return generate_sample_data()
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {e}")
        return generate_sample_data()

@st.cache_data
def generate_sample_data():
    """
    Generate sample data if real data cannot be loaded
    This serves as a fallback for demo purposes
    """
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2026-06-21', freq='D')
    n = len(dates)
    
    # Base demand with seasonality and trend
    trend = np.linspace(450, 550, n)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n) / 30)
    weekly = 30 * np.sin(2 * np.pi * np.arange(n) / 7)
    
    quantity_ordered = trend + seasonal + weekly + np.random.normal(0, 20, n)
    quantity_ordered = np.maximum(quantity_ordered, 10).astype(int)
    
    # Generate features
    stock_on_hand = np.random.randint(100, 5000, n)
    reorder_point = np.random.randint(50, 200, n)
    safety_stock_level = np.random.randint(20, 80, n)
    stock_in_transit = np.random.randint(20, 1000, n)
    lead_time_days = np.random.randint(3, 15, n)
    
    # Categorical features
    product_categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Books']
    product_category = np.random.choice(product_categories, n)
    
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    brand_name = np.random.choice(brands, n)
    
    supplier_countries = ['USA', 'China', 'Germany', 'Brazil', 'India', 'Vietnam']
    supplier_country = np.random.choice(supplier_countries, n)
    
    incident_types = ['None', 'Delay', 'Damage', 'Customs', 'Labor', 'Natural']
    incident_type = np.random.choice(incident_types, n, p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05])
    
    impact_levels = ['Low', 'Medium', 'High']
    impact_level = np.random.choice(impact_levels, n, p=[0.5, 0.3, 0.2])
    
    order_statuses = ['Pending', 'Shipped', 'Delivered', 'Cancelled']
    order_status = np.random.choice(order_statuses, n, p=[0.1, 0.4, 0.45, 0.05])
    
    customs_statuses = ['Cleared', 'Pending', 'Held']
    customs_clearance_status = np.random.choice(customs_statuses, n, p=[0.7, 0.2, 0.1])
    
    financial_loss_amount = np.random.exponential(50, n)
    shipping_cost = np.random.uniform(100, 3500, n)
    supplier_rating = np.random.uniform(3.5, 5.0, n)
    
    # Create predictions
    prophet_pred = quantity_ordered + np.random.normal(0, 25, n)
    xgboost_pred = quantity_ordered + np.random.normal(0, 20, n)
    lstm_pred = quantity_ordered + np.random.normal(0, 18, n)
    phi_pred = quantity_ordered + np.random.normal(0, 12, n)
    
    df = pd.DataFrame({
        'order_date': dates,
        'quantity_ordered': quantity_ordered,
        'stock_on_hand': stock_on_hand,
        'reorder_point': reorder_point,
        'safety_stock_level': safety_stock_level,
        'stock_in_transit': stock_in_transit,
        'lead_time_days': lead_time_days,
        'product_category': product_category,
        'brand_name': brand_name,
        'supplier_country': supplier_country,
        'incident_type': incident_type,
        'impact_level': impact_level,
        'order_status': order_status,
        'customs_clearance_status': customs_clearance_status,
        'financial_loss_amount': financial_loss_amount,
        'shipping_cost': shipping_cost,
        'supplier_rating': supplier_rating,
        'prophet_prediction': prophet_pred,
        'xgboost_prediction': xgboost_pred,
        'lstm_prediction': lstm_pred,
        'phi_prediction': phi_pred
    })
    
    df['stock_utilization_ratio'] = df['stock_on_hand'] / df['safety_stock_level']
    df['reorder_utilization'] = df['stock_on_hand'] / df['reorder_point']
    df['lead_time_category'] = pd.cut(df['lead_time_days'], bins=[0, 6, 10, 20], labels=['Short', 'Medium', 'Long'])
    df['phi_error'] = df['quantity_ordered'] - df['phi_prediction']
    df['prophet_error'] = df['quantity_ordered'] - df['prophet_prediction']
    df['xgboost_error'] = df['quantity_ordered'] - df['xgboost_prediction']
    df['lstm_error'] = df['quantity_ordered'] - df['lstm_prediction']
    
    st.info("ℹ️ Using sample data (real dataset could not be loaded)")
    return df

@st.cache_data
def load_tournament_results():
    """
    Generate tournament results for model comparison
    This data represents the 5-fold cross-validation results
    """
    models = ['Prophet', 'XGBoost', 'LSTM', 'Phi-Hybrid']
    folds = [1, 2, 3, 4, 5]
    
    # RMSE values (lower is better) - from tournament results
    rmse_data = {
        'Prophet': [45.2, 47.1, 46.8, 44.9, 46.3],
        'XGBoost': [42.5, 43.8, 41.9, 44.1, 42.7],
        'LSTM': [43.1, 45.2, 42.6, 46.5, 44.8],
        'Phi-Hybrid': [38.4, 37.9, 39.2, 38.7, 38.1]
    }
    
    # R² values (higher is better)
    r2_data = {
        'Prophet': [0.72, 0.70, 0.71, 0.73, 0.71],
        'XGBoost': [0.78, 0.76, 0.79, 0.75, 0.77],
        'LSTM': [0.76, 0.74, 0.77, 0.72, 0.75],
        'Phi-Hybrid': [0.85, 0.86, 0.84, 0.85, 0.86]
    }
    
    # MAE values (lower is better)
    mae_data = {
        'Prophet': [32.1, 33.5, 32.8, 31.9, 32.7],
        'XGBoost': [29.8, 30.5, 29.4, 30.8, 29.9],
        'LSTM': [30.2, 31.8, 29.9, 32.1, 31.2],
        'Phi-Hybrid': [26.5, 26.1, 27.0, 26.8, 26.3]
    }
    
    results = []
    for model in models:
        for fold in folds:
            results.append({
                'Model': model,
                'Fold': fold,
                'RMSE': rmse_data[model][fold-1],
                'R²': r2_data[model][fold-1],
                'MAE': mae_data[model][fold-1]
            })
    
    return pd.DataFrame(results)

# ============================================================================
# LOAD DATA
# ============================================================================

# Try to load real data, fallback to sample data if fails
df = load_data_from_github()
tournament_df = load_tournament_results()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
st.sidebar.markdown("## 🎯 Dashboard Filters")

# Date Range Filter
min_date = df['order_date'].min()
max_date = df['order_date'].max()
date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Product Category Filter
categories = ['All'] + sorted(df['product_category'].unique().tolist())
selected_category = st.sidebar.selectbox("🏷️ Product Category", categories)

# Supplier Country Filter
countries = ['All'] + sorted(df['supplier_country'].unique().tolist())
selected_country = st.sidebar.selectbox("🌍 Supplier Country", countries)

# Incident Type Filter
incidents = ['All'] + sorted(df['incident_type'].unique().tolist())
selected_incident = st.sidebar.selectbox("⚠️ Incident Type", incidents)

# Lead Time Category Filter
lead_times = ['All'] + sorted(df['lead_time_category'].unique().tolist())
selected_lead_time = st.sidebar.selectbox("📦 Lead Time Category", lead_times)

# Apply filters to data
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['order_date'] >= pd.to_datetime(date_range[0])) &
        (filtered_df['order_date'] <= pd.to_datetime(date_range[1]))
    ]
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['product_category'] == selected_category]
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['supplier_country'] == selected_country]
if selected_incident != 'All':
    filtered_df = filtered_df[filtered_df['incident_type'] == selected_incident]
if selected_lead_time != 'All':
    filtered_df = filtered_df[filtered_df['lead_time_category'] == selected_lead_time]

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-header">🏭 Alishamart Supply Chain AI Dashboard</div>', unsafe_allow_html=True)

# ============================================================================
# TOP ROW - KEY METRICS
# ============================================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_demand = filtered_df['quantity_ordered'].mean()
    st.metric("📊 Avg Daily Demand", f"{avg_demand:.0f} units")

with col2:
    avg_stock = filtered_df['stock_on_hand'].mean()
    st.metric("📦 Avg Stock on Hand", f"{avg_stock:.0f} units")

with col3:
    phi_rmse = np.sqrt(((filtered_df['quantity_ordered'] - filtered_df['phi_prediction']) ** 2).mean())
    st.metric("🎯 Phi-Hybrid RMSE", f"{phi_rmse:.1f}")

with col4:
    stockout_risk = (filtered_df['quantity_ordered'] > filtered_df['phi_prediction']).mean() * 100
    st.metric("⚠️ Stock-out Risk", f"{stockout_risk:.1f}%")

with col5:
    avg_lead_time = filtered_df['lead_time_days'].mean()
    st.metric("⏱️ Avg Lead Time", f"{avg_lead_time:.1f} days")

# ============================================================================
# ROW 1: DEMAND FORECAST
# ============================================================================
st.markdown("---")
st.markdown("### 📈 Demand Forecast (Actual vs Predicted)")

# Aggregate data by date for time series
ts_df = filtered_df.groupby('order_date').agg({
    'quantity_ordered': 'sum',
    'phi_prediction': 'sum',
    'prophet_prediction': 'sum',
    'xgboost_prediction': 'sum',
    'lstm_prediction': 'sum'
}).reset_index()

fig_ts = go.Figure()

fig_ts.add_trace(go.Scatter(
    x=ts_df['order_date'],
    y=ts_df['quantity_ordered'],
    mode='lines',
    name='Actual Demand',
    line=dict(color='#1E3A5F', width=3)
))

fig_ts.add_trace(go.Scatter(
    x=ts_df['order_date'],
    y=ts_df['phi_prediction'],
    mode='lines',
    name='Phi-Hybrid Prediction',
    line=dict(color='#28A745', width=2, dash='dash')
))

fig_ts.add_trace(go.Scatter(
    x=ts_df['order_date'],
    y=ts_df['prophet_prediction'],
    mode='lines',
    name='Prophet',
    line=dict(color='#FF6B6B', width=1.5, dash='dot')
))

fig_ts.add_trace(go.Scatter(
    x=ts_df['order_date'],
    y=ts_df['xgboost_prediction'],
    mode='lines',
    name='XGBoost',
    line=dict(color='#4ECDC4', width=1.5, dash='dot')
))

fig_ts.add_trace(go.Scatter(
    x=ts_df['order_date'],
    y=ts_df['lstm_prediction'],
    mode='lines',
    name='LSTM',
    line=dict(color='#FFE66D', width=1.5, dash='dot')
))

fig_ts.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Quantity",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
    margin=dict(l=50, r=50, t=30, b=30),
    template='plotly_white'
)

st.plotly_chart(fig_ts, use_container_width=True)

# ============================================================================
# ROW 2: SHAP FEATURE IMPORTANCE & ERROR DISTRIBUTION
# ============================================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 SHAP Feature Importance")
    
    # SHAP values from the project's XAI analysis
    shap_data = pd.DataFrame({
        'Feature': ['shipping_cost', 'reorder_point', 'stock_on_hand', 'supplier_rating', 
                   'stock_in_transit', 'safety_stock_level', 'lead_time_days', 'financial_loss_amount'],
        'SHAP_Value': [2.75, 2.38, 2.13, 1.81, 1.76, 1.53, 1.07, 0.28]
    }).sort_values('SHAP_Value', ascending=True)
    
    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=shap_data['SHAP_Value'],
        y=shap_data['Feature'],
        orientation='h',
        marker=dict(
            color=shap_data['SHAP_Value'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="SHAP Value")
        ),
        text=shap_data['SHAP_Value'].round(2),
        textposition='outside'
    ))
    
    fig_shap.update_layout(
        height=400,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        margin=dict(l=10, r=50, t=30, b=30),
        template='plotly_white'
    )
    
    st.plotly_chart(fig_shap, use_container_width=True)

with col2:
    st.markdown("### 📉 Error Distribution")
    
    residuals = filtered_df['phi_error']
    
    fig_err = go.Figure()
    
    fig_err.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        name='Residuals',
        marker=dict(color='#1E3A5F', opacity=0.7),
        histnorm='probability density'
    ))
    
    # Add normal distribution curve
    mu, sigma = residuals.mean(), residuals.std()
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    
    fig_err.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_err.add_vline(x=0, line_dash="dash", line_color="green", 
                      annotation_text="Zero Error", annotation_position="top right")
    
    fig_err.update_layout(
        height=400,
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=30, b=30),
        template='plotly_white'
    )
    
    st.plotly_chart(fig_err, use_container_width=True)

# ============================================================================
# ROW 3: BUSINESS IMPACT METRICS
# ============================================================================
st.markdown("---")
st.markdown("### 💰 Business Impact Metrics")

# Calculate business metrics
under_forecast = np.maximum(0, filtered_df['quantity_ordered'] - filtered_df['phi_prediction']).sum()
over_forecast = np.maximum(0, filtered_df['phi_prediction'] - filtered_df['quantity_ordered']).sum()

marginal_profit = 10.0  # Assumed marginal profit per unit
carrying_cost = 2.0     # Assumed carrying cost per unit

stockout_savings = under_forecast * marginal_profit
holding_savings = over_forecast * carrying_cost
total_savings = stockout_savings + holding_savings

impact_data = pd.DataFrame({
    'Category': ['Stock-out Cost Savings', 'Holding Cost Optimization', 'Total Projected Savings'],
    'Amount': [stockout_savings, holding_savings, total_savings],
    'Color': ['#DC3545', '#28A745', '#1E3A5F']
})

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📉 Stock-out Cost Savings</div>
        <div class="metric-value" style="color:#DC3545;">${stockout_savings:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📦 Holding Cost Optimization</div>
        <div class="metric-value" style="color:#28A745;">${holding_savings:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💰 Total Projected Savings</div>
        <div class="metric-value" style="color:#1E3A5F;">${total_savings:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# Waterfall chart for business impact
fig_business = go.Figure(go.Waterfall(
    name="Impact",
    orientation="v",
    measure=["relative", "relative", "total"],
    x=impact_data['Category'],
    y=impact_data['Amount'],
    text=[f"${v:,.0f}" for v in impact_data['Amount']],
    textposition="outside",
    decreasing={"marker": {"color": "#DC3545"}},
    increasing={"marker": {"color": "#28A745"}},
    totals={"marker": {"color": "#1E3A5F"}},
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))

fig_business.update_layout(
    height=350,
    title="Supply Chain Cost Savings Breakdown",
    yaxis_title="Amount ($)",
    margin=dict(l=50, r=50, t=50, b=30),
    template='plotly_white'
)

st.plotly_chart(fig_business, use_container_width=True)

# ============================================================================
# ROW 4: SUPPLIER PERFORMANCE & INCIDENT-WISE ERROR
# ============================================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌍 Supplier Performance")
    
    supplier_perf = filtered_df.groupby('supplier_country').agg({
        'quantity_ordered': 'mean',
        'phi_prediction': 'mean',
        'supplier_rating': 'mean',
        'financial_loss_amount': 'mean'
    }).reset_index()
    
    supplier_perf['error'] = (supplier_perf['quantity_ordered'] - supplier_perf['phi_prediction']).abs()
    
    fig_supplier = go.Figure()
    fig_supplier.add_trace(go.Bar(
        x=supplier_perf['supplier_country'],
        y=supplier_perf['error'],
        name='Mean Absolute Error',
        marker=dict(color='#1E3A5F', opacity=0.7)
    ))
    
    fig_supplier.add_trace(go.Scatter(
        x=supplier_perf['supplier_country'],
        y=supplier_perf['supplier_rating'] * 10,
        name='Supplier Rating (scaled)',
        mode='lines+markers',
        line=dict(color='#28A745', width=2),
        marker=dict(size=10)
    ))
    
    fig_supplier.update_layout(
        height=350,
        xaxis_title="Supplier Country",
        yaxis_title="Mean Absolute Error",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=30, b=30),
        template='plotly_white'
    )
    
    st.plotly_chart(fig_supplier, use_container_width=True)

with col2:
    st.markdown("### ⚠️ Incident-wise Error")
    
    fig_incident = go.Figure()
    fig_incident.add_trace(go.Box(
        x=filtered_df['incident_type'],
        y=filtered_df['phi_error'],
        name='Error Distribution',
        boxmean='sd',
        marker=dict(color='#1E3A5F', opacity=0.6)
    ))
    
    fig_incident.add_hline(y=0, line_dash="dash", line_color="green")
    
    fig_incident.update_layout(
        height=350,
        xaxis_title="Incident Type",
        yaxis_title="Prediction Error",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=30, b=30),
        template='plotly_white'
    )
    
    st.plotly_chart(fig_incident, use_container_width=True)

# ============================================================================
# ROW 5: MODEL COMPARISON (TOURNAMENT RESULTS)
# ============================================================================
st.markdown("---")
st.markdown("### 🏆 Model Comparison (Tournament Results)")

tournament_summary = tournament_df.groupby('Model').agg({
    'RMSE': ['mean', 'std'],
    'R²': ['mean', 'std'],
    'MAE': ['mean', 'std']
}).round(4)

tournament_summary.columns = ['RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std', 'MAE_mean', 'MAE_std']
tournament_summary = tournament_summary.reset_index()

fig_tournament = make_subplots(
    rows=1, cols=2,
    subplot_titles=('RMSE Comparison', 'R² Score Comparison'),
    shared_yaxes=False
)

# RMSE Bar Chart
fig_tournament.add_trace(
    go.Bar(
        x=tournament_summary['Model'],
        y=tournament_summary['RMSE_mean'],
        error_y=dict(type='data', array=tournament_summary['RMSE_std'], visible=True),
        marker=dict(
            color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#28A745'],
            opacity=0.8
        ),
        text=tournament_summary['RMSE_mean'].round(2),
        textposition='outside'
    ),
    row=1, col=1
)

# R² Score Bar Chart
fig_tournament.add_trace(
    go.Bar(
        x=tournament_summary['Model'],
        y=tournament_summary['R2_mean'],
        error_y=dict(type='data', array=tournament_summary['R2_std'], visible=True),
        marker=dict(
            color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#28A745'],
            opacity=0.8
        ),
        text=tournament_summary['R2_mean'].round(3),
        textposition='outside'
    ),
    row=1, col=2
)

fig_tournament.update_layout(
    height=400,
    showlegend=False,
    template='plotly_white',
    margin=dict(l=50, r=50, t=50, b=30)
)

fig_tournament.update_xaxes(title_text="Model", row=1, col=1)
fig_tournament.update_xaxes(title_text="Model", row=1, col=2)
fig_tournament.update_yaxes(title_text="RMSE (lower is better)", row=1, col=1)
fig_tournament.update_yaxes(title_text="R² Score (higher is better)", row=1, col=2)

st.plotly_chart(fig_tournament, use_container_width=True)

# ============================================================================
# INSIGHTS & RECOMMENDATIONS
# ============================================================================
st.markdown("---")
st.markdown("### 💡 Key Insights & Recommendations")

# Extract key insights from the data
best_model = tournament_summary.loc[tournament_summary['RMSE_mean'].idxmin(), 'Model']
best_rmse = tournament_summary['RMSE_mean'].min()
phi_rmse = tournament_summary[tournament_summary['Model'] == 'Phi-Hybrid']['RMSE_mean'].values[0]
improvement_pct = ((best_rmse - phi_rmse) / best_rmse * 100)

insights = [
    {
        "icon": "✅",
        "title": "Phi-Hybrid Model Superiority",
        "description": f"The Phi-Hybrid model achieves RMSE = {phi_rmse:.2f}, which is {improvement_pct:.1f}% better than the best individual model ({best_model})."
    },
    {
        "icon": "📊",
        "title": "Top Influential Features",
        "description": "Shipping Cost (SHAP=2.75) and Reorder Point (SHAP=2.38) are the most influential features for demand prediction, indicating logistics and inventory management are key drivers."
    },
    {
        "icon": "💰",
        "title": "Business Impact",
        "description": f"The model provides ${total_savings:,.0f} in projected savings through ${stockout_savings:,.0f} in stock-out cost reduction and ${holding_savings:,.0f} in holding cost optimization."
    },
    {
        "icon": "⚠️",
        "title": "Risk Management",
        "description": f"Incidents like '{filtered_df.groupby('incident_type')['phi_error'].mean().idxmax()}' show the highest prediction errors. Consider reviewing risk mitigation strategies for these incident types."
    }
]

cols = st.columns(4)
for i, insight in enumerate(insights):
    with cols[i]:
        st.markdown(f"""
        <div class="insight-box">
            <h3>{insight['icon']} {insight['title']}</h3>
            <p style="font-size:0.9rem;">{insight['description']}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.8rem; padding: 1rem 0;">
    🏭 Alishamart Supply Chain Hybrid AI Framework | 
    Dashboard built with Streamlit & Plotly | 
    Data: 2025-2026
</div>
""", unsafe_allow_html=True)
