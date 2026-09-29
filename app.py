import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="StockSense Retail Intelligence Platform", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .kpi-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 2rem;
        color: #0f172a;
        font-weight: 700;
    }
    .badge-high { background-color: #fecdd3; color: #9f1239; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 1. Load models and data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/master_analytics_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        # Add a couple simple features for display
        df['revenue'] = df['quantity'] * df['price']
        return df, True
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        return pd.DataFrame(), False
        
@st.cache_resource
def load_models():
    try:
        reg = joblib.load('models/demand_forecast_model.pkl')
        clf = joblib.load('models/stockout_risk_model.pkl')
        return reg, clf, True
    except Exception:
        return None, None, False

df, data_loaded = load_data()
reg_model, clf_model, models_loaded = load_models()

# Global Header
st.markdown("## StockSense Intelligence\n#### Retail Demand & Inventory Intelligence")
if data_loaded and models_loaded:
    st.markdown("🟢 **Data & Models Loaded Successfully**")
else:
    st.markdown("🔴 **Error Loading Data or Models**")
    st.stop()

# Recreate Feature Engineering for inference on latest available date per store/product
@st.cache_data
def prepare_inference_data(df):
    inf_df = df.sort_values(by=['store_id', 'product_id', 'date']).copy()
    
    # Feature Engineering matching ML Pipeline
    inf_df['day_of_week'] = inf_df['date'].dt.dayofweek
    inf_df['weekend_flag'] = (inf_df['day_of_week'] >= 5).astype(int)
    inf_df['month'] = inf_df['date'].dt.month
    inf_df['week_no'] = inf_df['date'].dt.isocalendar().week.astype(int)
    inf_df['festival_flag'] = inf_df['holiday']
    
    inf_df['lag_1'] = inf_df.groupby(['store_id', 'product_id'])['quantity'].shift(1)
    inf_df['lag_7'] = inf_df.groupby(['store_id', 'product_id'])['quantity'].shift(7)
    inf_df['lag_14'] = inf_df.groupby(['store_id', 'product_id'])['quantity'].shift(14)
    
    inf_df['rolling_mean_7'] = inf_df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(1).rolling(7).mean())
    inf_df['rolling_std_7'] = inf_df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(1).rolling(7).std())
    
    safe_demand = inf_df['rolling_mean_7'].replace(0, 1)
    inf_df['days_of_inventory'] = inf_df['stock_level'] / safe_demand
    
    inf_df['promotion_flag'] = inf_df['holiday']
    
    # Get latest date per product-store for predictions
    latest = inf_df.groupby(['store_id', 'product_id']).last().reset_index()
    latest = latest.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'rolling_mean_7'])
    
    features = ['day_of_week', 'weekend_flag', 'month', 'week_no', 'festival_flag', 
                'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7', 
                'days_of_inventory', 'promotion_flag', 'store_id', 'product_id']
    
    if len(latest) > 0:
        X = latest[features]
        latest['forecast_demand'] = reg_model.predict(X)
        latest['stockout_prob'] = clf_model.predict_proba(X)[:, 1] if hasattr(clf_model, 'predict_proba') else 0.0
    else:
        # Fallback if window is too small (e.g. data < 14 days)
        # Using mocked safe values to keep app running
        latest = df.groupby(['store_id', 'product_id']).last().reset_index()
        latest['forecast_demand'] = latest['quantity'] * 7 # heuristic
        latest['stockout_prob'] = np.where(latest['stock_level'] < latest['forecast_demand'], 0.8, 0.2)
        
    # Business Action Layer
    latest['safety_stock'] = latest['forecast_demand'] * 0.20
    latest['recommended_stock'] = latest['forecast_demand'] + latest['safety_stock']
    incoming_stock = 0 
    latest['reorder_quantity'] = np.maximum(0, latest['recommended_stock'] - latest['stock_level'] - incoming_stock).astype(int)
    
    def classify_risk(prob):
        if prob >= 0.70: return 'High'
        elif prob >= 0.40: return 'Medium'
        else: return 'Low'
    
    latest['risk_level'] = latest['stockout_prob'].apply(classify_risk)
    latest['manager_action'] = "Reorder " + latest['reorder_quantity'].astype(str) + " units."
    
    return latest

latest_preds = prepare_inference_data(df)

# Global Sidebar Filters
st.sidebar.title("Filters")
selected_stores = st.sidebar.multiselect("Store ID", options=df['store_id'].unique())
selected_regions = st.sidebar.multiselect("Region", options=df['region'].dropna().unique())
selected_categories = st.sidebar.multiselect("Category", options=df['category'].dropna().unique())
selected_products = st.sidebar.multiselect("Product ID", options=df['product_id'].unique())
selected_risks = st.sidebar.multiselect("Risk Level", options=['High', 'Medium', 'Low'])

if st.sidebar.button("Reset Filters"):
    selected_stores = []
    selected_regions = []
    selected_categories = []
    selected_products = []
    selected_risks = []

# Apply filters
filtered_df = df.copy()
filtered_preds = latest_preds.copy()

if selected_stores:
    filtered_df = filtered_df[filtered_df['store_id'].isin(selected_stores)]
    filtered_preds = filtered_preds[filtered_preds['store_id'].isin(selected_stores)]
if selected_regions:
    filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    # region might not be in preds if we dropped it, join it
    if 'region' not in filtered_preds.columns:
        filtered_preds = filtered_preds.merge(df[['store_id', 'product_id', 'region']].drop_duplicates(), on=['store_id', 'product_id'], how='left')
    filtered_preds = filtered_preds[filtered_preds['region'].isin(selected_regions)]
if selected_categories:
    filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    if 'category' not in filtered_preds.columns:
        filtered_preds = filtered_preds.merge(df[['store_id', 'product_id', 'category']].drop_duplicates(), on=['store_id', 'product_id'], how='left')
    filtered_preds = filtered_preds[filtered_preds['category'].isin(selected_categories)]
if selected_products:
    filtered_df = filtered_df[filtered_df['product_id'].isin(selected_products)]
    filtered_preds = filtered_preds[filtered_preds['product_id'].isin(selected_products)]
if selected_risks:
    filtered_preds = filtered_preds[filtered_preds['risk_level'].isin(selected_risks)]

# Tabs Navigation
tabs = st.tabs([
    "Executive Overview", 
    "Demand Intelligence", 
    "Inventory Risk Centre", 
    "Manager Action Centre", 
    "Product Deep Dive", 
    "Model Performance", 
    "EDA Insights"
])

with tabs[0]:
    st.header("Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    total_rev = filtered_df['revenue'].sum()
    units_sold = filtered_df['quantity'].sum()
    inv_value = (filtered_preds['stock_level'] * filtered_preds['price']).sum()
    high_risk_count = len(filtered_preds[filtered_preds['risk_level'] == 'High'])
    
    col1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Revenue</div><div class="kpi-value">₹{total_rev:,.2f}</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="kpi-card"><div class="kpi-label">Units Sold</div><div class="kpi-value">{units_sold:,.0f}</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="kpi-card"><div class="kpi-label">Current Inv Value</div><div class="kpi-value">₹{inv_value:,.2f}</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk Products</div><div class="kpi-value">{high_risk_count}</div></div>', unsafe_allow_html=True)

with tabs[1]:
    st.header("Demand Intelligence")
    st.subheader("Historical Demand Trend")
    daily_demand = filtered_df.groupby('date')['quantity'].sum().reset_index()
    if not daily_demand.empty:
        st.line_chart(daily_demand.set_index('date'), use_container_width=True)
    
    st.subheader("7-Day Forecast vs Current Stock")
    agg_preds = filtered_preds.groupby('category')[['forecast_demand', 'stock_level']].sum().reset_index() if 'category' in filtered_preds.columns else filtered_preds.groupby('product_id')[['forecast_demand', 'stock_level']].sum().reset_index()
    st.bar_chart(agg_preds.set_index(agg_preds.columns[0]), use_container_width=True)

with tabs[2]:
    st.header("Inventory Risk Centre")
    risk_counts = filtered_preds['risk_level'].value_counts()
    st.bar_chart(risk_counts, use_container_width=True)
    
    st.write("### High Risk Products")
    high_risk_df = filtered_preds[filtered_preds['risk_level'] == 'High']
    if not high_risk_df.empty:
        st.dataframe(high_risk_df[['store_id', 'product_id', 'stock_level', 'forecast_demand', 'stockout_prob']], use_container_width=True)
    else:
        st.info("No high risk products currently.")

with tabs[3]:
    st.header("Manager Action Centre")
    display_cols = ['store_id', 'product_id', 'stock_level', 'forecast_demand', 'stockout_prob', 'risk_level', 'reorder_quantity', 'manager_action']
    action_df = filtered_preds[display_cols].copy()
    
    def color_risk(val):
        if val == 'High': return 'background-color: #fecdd3; color: #9f1239'
        elif val == 'Medium': return 'background-color: #fef3c7; color: #92400e'
        elif val == 'Low': return 'background-color: #d1fae5; color: #065f46'
        return ''
        
    st.dataframe(action_df.style.map(color_risk, subset=['risk_level']).format({
        'forecast_demand': '{:.1f}',
        'stockout_prob': '{:.2%}'
    }), use_container_width=True)

with tabs[4]:
    st.header("Product Deep Dive")
    if not filtered_preds.empty:
        selected_p = st.selectbox("Select Product-Store Combination", 
                                  options=[f"Store {r.store_id} - Prod {r.product_id}" for _, r in filtered_preds.iterrows()])
        s_id = int(selected_p.split(' ')[1])
        p_id = int(selected_p.split(' ')[4])
        
        prod_data = filtered_preds[(filtered_preds['store_id'] == s_id) & (filtered_preds['product_id'] == p_id)].iloc[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Stock", prod_data['stock_level'])
        c2.metric("7-Day Forecast", f"{prod_data['forecast_demand']:.1f}")
        c3.metric("Stock-out Prob", f"{prod_data['stockout_prob']:.2%}")
        
        st.write("### Why is this product at risk?")
        if hasattr(clf_model, 'feature_importances_'):
            features = ['day_of_week', 'weekend_flag', 'month', 'week_no', 'festival_flag', 
                        'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7', 
                        'days_of_inventory', 'promotion_flag', 'store_id', 'product_id']
            # Limit to feature array length to avoid mismatch
            n_feat = len(clf_model.feature_importances_)
            f_names = features[:n_feat] if n_feat <= len(features) else [f"Feature {i}" for i in range(n_feat)]
            imp = pd.Series(clf_model.feature_importances_, index=f_names).sort_values(ascending=False).head(5)
            st.bar_chart(imp)
        else:
            st.write("Feature importance not available for this model.")

with tabs[5]:
    st.header("Model Performance")
    st.write("*(Note: For Hackathon validation on the test set, refer to ml_pipeline.py terminal output)*")
    st.markdown("### Model 1: Demand Forecast (RandomForestRegressor)")
    st.write("Validation Strategy: Chronological Time-Aware Split (80/20)")
    
    st.markdown("### Model 2: Stock-out Risk (RandomForestClassifier)")
    st.write("Validation Strategy: Chronological Time-Aware Split (80/20)")
    
    st.info("Metrics like MAE, RMSE, MAPE, Accuracy, and F1 were generated during the training phase. Please check `reports/` or the training logs for exact test-set figures.")

with tabs[6]:
    st.header("EDA & Statistical Insights")
    st.write("Referencing Statistical Hypothesis tests from Round 1:")
    
    st.markdown("""
    **Test 1: Promotion Lift**
    - **Business Question**: Do promotions significantly increase units sold?
    - **H0**: Mean demand during promotions = non-promotions.
    - **Conclusion**: Evaluated via Welch's T-test.
    
    **Test 2: Store Type Demand**
    - **Business Question**: Does mean demand differ across regions?
    - **H0**: Mean demand is the same across all regions.
    - **Conclusion**: Evaluated via One-way ANOVA.
    """)
    st.image('reports/eda_charts/category_revenue.png', caption="Category Revenue Share")
    st.image('reports/eda_charts/daily_demand_store.png', caption="Daily Demand by Store")
