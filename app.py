# app.py

import streamlit as st
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import plotly.express as px

# --- Streamlit Setup ---
st.set_page_config("Aave V2 Credit Scoring Dashboard", layout="wide")
st.title("🏦 Aave V2 Wallet Credit Scoring Dashboard")

# --- Sidebar: Sample JSON Download ---
with st.sidebar:
    st.markdown("## 📁 Download Sample JSON")
    try:
        with open("scripts/user-wallet-transactions.json", "rb") as f:
            st.download_button(
                label="⬇️ Download Sample File",
                data=f,
                file_name="user-wallet-transactions.json",
                mime="application/json"
            )
    except FileNotFoundError:
        st.warning("Sample file not found in /scripts folder.")
    
    st.markdown("---")
    st.markdown("👈 Upload your JSON file on the main screen to analyze and score wallets.")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload your `user-wallet-transactions.json` file", type="json")

if uploaded_file:
    # --- Load JSON Data ---
    data = json.load(uploaded_file)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date
    df['user_id'] = df['userWallet']
    df['action_type'] = df['action'].str.lower()

    # --- USD Value Calculation ---
    df['usd_value'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) * float(x.get('assetPriceUSD', 0))) / 1e18

    # --- Feature Engineering ---
    feat = df.groupby('user_id').agg(
        deposit_count=('action_type', lambda x: (x == 'deposit').sum()),
        borrow_count=('action_type', lambda x: (x == 'borrow').sum()),
        repay_count=('action_type', lambda x: (x == 'repay').sum()),
        redeem_count=('action_type', lambda x: (x == 'redeemunderlying').sum()),
        liquidation_count=('action_type', lambda x: (x == 'liquidationcall').sum()),
        total_txns=('action_type', 'count'),
        total_usd=('usd_value', 'sum'),
        avg_usd=('usd_value', 'mean'),
        std_usd=('usd_value', 'std'),
        max_usd=('usd_value', 'max'),
        active_days=('date', pd.Series.nunique)
    ).fillna(0)

    feat['repay_borrow_ratio'] = feat['repay_count'] / (feat['borrow_count'] + 1)
    feat['deposit_redeem_ratio'] = feat['deposit_count'] / (feat['redeem_count'] + 1)

    # --- Credit Score Target (Rule-Based) ---
    feat['target_score'] = (
        feat['repay_borrow_ratio'] * 3 +
        feat['deposit_redeem_ratio'] * 2 +
        feat['active_days'] * 0.1 +
        feat['avg_usd'] * 0.01 +
        feat['total_txns'] * 0.5 -
        feat['liquidation_count'] * 5
    )

    features = [
        "deposit_count", "borrow_count", "repay_count", "redeem_count", "liquidation_count",
        "total_txns", "total_usd", "avg_usd", "std_usd", "max_usd",
        "active_days", "repay_borrow_ratio", "deposit_redeem_ratio"
    ]

    # --- ML Model Training ---
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(feat[features], feat['target_score'])
    feat['raw_score'] = model.predict(feat[features])

    # --- Normalize Score to 0–1000 ---
    scaler = MinMaxScaler((0, 1000))
    feat['credit_score'] = scaler.fit_transform(feat[['raw_score']]).round().astype(int)

    result_df = feat.reset_index()[['user_id', 'credit_score']]
    df_merged = df.merge(result_df, on='user_id', how='left')

    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Average Score", f"{result_df['credit_score'].mean():.2f}")
    col2.metric("🏆 Max Score", f"{result_df['credit_score'].max()}")
    col3.metric("🧯 Min Score", f"{result_df['credit_score'].min()}")
    col4.metric("❗ Wallets < 200", f"{(result_df['credit_score'] < 200).sum()}")

    # --- Bucket Distribution ---
    result_df['bucket'] = pd.cut(
        result_df['credit_score'],
        bins=[0,100,200,300,400,500,600,700,800,900,1000],
        labels=["0-100","100-200","200-300","300-400","400-500",
                "500-600","600-700","700-800","800-900","900-1000"]
    )

    bucket_df = result_df['bucket'].value_counts().sort_index().reset_index()
    bucket_df.columns = ['score_range', 'count']

    dist_fig = px.bar(
        bucket_df, x='score_range', y='count',
        color='score_range', title="📊 Credit Score Bucket Distribution",
        labels={'score_range': 'Score Range', 'count': 'Wallet Count'},
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(dist_fig, use_container_width=True)

    # --- Action Distribution Pie Chart ---
    pie_df = df['action_type'].value_counts().reset_index()
    pie_df.columns = ['action', 'count']
    pie_fig = px.pie(pie_df, names='action', values='count', title='🧩 Action Type Distribution')
    st.plotly_chart(pie_fig, use_container_width=True)

    # --- USD Value per Action ---
    usd_action = df.groupby('action_type')['usd_value'].mean().reset_index()
    usd_fig = px.bar(usd_action, x='action_type', y='usd_value',
                     title="💸 Avg USD Volume per Action Type",
                     color='usd_value', color_continuous_scale='Viridis')
    st.plotly_chart(usd_fig, use_container_width=True)

    # --- Score over Time Trend ---
    score_time = df_merged.groupby(df_merged['timestamp'].dt.to_period('M')).agg(
        avg_score=('credit_score', 'mean')).reset_index()
    score_time['timestamp'] = score_time['timestamp'].astype(str)

    score_line = px.line(score_time, x='timestamp', y='avg_score',
                         title="📈 Average Credit Score Over Time",
                         markers=True)
    st.plotly_chart(score_line, use_container_width=True)

    # --- Top & Bottom Wallet Tables ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🥇 Top 10 Wallets")
        st.dataframe(result_df.sort_values('credit_score', ascending=False).head(10), use_container_width=True)
    with col2:
        st.markdown("### 🔻 Bottom 10 Wallets")
        st.dataframe(result_df.sort_values('credit_score').head(10), use_container_width=True)

    # --- Download Button ---
    st.download_button("📥 Download Credit Score CSV",
                       result_df.to_csv(index=False).encode('utf-8'),
                       file_name="wallet_scores.csv",
                       mime="text/csv")
else:
    st.info("👆 Please upload a valid Aave user-transactions JSON file to begin analysis.")
