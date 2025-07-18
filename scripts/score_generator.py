import argparse
import pandas as pd
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

def load_data(input_file):
    print("[*] Loading data...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def clean_transform(df):
    print("[*] Transforming and cleaning data...")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date

    df['usd_value'] = df['actionData'].apply(
        lambda x: float(x.get('amount', 0)) * float(x.get('assetPriceUSD', 0))
    ) / 1e18

    df['user_id'] = df['userWallet']
    df['action_type'] = df['action'].str.lower()

    return df

def engineer_features(df):
    print("[*] Engineering wallet-level features...")

    features = df.groupby('user_id').agg(
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

    # Additional ratios
    features['repay_borrow_ratio'] = features['repay_count'] / (features['borrow_count'] + 1)
    features['deposit_redeem_ratio'] = features['deposit_count'] / (features['redeem_count'] + 1)

    return features.reset_index()

def compute_credit_score(df):
    print("[*] Training ML model to compute scores...")

    # Synthetic target (domain-driven logic)
    df['target_score'] = (
        df['repay_borrow_ratio'] * 3 +
        df['deposit_redeem_ratio'] * 2 +
        df['active_days'] * 0.1 +
        df['avg_usd'] * 0.01 +
        df['total_txns'] * 0.5 -
        df['liquidation_count'] * 5
    )

    feature_cols = [
        "deposit_count", "borrow_count", "repay_count", "redeem_count", "liquidation_count",
        "total_txns", "total_usd", "avg_usd", "std_usd", "max_usd",
        "active_days", "repay_borrow_ratio", "deposit_redeem_ratio"
    ]

    X = df[feature_cols]
    y = df['target_score']

    # Replace RandomForest with XGBoost
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    df['raw_score'] = model.predict(X)

    # Normalize scores between 0 and 1000
    scaler = MinMaxScaler(feature_range=(0, 1000))
    df['credit_score'] = scaler.fit_transform(df[['raw_score']])
    df['credit_score'] = df['credit_score'].round().astype(int)

    return df[['user_id', 'credit_score']]

def main(input_file, output_file):
    df = load_data(input_file)
    df = clean_transform(df)
    features_df = engineer_features(df)
    scored = compute_credit_score(features_df)

    print(f"[✓] Scored {len(scored)} wallets.")
    scored.to_csv(output_file, index=False)
    print(f"[✓] Saved credit scores to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    main(args.input, args.output)
