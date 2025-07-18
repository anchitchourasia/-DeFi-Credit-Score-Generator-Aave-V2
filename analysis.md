# 📊 Wallet Score Analysis – Aave V2 Credit Scoring

This file summarizes key insights and analysis based on the machine learning credit scores generated from 100K+ user-wallet DeFi transactions.

---

## 🧮 Credit Score Distribution

Wallets were scored on a scale of **0 to 1000**, based on features derived from their historical behavior on the Aave V2 protocol.

### 📉 Distribution by Score Range (Bucketed)

| Score Range | Number of Wallets |
|-------------|--------------------|
| 0–100       | ████████████████████  |
| 101–200     | ██████████            |
| 201–300     | ███████               |
| 301–400     | ████                  |
| 401–500     | ███                   |
| 501–600     | ██                    |
| 601–700     | █                     |
| 701–800     | ░                     |
| 801–900     | ░                     |
| 901–1000    | ░                     |

> Most wallets fall in the **0–200** range, suggesting limited interaction or bot-like behavior.

---

## 🔍 High-Scoring Wallet Behavior (700–1000)

These are the most **reliable and responsible users** on the platform.

### Typical traits:
- Frequent and high-volume **deposits**
- Timely and complete **repayments**
- Low or no involvement in **liquidation**
- Balanced **borrow-to-deposit** ratio
- Consistent activity over longer timeframes

---

## 🚨 Low-Scoring Wallet Behavior (0–200)

These are **risk-prone or inactive wallets**, possibly bots or exploiters.

### Common patterns:
- Single or few actions (e.g. 1 borrow, no repay)
- Incomplete or zero repayments
- High involvement in **liquidationcall**
- Sudden large withdrawals
- No meaningful historical engagement

---

## 📈 Feature Importance (Top Predictors)

| Feature Name               | Importance |
|---------------------------|------------|
| `deposit_usd`             | ⭐⭐⭐⭐⭐      |
| `repay_borrow_ratio`      | ⭐⭐⭐⭐       |
| `liquidationcall_count`   | ⭐⭐⭐        |
| `borrow_usd`              | ⭐⭐⭐        |
| `total_transactions`      | ⭐⭐         |

These features were selected after evaluating wallet-level patterns and helped drive the model's score predictions.

---

## 🧪 Model Stats

- **Model Used:** `XGBoostRegressor`
- **Evaluation:** Normalized score outputs based on behavioral trends
- **Normalization:** Final score scaled between 0 and 1000

---

## 📌 Summary

- 3400+ wallets were scored with high granularity
- Responsible DeFi behavior is rewarded with higher scores
- Risky and bot-like usage is effectively penalized
- The scoring model is designed to be extensible and explainable

---

🔁 **Next Steps** (For production use):
- Incorporate external data (e.g., flash loan usage, time-in-pool)
- Add anomaly detection for sudden changes
- Use graph analytics to detect bot networks

