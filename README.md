# DeFi Wallet Credit Scoring using Aave V2 Transaction Data

This project builds a machine learning pipeline to assign **credit scores (0–1000)** to wallets based on their **on-chain behavior** on the Aave V2 protocol.

🔗 **Live Demo**: [Streamlit App](https://anchitchourasia--defi-credit-score-generator-aave-v2-app-eimd0t.streamlit.app/)

---

## 💼 Objective

Analyze ~100K DeFi transactions and generate wallet-level credit scores that reflect:
- ✅ Responsible usage (higher scores)
- ❌ Risky/bot/exploitative behavior (lower scores)

---

## 🧠 Approach

### 1. Feature Engineering:
- Counts of each action type (`deposit`, `borrow`, `repay`, `redeemunderlying`, `liquidationcall`)
- Total amounts in USD by action type
- Behavior ratios: `repay/borrow`, `borrow/deposit`
- Frequency-based stats: total transactions, active days, avg. time between txns

### 2. ML Model:
- Used: `XGBoostRegressor`
- Trained on synthetic labels
- Score scaled to 0–1000
- Script: `score_generator.py`

### 3. Visual Dashboard:
- Built with `Streamlit` and `Plotly`
- Upload your own JSON or use sample
- See score buckets, wallet lists, score distribution

---

## 📂 Files

| File                  | Description                            |
|-----------------------|----------------------------------------|
| `app.py`              | Streamlit app for visual UI            |
| `score_generator.py`  | Core script for scoring wallets        |
| `analysis.md`         | Score distribution & wallet insights   |
| `requirements.txt`    | Python dependencies                    |
| `scripts/user-wallet-transactions.json` | Sample input JSON |

---

## ▶️ How to Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
