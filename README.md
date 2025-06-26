
# 🚀 Advanced Automated Options Trading System

An AI-powered, fully automated options trading engine for Nifty Index, designed to **sell options** (SHORT CALL/PUT) based on real-time data, machine learning models, and quantitative risk controls. It integrates with **Tradetron API** for live execution and supports continuous learning and simulation.

---

## 📌 Features

- ✅ SHORT CALL/PUT Strategy (Signal: -1, +1, 0)
- ✅ Ensemble ML models (Random Forest, XGBoost, LSTM)
- ✅ Real-time NSE option chain analysis
- ✅ 35+ features: OI, Volume, PCR, Volatility, Time, etc.
- ✅ Tradetron API integration for live signal execution
- ✅ Offline simulation + reinforcement learning
- ✅ Risk-managed: SL, targets, Kelly sizing, exposure caps
- ✅ SHAP-based explainability
- ✅ Modular architecture with fallback and recovery

---

## 🧠 Trading Strategy

| Signal | Action      | Logic                     |
|--------|-------------|---------------------------|
| +1     | SHORT PUT   | Expecting price to fall   |
| -1     | SHORT CALL  | Expecting price to fall   |
| 0      | Neutral     | Exit all/open no position |

> 💰 Profits are generated when the shorted option premium drops.

---

## 🔍 Machine Learning Stack

- **Random Forest Classifier**: Core signal generator
- **Gradient Boosting Models**: Confidence boosting
- **LSTM (PyTorch)**: Sequential signal patterns
- **Meta-Learner**: Ensemble final signal
- **SHAP**: Feature contribution & explainability

---

## 🧾 Features Engine

- 📈 Option Prices (CALL/PUT LTP)
- 🔍 OI & OI Change
- 📊 Volume analysis
- ⚖️ Put-Call Ratios
- ⏰ Time (hour, session, volatility state)
- 📉 Volatility Indicators (IV, VIX, HV)
- 🔁 Historical stats and technical indicators

---

## 🔄 Operating Modes

### 🟢 LIVE MODE (9:15 AM - 3:30 PM IST)
- Fetch real-time data
- Generate signals
- Send to Tradetron API
- Paper trade + log data

### 🌙 OFFLINE MODE (Post Market)
- Replay full-day simulations
- Model retraining every 25–50 trades
- PnL-based reward updates

---

## 💼 Risk Management

- 🔐 Kelly Criterion for sizing
- 🛡️ Stop Loss / Profit Target / Trailing Stop
- 📉 Max exposure per trade
- 📊 Limit to 3 concurrent trades
- 🔄 Volatility-adjusted scaling

---

## 🔗 Integrations

- **NSE India API** – Live option chain data
- **Tradetron API** – Signal automation
- **Fallbacks** – Traditional logic if ML fails

---

## 📈 Performance Monitoring

- Win Rate, Sharpe Ratio, Drawdown
- Average PnL per trade
- Confidence tracking
- SHAP-based feature ranking

---

## 🔄 Learning Loop

- Reinforcement Learning based on trade outcomes
- Adaptive confidence thresholds
- Regime-aware learning rate adjustment
- Continuous backtesting + performance checks

---

## ⚙️ Requirements

### 🧱 Dependencies
```bash
Python 3.8+
pandas, numpy
scikit-learn, xgboost
torch
requests
shap
```

### 💾 System Requirements
- 1–2 GB RAM
- Internet connection (live data)
- ~1 GB for historical JSON logs

---

## 🧪 Usage

### ▶️ Run Live Trading
```python
from trading_system import TradingSystem
TradingSystem().run_live()
```

### 🧠 Simulate Offline
```python
TradingSystem().simulate_offline('2025-06-01')
```

### 🔁 Train Models
```python
TradingSystem().train_models()
```

---

## 📂 Project Structure

```
trading_system/
├── core/                   # ML logic, strategies
│   ├── models/             # Trained model files
│   ├── features/           # Feature engineering logic
│   └── risk/               # Risk management utilities
├── data/                   # Historical & live data cache
├── api/                    # NSE & Tradetron API handlers
├── simulation/             # Backtest & replay modules
├── logs/                   # Execution logs
└── main.py                 # Entry point
```

---

## 🛡️ Reliability

- ✅ Circuit breakers & fallbacks
- ✅ API error handling + retry logic
- ✅ Self-healing recovery
- ✅ Logging & alerting

---

## 📊 Example Trade Log (Sample)

```
⏱️ Time: 11:28:12
📡 SIGNAL: 🔴 SHORT CALL
⚡ Strike: 24800
💰 Price: ₹84.25
🎯 Target: ₹72.00 | SL: ₹92.00
🤖 ML Confidence: High | Pattern: Range Breakdown
```

---

## 🧠 SHAP Feature Insight

| Feature            | Impact Score |
|--------------------|--------------|
| OI Change          | +0.42        |
| Volume Spike       | +0.36        |
| Time (Session)     | -0.19        |
| PCR Drop           | +0.18        |
| IV Spread          | -0.11        |

---

## 🚧 Roadmap

- [x] Tradetron integration
- [x] SHAP explainability
- [x] Reinforcement Learning loop
- [ ] Web dashboard with real-time monitoring
- [ ] Multi-strategy comparison module
- [ ] Strategy performance visualizations

---

## 📬 Contact / Support

For questions, feedback, or collaboration:

📧 Email: `your_email@example.com`  
🧠 Developer: `@yourgithubhandle`  
🔗 Tradetron Strategy Page: *(optional link)*

---

## 📄 License

MIT License. Feel free to fork and adapt.

---

> ⚠️ **Disclaimer**: This software is for educational and research purposes only. Options trading involves high risk. Use at your own discretion.
