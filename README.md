
# 🚀 Advanced Automated Options Trading System generated using AI needs professional help to improve the code

An AI-powered, fully automated options trading engine for the Nifty Index, designed to **sell options** (SHORT CALL/PUT) based on real-time data, machine learning models, and quantitative risk controls. It integrates with **Tradetron API** for live execution and supports continuous learning and simulation.

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

📧 Email: uday14viru@gmail.com  
🔗 Tradetron Strategy Page: https://tradetron.tech/strategy/7909313 (Duplicate, you can modify to add stoploss and target, and other technical analysis using tradetron keywords )

**Need to generate API token in Tradetron after duplicating the strategy**
```python
        self.TRADETRON_URLS = {
            1: "https://api.tradetron.tech/api?auth-token=<YOURAPITOKEN>&key=gap&value=1",

            -1: "https://api.tradetron.tech/api?auth-token=YOURAPITOKEN&key=gap&value=-1",

            0: "https://api.tradetron.tech/api?auth-token=YOURAPITOKEN&key=gap&value=0"

        }
```     
**Replace the token <YOURAPITOKEN> with the generated token example**
```python
self.TRADETRON_URLS = {
            1: "https://api.tradetron.tech/api?auth-token=2621e9d9-5349-41bf-b4d4-27acad38abe&key=gap&value=1",

            -1: "https://api.tradetron.tech/api?auth-token=2621e9d9-5349-41bf-b4d4-27acad38abe&key=gap&value=-1",

            0: "https://api.tradetron.tech/api?auth-token=2621e9d9-5349-41bf-b4d4-27acad38abe&key=gap&value=0"

        }
```
---

## 📄 License

MIT License. Feel free to fork and adapt.

---

> ⚠️ **Disclaimer**: This software is for educational and research purposes only. Options trading involves high risk. Use at your discretion. All code is generated using AI.

**Follow to generate token**

<img width="866" height="254" alt="Screenshot 2025-08-22 114928" src="https://github.com/user-attachments/assets/2585a968-fb62-4920-af25-cb2cb93a632b" />

<img width="848" height="247" alt="Screenshot 2025-08-22 115147" src="https://github.com/user-attachments/assets/435c42b0-4ba0-4928-bb96-d9b2717e3883" />

<img width="782" height="495" alt="Screenshot 2025-08-22 115244" src="https://github.com/user-attachments/assets/f0e45446-8589-420f-8388-6f3982b042af" />

<img width="822" height="517" alt="Screenshot 2025-08-22 115322" src="https://github.com/user-attachments/assets/249b363c-8d3f-496c-ad70-3a09ff8c3a4d" />

<img width="793" height="560" alt="Screenshot 2025-08-22 115500" src="https://github.com/user-attachments/assets/ff76833d-77b8-4eac-b27d-7b51caa72234" />

flowchart TD
    A[🚀 System Start] --> B[Initialize Components]
    B --> C[Load Existing Models]
    C --> D[Initialize with Stored Data]
    D --> E[Main Analysis Loop]
    
    E --> F{Market Hours?}
    
    F -->|Yes - Live Mode| G[🟢 LIVE TRADING MODE]
    F -->|No - Offline Mode| H[🌙 OFFLINE SIMULATION MODE]
    
    G --> I[Fetch Live Option Data]
    I --> J{Data Fetch Success?}
    J -->|No| K[Use Fallback Data]
    J -->|Yes| L[Store Live Data]
    
    H --> M[Load Stored Data]
    M --> N[Create Trading Simulation]
    N --> O[Use Sequential Data]
    
    K --> P[Get Nearest Strikes]
    L --> P
    O --> P
    
    P --> Q[Extract ML Features]
    Q --> R[Market Regime Detection]
    R --> S[ML Analysis & Predictions]
    
    S --> T{ML Model Trained?}
    T -->|No| U[Use Traditional Analysis]
    T -->|Yes| V[Advanced ML Predictions]
    
    U --> W[Feature-based Signals]
    V --> X[Ensemble ML Signals]
    
    W --> Y[Signal Generation]
    X --> Y
    
    Y --> Z{Paper Trade Active?}
    
    Z -->|Yes| AA[Check Exit Conditions]
    Z -->|No| BB[Check Entry Conditions]
    
    AA --> CC{Exit Signal?}
    CC -->|Yes| DD[🚨 Exit Trade]
    CC -->|No| EE[Continue Monitoring]
    
    BB --> FF{Signal Strength > Threshold?}
    FF -->|Yes| GG[Enter Paper Trade]
    FF -->|No| HH[No Action]
    
    DD --> II[📡 Send EXIT Signal to Tradetron]
    GG --> JJ[📡 Send ENTRY Signal to Tradetron]
    
    II --> KK[Update ML with Trade Outcome]
    JJ --> LL[Start Position Monitoring]
    
    KK --> MM[Calculate Performance Metrics]
    LL --> EE
    HH --> EE
    
    MM --> NN[Continuous Learning Update]
    EE --> NN
    
    NN --> OO{Retrain Models?}
    OO -->|Yes| PP[🔄 Model Training]
    OO -->|No| QQ[Collect Market Data]
    
    PP --> RR[SHAP Feature Analysis]
    RR --> SS[Save Updated Models]
    SS --> QQ
    
    QQ --> TT[Risk Management Check]
    TT --> UU[Performance Display]
    UU --> VV[System Cleanup]
    VV --> WW[Sleep & Wait]
    
    WW --> E
    
    subgraph "🧠 ML Components"
        XX[Random Forest Classifier]
        YY[Gradient Boosting Models]
        ZZ[LSTM Deep Learning]
        AAA[SHAP Explainer]
        XX --> YY --> ZZ --> AAA
    end
    
    subgraph "💰 Risk Management"
        BBB[Position Sizing]
        CCC[Stop Loss Calculation]
        DDD[Trailing Stop Logic]
        EEE[Portfolio Risk Check]
        BBB --> CCC --> DDD --> EEE
    end
    
    subgraph "📊 Data Management"
        FFF[Live Data Storage]
        GGG[Historical Data Access]
        HHH[Feature Extraction]
        III[Data Cleanup]
        FFF --> GGG --> HHH --> III
    end
    
    subgraph "🎯 Signal Types"
        JJJ[🟢 SHORT PUT<br/>Signal = 1]
        KKK[🔴 SHORT CALL<br/>Signal = -1]
        LLL[⚪ NEUTRAL<br/>Signal = 0]
    end
    
    S -.-> XX
    Y -.-> JJJ
    Y -.-> KKK
    Y -.-> LLL
    
    L -.-> FFF
    Q -.-> HHH
    GG -.-> BBB
    AA -.-> CCC
    
    style A fill:#ff9999
    style G fill:#90EE90
    style H fill:#87CEEB
    style DD fill:#FFB6C1
    style GG fill:#98FB98
    style II fill:#FFA500
    style JJ fill:#FFA500
    style PP fill:#DDA0DD
    style JJJ fill:#90EE90
    style KKK fill:#FFB6C1
    style LLL fill:#F0F0F0
