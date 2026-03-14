# FinSight AI 🧠
### Financial Time Series Forecasting + RL Trading Agent
**Author:** Suman Das — Senior Applied Scientist, Financial AI
**Stack:** Python · PyTorch · Reinforcement Learning · Streamlit
**Domain:** 10 years Banking (PNB) + MTech AI (Jadavpur University, CGPA 9.81)

## Project Goal
End-to-end Financial AI system combining:
- LSTM / GRU / Temporal Fusion Transformer for price forecasting
- PPO Reinforcement Learning agent for portfolio optimization
- BalancedTradingEnv with inactivity penalty for active trading
- Live Streamlit dashboard (coming Day 6)

## Progress
- [x] Day 1 — Data pipeline, EDA, stationarity tests, ARIMA baseline
- [x] Day 2 — LSTM forecaster, walk-forward validation, scaling pipeline
- [x] Day 3 — Temporal Fusion Transformer + attention heatmap
- [x] Day 4 — RL Environment, Gymnasium TradingEnv, random agent baseline
- [x] Day 5 — PPO V1+V2+V3 trained, BalancedTradingEnv, bear market analysis
- [ ] Day 6 — Streamlit dashboard
- [ ] Day 7 — GitHub polish + profile launch

## Assets Covered
NIFTY50 · Reliance · TCS · Bitcoin (2019-2024)

## Forecasting Results — Reliance 2024
| Model          | MAE      | RMSE     | Directional Acc |
|----------------|----------|----------|-----------------|
| ARIMA baseline | 0.011502 | 0.013421 | ~50%            |
| LSTM V2        | 0.015139 | 0.018931 | 49.7%           |
| TFT            | 0.012382 | 0.018418 | 53.8%           |

## RL Trading Results — Reliance 2024 (Bear Market)
| Agent          | Total Return | Sharpe | Max Drawdown | Behaviour          |
|----------------|-------------|--------|--------------|--------------------|
| PPO V1         | 0.00%       | 0.000  | 0.00%        | Cash preservation  |
| PPO V2         | 0.00%       | 0.000  | 0.00%        | Cash preservation  |
| PPO V3         | -5.47%      | -0.745 | -7.79%       | Active trading ⭐  |
| Buy & Hold     | -18.08%     | -0.963 | -24.46%      | Market benchmark   |
| Random Agent   | -14.37%     | -1.456 | -16.07%      | Random baseline    |

PPO V3 outperformed Buy & Hold by +12.61% in a bear market.
Max drawdown reduced from 24.46% to 7.79% — 16.67% improvement.

## RL Agent Evolution Story
- PPO V1: Learned cash preservation — rational in bear market
- PPO V2: Deeper network [128,128,64] — same cash preservation
- PPO V3: BalancedTradingEnv with inactivity penalty (-0.0002)
  broke cash-preservation bias — agent learned active trading
  (38 buys, 205 sells across 243 trading days)

## Key Technical Decisions
- Walk-forward validation — no data leakage
- Log returns instead of raw prices — ensures stationarity
- ADF test — statistically confirmed stationarity
- StandardScaler with inverse transform — fair model comparison
- Dropout 0.3 + gradient clipping — overfitting control
- Sharpe-adjusted reward — penalises volatility not just losses
- Transaction cost 0.1% — realistic trading simulation
- Stop loss at 70% — prevents catastrophic drawdown
- PPO clip_range=0.2 — stable policy updates
- Inactivity penalty -0.0002 — forces active trading behaviour

## TFT Attention Insight
TFT independently discovered three financially meaningful patterns:
- Days 0-25: Near zero attention — old data is noise
- Day 30 spike: Monthly options expiry cycle detected
- Days 50-60: Highest attention — recency matters most
The model learned these patterns without being explicitly programmed.

## RL Key Insight
PPO V1 and V2 learned cash preservation in bear market —
outperforming Buy & Hold on risk metrics (Sharpe, Drawdown).
PPO V3 with inactivity penalty learned active trading —
beating Buy & Hold by 12.61% on total return.
Training on 2019-2022 bull data generalised correctly
to avoid 2024 bear market losses.

## Project Structure
notebooks/
  FinSight_Day1.ipynb                — EDA + ARIMA baseline
  FinSight_Day2_LSTM.ipynb           — LSTM training pipeline
  FinSight_Day3_TFT.ipynb            — TFT + attention heatmap
  FinSight_Day4_RL_Environment.ipynb — Custom Gymnasium environment
  FinSight_Day5_PPO_Agent.ipynb      — PPO V1+V2+V3 training + backtest

models/
  ppo_trading_agent.zip              — PPO V1 weights
  ppo_finsight_v2_optimized.zip      — PPO V2 weights
  ppo_balanced_agent_v3.zip          — PPO V3 weights (best)

visuals/
  tft_attention.png                  — TFT attention heatmap
  random_agent_baseline.png          — Random agent equity curve
  ppo_backtest_results.png           — PPO vs baselines
  ppo_action_analysis.png            — PPO action distribution
  final_equity_comparison.png        — All agents comparison ⭐

## Contact
📧 suman.ju.ai@gmail.com
🔗 linkedin.com/in/suman-das-6b0749276
