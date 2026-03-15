import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title = "FinSight AI",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1a6b72;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 18px;
        color: #6b6355;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 FinSight AI")
st.subheader("Financial Time Series Forecasting + RL Trading Agent")
st.caption("by Suman Das | PNB 11 yrs Banking · Jadavpur University MTech IAR · CGPA 9.79")

st.divider()

st.sidebar.title("FinSight AI")
st.sidebar.markdown("**Author:** Suman Das")
st.sidebar.markdown("**Domain:** 11 yrs Banking + MTech IAR")
st.sidebar.markdown("**University:** Jadavpur University")
st.sidebar.markdown("**CGPA:** 9.79")
st.sidebar.divider()

asset_options = {
    "Reliance Industries" : "RELIANCE.NS",
    "TCS"                 : "TCS.NS",
    "NIFTY 50"            : "^NSEI",
    "Bitcoin"             : "BTC-USD"
}

selected_asset = st.sidebar.selectbox(
    "Select Asset",
    list(asset_options.keys())
)
ticker = asset_options[selected_asset]

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime("2024-12-31"))

st.sidebar.divider()
st.sidebar.markdown("📧 suman.ju.ai@gmail.com")
st.sidebar.markdown(
    "🔗 [LinkedIn](https://linkedin.com/in/suman-das-6b0749276)")
st.sidebar.markdown(
    "💻 [GitHub](https://github.com/suman-ju-ai)")

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker,
                     start=start, end=end,
                     progress=False)
    if df.empty:
        return None
    feat = pd.DataFrame(index=df.index)
    feat["Open"]   = df["Open"].squeeze()
    feat["High"]   = df["High"].squeeze()
    feat["Low"]    = df["Low"].squeeze()
    feat["Close"]  = df["Close"].squeeze()
    feat["Volume"] = df["Volume"].squeeze()
    feat["Log_Return"] = np.log(
        feat["Close"] / feat["Close"].shift(1))
    feat["Volume_Ratio"] = (
        feat["Volume"] /
        feat["Volume"].rolling(20).mean())
    feat["RSI"]  = ta.rsi(feat["Close"], length=14)
    macd_df = ta.macd(feat["Close"])
    if macd_df is not None:
        feat["MACD"] = macd_df.iloc[:, 0]
    bb_df = ta.bbands(feat["Close"], length=20)
    if bb_df is not None:
        feat["BB_upper"]  = bb_df.iloc[:, 2]
        feat["BB_middle"] = bb_df.iloc[:, 1]
        feat["BB_lower"]  = bb_df.iloc[:, 0]
    feat["MA_20"] = feat["Close"].rolling(20).mean()
    feat["MA_50"] = feat["Close"].rolling(50).mean()
    return feat.dropna()

with st.spinner(f"Loading {selected_asset} data..."):
    df = load_data(ticker,
                   start_date.strftime("%Y-%m-%d"),
                   end_date.strftime("%Y-%m-%d"))

if df is None or df.empty:
    st.error("No data found.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "📈 Market Analysis",
    "🤖 RL Trading Simulator",
    "📊 Model Results"
])

with tab1:
    st.header(f"📈 {selected_asset} — Market Analysis")

    col1, col2, col3, col4 = st.columns(4)
    latest_price  = df["Close"].iloc[-1]
    prev_price    = df["Close"].iloc[-2]
    price_change  = ((latest_price - prev_price) /
                     prev_price * 100)
    total_return  = ((df["Close"].iloc[-1] -
                      df["Close"].iloc[0]) /
                     df["Close"].iloc[0] * 100)
    volatility    = df["Log_Return"].std() * np.sqrt(252) * 100
    latest_rsi    = df["RSI"].iloc[-1]

    col1.metric("Latest Price",
                f"₹{latest_price:,.2f}",
                f"{price_change:+.2f}%")
    col2.metric("Period Return",
                f"{total_return:+.2f}%")
    col3.metric("Annual Volatility",
                f"{volatility:.1f}%")
    col4.metric("RSI", f"{latest_rsi:.1f}",
                "Overbought" if latest_rsi > 70
                else "Oversold" if latest_rsi < 30
                else "Neutral")

    st.subheader("Price Chart with Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLCV",
        increasing_line_color="#1a6b72",
        decreasing_line_color="#b84c1e"))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_upper"],
        name="BB Upper",
        line=dict(color="#c8a84b",
                  width=1, dash="dash"),
        opacity=0.7))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_lower"],
        name="BB Lower",
        line=dict(color="#c8a84b",
                  width=1, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(200,168,75,0.1)",
        opacity=0.7))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA_20"],
        name="MA 20",
        line=dict(color="#2a9ba5", width=1.5)))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA_50"],
        name="MA 50",
        line=dict(color="#6b6355",
                  width=1.5, dash="dot")))
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            line=dict(color="#1a6b72", width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dash",
                          line_color="#b84c1e")
        fig_rsi.add_hline(y=30, line_dash="dash",
                          line_color="#1a6b72")
        fig_rsi.update_layout(height=300,
                              template="plotly_white")
        st.plotly_chart(fig_rsi,
                        use_container_width=True)
    with col2:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            line=dict(color="#c8a84b", width=1.5)))
        fig_macd.add_hline(y=0, line_dash="dash",
                           line_color="black",
                           line_width=0.5)
        fig_macd.update_layout(height=300,
                               template="plotly_white")
        st.plotly_chart(fig_macd,
                        use_container_width=True)

with tab2:
    st.header("🤖 RL Trading Simulator")
    st.markdown("""
    Simulate different trading strategies on
    the selected asset and date range.
    PPO V3 results shown are from Reliance 2024 backtest.
    """)

    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            min_value=1000, max_value=1000000,
            value=10000, step=1000)
    with col2:
        tc = st.slider(
            "Transaction Cost (%)",
            min_value=0.0, max_value=1.0,
            value=0.1, step=0.05) / 100

    def simulate_buy_hold(df, capital, tc):
        prices    = df["Close"].values
        shares    = capital * (1 - tc) / prices[0]
        portfolio = [capital]
        for price in prices[1:]:
            portfolio.append(shares * price)
        return portfolio

    def simulate_random(df, capital, tc, seed=42):
        np.random.seed(seed)
        prices   = df["Close"].values
        portfolio = [capital]
        cash     = capital
        shares   = 0
        position = 0
        for i in range(1, len(prices)):
            action = np.random.randint(0, 3)
            if action == 1 and position == 0:
                shares   = cash * (1-tc) / prices[i]
                cash     = 0
                position = 1
            elif action == 2 and position == 1:
                cash     = shares * prices[i] * (1-tc)
                shares   = 0
                position = 0
            value = (shares * prices[i]
                    if position == 1 else cash)
            portfolio.append(value)
        return portfolio

    def quick_metrics(portfolio):
        values  = np.array(portfolio)
        returns = np.diff(values) / values[:-1]
        total_r = ((values[-1]-values[0]) /
                   values[0] * 100)
        sharpe  = (np.mean(returns) /
                   np.std(returns) * np.sqrt(252)
                   if np.std(returns) > 0 else 0)
        peak   = np.maximum.accumulate(values)
        max_dd = ((values-peak)/peak).min() * 100
        return total_r, sharpe, max_dd

    bh_port   = simulate_buy_hold(df, initial_capital, tc)
    rand_port = simulate_random(df, initial_capital, tc)
    cons_port = [initial_capital] * len(df)

    bh_r,   bh_s,   bh_d   = quick_metrics(bh_port)
    rand_r, rand_s, rand_d = quick_metrics(rand_port)

    metrics_df = pd.DataFrame({
        "Strategy"     : ["Buy & Hold",
                          "Random Agent",
                          "Conservative (Cash)"],
        "Total Return" : [f"{bh_r:+.2f}%",
                          f"{rand_r:+.2f}%",
                          "0.00%"],
        "Sharpe Ratio" : [f"{bh_s:.3f}",
                          f"{rand_s:.3f}",
                          "0.000"],
        "Max Drawdown" : [f"{bh_d:.2f}%",
                          f"{rand_d:.2f}%",
                          "0.00%"]
    })
    st.subheader("Strategy Comparison")
    st.dataframe(metrics_df,
                 use_container_width=True,
                 hide_index=True)

    min_len = min(len(bh_port), len(rand_port),
                  len(cons_port))
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=df.index[:min_len],
        y=bh_port[:min_len],
        name=f"Buy & Hold ({bh_r:+.1f}%)",
        line=dict(color="#c8a84b", width=2)))
    fig_eq.add_trace(go.Scatter(
        x=df.index[:min_len],
        y=rand_port[:min_len],
        name=f"Random ({rand_r:+.1f}%)",
        line=dict(color="#b84c1e",
                  width=1.5, dash="dot")))
    fig_eq.add_trace(go.Scatter(
        x=df.index[:min_len],
        y=cons_port[:min_len],
        name="Conservative (0.00%)",
        line=dict(color="#1a6b72",
                  width=2, dash="dash")))
    fig_eq.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="black",
        line_width=0.8)
    fig_eq.update_layout(
        height=400,
        template="plotly_white",
        yaxis_title="Portfolio Value (₹)")
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("🏆 PPO V3 Results — Reliance 2024")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", "-5.47%",
                "+12.61% vs Buy&Hold")
    col2.metric("Sharpe Ratio", "-0.745",
                "+0.218 vs Buy&Hold")
    col3.metric("Max Drawdown", "-7.79%",
                "+16.67% vs Buy&Hold")
    col4.metric("Trades Made", "243",
                "38 buys, 205 sells")

with tab3:
    st.header("📊 Complete Model Results")

    st.subheader("Forecasting Results — Reliance 2024")
    forecast_df = pd.DataFrame({
        "Model"           : ["ARIMA Baseline",
                             "LSTM V2",
                             "TFT (Best) 🏆"],
        "MAE"             : [0.011502,
                             0.015139,
                             0.012382],
        "RMSE"            : [0.013421,
                             0.018931,
                             0.018418],
        "Directional Acc" : ["~50%", "49.7%", "53.8%"],
        "Status"          : ["✅ Baseline",
                             "✅ Complete",
                             "🏆 Best"]
    })
    st.dataframe(forecast_df,
                 use_container_width=True,
                 hide_index=True)

    st.subheader("RL Trading Results — Reliance 2024")
    rl_df = pd.DataFrame({
        "Agent"         : ["PPO V1", "PPO V2",
                           "PPO V3 ⭐",
                           "Buy & Hold",
                           "Random"],
        "Total Return"  : ["0.00%", "0.00%",
                           "-5.47%",
                           "-18.08%", "-14.37%"],
        "Sharpe"        : ["0.000", "0.000",
                           "-0.745",
                           "-0.963", "-1.456"],
        "Max Drawdown"  : ["0.00%", "0.00%",
                           "-7.79%",
                           "-24.46%", "-16.07%"],
        "Behaviour"     : ["Cash preservation",
                           "Cash preservation",
                           "Active trading ⭐",
                           "Market benchmark",
                           "Random baseline"]
    })
    st.dataframe(rl_df,
                 use_container_width=True,
                 hide_index=True)

    st.divider()
    st.subheader("👨‍💼 About the Author")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Suman Das**
        Senior Applied Scientist — Financial AI
        - 11+ years Banking (Punjab National Bank)
        - MTech in Intelligent Automation & Robotics (IAR)
        - Jadavpur University | CGPA: 9.79
        - Chartered Engineer (India)
        """)
    with col2:
        st.markdown("""
        **Available for:**
        - Freelance ML projects (Upwork/Fiverr)
        - Remote ML Engineer roles
        - Financial AI consulting

        📧 suman.ju.ai@gmail.com
        🔗 linkedin.com/in/suman-das-6b0749276
        💻 github.com/suman-ju-ai
        """)
