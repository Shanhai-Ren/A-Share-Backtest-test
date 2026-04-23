import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. 网页标题与布局设置
st.set_page_config(page_title="全球市场量化回测 (实战版)", layout="wide")
st.title("🌍 真实市场双因子回测 (含摩擦成本)")

# 2. 侧边栏：参数输入
st.sidebar.header("1. 基础参数")
market_choice = st.sidebar.radio("选择交易市场", ["A股 (沪深)", "美股 (NASDAQ/NYSE)"])

if market_choice == "A股 (沪深)":
    symbol_input = st.sidebar.text_input("输入A股6位代码 (如: 600519)", "600519")
    symbol = f"{symbol_input}.SS" if symbol_input.startswith("6") else f"{symbol_input}.SZ"
else:
    symbol = st.sidebar.text_input("输入美股代码 (如: AAPL)", "AAPL")

start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.header("2. 策略与风控参数")
st.sidebar.subheader("趋势: 均线参数")
fast_ma_days = st.sidebar.number_input("快速均线 (天)", min_value=1, max_value=100, value=5)
slow_ma_days = st.sidebar.number_input("慢速均线 (天)", min_value=2, max_value=250, value=20)

st.sidebar.subheader("动量: MACD参数")
macd_short = st.sidebar.number_input("短周期", value=12)
macd_long = st.sidebar.number_input("长周期", value=26)
macd_signal = st.sidebar.number_input("信号线", value=9)

st.sidebar.markdown("---")
st.sidebar.header("3. 真实环境模拟")
# 加入单边交易成本设置（千分之几），A股通常包含万分之几的佣金 + 千分之一的印花税(仅卖出) + 滑点
cost_rate_input = st.sidebar.number_input("单边交易综合成本 (千分之)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
trade_cost = cost_rate_input / 1000.0  # 转换为小数计算

# 3. 核心获取数据逻辑
@st.cache_data
def fetch_global_data(code, start, end):
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(start=start, end=end)
        if not df.empty:
            df.index = df.index.tz_localize(None)
            df.rename(columns={'Open': '开盘', 'High': '最高', 'Low': '最低', 'Close': '收盘', 'Volume': '成交量'}, inplace=True)
            return df[['开盘', '最高', '最低', '收盘', '成交量']]
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# 4. 执行回测
if st.sidebar.button("运行实战回测"):
    with st.spinner(f"正在拉取 {symbol_input if market_choice == 'A股 (沪深)' else symbol} 数据并模拟摩擦成本..."):
        data = fetch_global_data(symbol, start_date, end_date)

    if not data.empty:
        # === A. 指标计算 ===
        data['每日收益率'] = data['收盘'].pct_change()
        data['Fast_MA'] = data['收盘'].rolling(window=fast_ma_days).mean()
        data['Slow_MA'] = data['收盘'].rolling(window=slow_ma_days).mean()
        
        ema_short = data['收盘'].ewm(span=macd_short, adjust=False).mean()
        ema_long = data['收盘'].ewm(span=macd_long, adjust=False).mean()
        data['DIF'] = ema_short - ema_long
        data['DEA'] = data['DIF'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Hist'] = 2 * (data['DIF'] - data['DEA']) 
        
        # === B. 交易信号与【真实成本扣除】 ===
        data['Signal'] = np.where((data['Fast_MA'] > data['Slow_MA']) & (data['MACD_Hist'] > 0), 1, 0)
        
        # 判断是否发生交易：今天的信号减去昨天的信号，取绝对值。
        # 1代表发生买入或卖出，0代表持仓不动或空仓不动
        data['Trade_Action'] = data['Signal'].diff().abs().fillna(0)
        
        # 计算毛收益 (未扣除成本)
        data['策略每日毛收益'] = data['Signal'].shift(1) * data['每日收益率']
        
        # 计算净收益 (毛收益 - 交易当天的综合成本)
        data['策略每日收益'] = data['策略每日毛收益'] - (data['Trade_Action'] * trade_cost)
        
        data = data.dropna()
        
        # === C. 净值与回撤 ===
        data['基准净值'] = (1 + data['每日收益率']).cumprod()
        data['策略净值'] = (1 + data['策略每日收益']).cumprod()
        data['High_Water_Mark'] = data['策略净值'].cummax()
        data['Drawdown'] = (data['策略净值'] - data['High_Water_Mark']) / data['High_Water_Mark']
        
        # === D. 界面渲染 ===
        st.success("回测完成！")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📊 资金净值走势 (已扣除交易成本)")
            st.line_chart(data[['基准净值', '策略净值']])
            st.subheader("📉 策略动态回撤曲线")
            st.area_chart(data['Drawdown'] * 100)
            
        with col2:
            st.subheader("🏆 实战业绩评估")
            base_return = (data['基准净值'].iloc[-1] - 1) * 100
            strategy_return = (data['策略净值'].iloc[-1] - 1) * 100
            max_drawdown = data['Drawdown'].min() * 100
            
            # 统计买卖总次数
            total_trades = data['Trade_Action'].sum()
            
            winning_days = len(data[data['策略每日收益'] > 0])
            total_trading_days = len(data[data['Signal'].shift(1) == 1])
            win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0
            
            st.metric(label="策略累计净收益", value=f"{strategy_return:.2f}%")
            st.metric(label="基准累计收益", value=f"{base_return:.2f}%")
            st.metric(label="极限最大回撤 (MDD)", value=f"{max_drawdown:.2f}%", delta="实盘风控", delta_color="inverse")
            # 新增交易频率统计
            st.metric(label="买卖交易总次数", value=f"{int(total_trades)} 次", delta=f"总磨损成本约 {total_trades * cost_rate_input:.1f}‰", delta_color="inverse")
            
    else:
        st.error("获取数据失败，请重试。")