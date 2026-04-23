import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. 网页标题与布局设置
st.set_page_config(page_title="全球市场量化回测", layout="wide")
st.title("🌍 A股/美股 双均线 + MACD 进阶回测系统")

# 2. 侧边栏：参数输入
st.sidebar.header("回测参数")

# 市场选择器：让系统自动处理不同市场的代码规则
market_choice = st.sidebar.radio("选择交易市场", ["A股 (沪深)", "美股 (NASDAQ/NYSE)"])

if market_choice == "A股 (沪深)":
    symbol_input = st.sidebar.text_input("输入A股6位代码 (例如: 600519, 000001)", "600519")
    # 自动判断沪市还是深市并添加后缀 (6开头是沪市，0或3开头是深市)
    if symbol_input.startswith("6"):
        symbol = f"{symbol_input}.SS"
    else:
        symbol = f"{symbol_input}.SZ"
else:
    symbol = st.sidebar.text_input("输入美股代码 (例如: AAPL, TSLA)", "AAPL")

start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.subheader("趋势指标: 均线参数")
fast_ma_days = st.sidebar.number_input("快速均线 (天)", min_value=1, max_value=100, value=5)
slow_ma_days = st.sidebar.number_input("慢速均线 (天)", min_value=2, max_value=250, value=20)

st.sidebar.subheader("动量指标: MACD参数")
macd_short = st.sidebar.number_input("MACD 短周期", value=12)
macd_long = st.sidebar.number_input("MACD 长周期", value=26)
macd_signal = st.sidebar.number_input("MACD 信号线", value=9)

# 3. 核心获取数据逻辑 (利用 yfinance 跨越国界)
@st.cache_data
def fetch_global_data(code, start, end):
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(start=start, end=end)
        
        if not df.empty:
            df.index = df.index.tz_localize(None)
            df.rename(columns={
                'Open': '开盘', 'High': '最高', 'Low': '最低', 
                'Close': '收盘', 'Volume': '成交量'
            }, inplace=True)
            return df[['开盘', '最高', '最低', '收盘', '成交量']]
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ 数据拉取失败: {e}")
        return pd.DataFrame()

# 4. 执行回测
if st.sidebar.button("运行策略回测"):
    with st.spinner(f"正在从全球接口拉取 {symbol_input if market_choice == 'A股 (沪深)' else symbol} 数据..."):
        data = fetch_global_data(symbol, start_date, end_date)

    if not data.empty:
        # === 策略核心逻辑 ===
        data['每日收益率'] = data['收盘'].pct_change()
        data['Fast_MA'] = data['收盘'].rolling(window=fast_ma_days).mean()
        data['Slow_MA'] = data['收盘'].rolling(window=slow_ma_days).mean()
        
        ema_short = data['收盘'].ewm(span=macd_short, adjust=False).mean()
        ema_long = data['收盘'].ewm(span=macd_long, adjust=False).mean()
        data['DIF'] = ema_short - ema_long
        data['DEA'] = data['DIF'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Hist'] = 2 * (data['DIF'] - data['DEA']) 
        
        data['Signal'] = np.where((data['Fast_MA'] > data['Slow_MA']) & (data['MACD_Hist'] > 0), 1, 0)
        data['策略每日收益'] = data['Signal'].shift(1) * data['每日收益率']
        data = data.dropna()
        
        data['基准净值'] = (1 + data['每日收益率']).cumprod()
        data['策略净值'] = (1 + data['策略每日收益']).cumprod()
        
        data['High_Water_Mark'] = data['策略净值'].cummax()
        data['Drawdown'] = (data['策略净值'] - data['High_Water_Mark']) / data['High_Water_Mark']
        
        # === 界面渲染 ===
        st.success("回测完成！")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📊 资金净值走势：策略 vs 一直持有")
            st.line_chart(data[['基准净值', '策略净值']])
            st.subheader("📉 策略动态回撤曲线")
            st.area_chart(data['Drawdown'] * 100)
            
        with col2:
            st.subheader("🏆 风险与业绩评估")
            base_return = (data['基准净值'].iloc[-1] - 1) * 100
            strategy_return = (data['策略净值'].iloc[-1] - 1) * 100
            max_drawdown = data['Drawdown'].min() * 100
            
            winning_days = len(data[data['策略每日收益'] > 0])
            total_trading_days = len(data[data['Signal'].shift(1) == 1])
            win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0
            
            st.metric(label="策略累计收益", value=f"{strategy_return:.2f}%")
            st.metric(label="基准累计收益", value=f"{base_return:.2f}%")
            st.metric(label="极限最大回撤 (MDD)", value=f"{max_drawdown:.2f}%", delta="风险控制", delta_color="inverse")
            st.metric(label="持仓日胜率", value=f"{win_rate:.1f}%")
            
    else:
        st.error("无法获取数据，请检查代码是否正确或是否停牌。")