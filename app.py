import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests  # <--- 新增这一行

# 1. 网页标题与布局设置
st.set_page_config(page_title="美股进阶量化回测", layout="wide")
st.title("🦅 美股双均线 + MACD 过滤与风险回测系统")

# 2. 侧边栏：参数输入
st.sidebar.header("回测参数")
# 默认标的改为美股科技巨头苹果 (AAPL)
symbol = st.sidebar.text_input("输入美股代码 (例如: AAPL, TSLA, SPY)", "AAPL")
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

# 3. 核心获取数据逻辑 (加入反爬虫浏览器伪装)
@st.cache_data
def fetch_us_data(code, start, end):
    try:
        # 创建一个自定义的网络会话，把自己伪装成真实的 Chrome 浏览器
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        })
        
        # 使用带有伪装的 session 获取数据
        ticker = yf.Ticker(code, session=session)
        df = ticker.history(start=start, end=end)
        
        if not df.empty:
            df.index = df.index.tz_localize(None)
            df.rename(columns={
                'Open': '开盘', 
                'High': '最高', 
                'Low': '最低', 
                'Close': '收盘', 
                'Volume': '成交量'
            }, inplace=True)
            df = df[['开盘', '最高', '最低', '收盘', '成交量']]
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"⚠️ 数据拉取失败: {e}")
        return pd.DataFrame()

# 4. 执行回测
if st.sidebar.button("运行策略回测"):
    with st.spinner(f"正在从华尔街拉取 {symbol} 真实数据..."):
        data = fetch_us_data(symbol, start_date, end_date)

    if not data.empty:
        # === A. 基础数据与均线计算 ===
        data['每日收益率'] = data['收盘'].pct_change()
        data['Fast_MA'] = data['收盘'].rolling(window=fast_ma_days).mean()
        data['Slow_MA'] = data['收盘'].rolling(window=slow_ma_days).mean()
        
        # === B. MACD 指标计算 ===
        ema_short = data['收盘'].ewm(span=macd_short, adjust=False).mean()
        ema_long = data['收盘'].ewm(span=macd_long, adjust=False).mean()
        data['DIF'] = ema_short - ema_long
        data['DEA'] = data['DIF'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Hist'] = 2 * (data['DIF'] - data['DEA']) 
        
        # === C. 核心交易逻辑：双因子共振 ===
        data['Signal'] = np.where((data['Fast_MA'] > data['Slow_MA']) & (data['MACD_Hist'] > 0), 1, 0)
        
        data['策略每日收益'] = data['Signal'].shift(1) * data['每日收益率']
        data = data.dropna()
        
        # === D. 计算资金净值与最大回撤 ===
        data['基准净值'] = (1 + data['每日收益率']).cumprod()
        data['策略净值'] = (1 + data['策略每日收益']).cumprod()
        
        data['High_Water_Mark'] = data['策略净值'].cummax()
        data['Drawdown'] = (data['策略净值'] - data['High_Water_Mark']) / data['High_Water_Mark']
        
        # === E. 界面渲染与展示 ===
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
            st.metric(label="极限最大回撤 (MDD)", value=f"{max_drawdown:.2f}%", delta="控制风险核心指标", delta_color="inverse")
            st.metric(label="持仓日胜率", value=f"{win_rate:.1f}%")
            
        st.subheader("📋 交易信号与风控明细")
        st.dataframe(data[['收盘', 'Fast_MA', 'Slow_MA', 'MACD_Hist', 'Signal', '策略净值', 'Drawdown']].tail(15))
        
    else:
        st.error("无法获取该股票代码的数据，请检查代码是否正确（如苹果是 AAPL，特斯拉是 TSLA）。")