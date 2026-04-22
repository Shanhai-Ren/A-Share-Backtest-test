import os
# 强制清除所有可能残留的代理环境变量
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['all_proxy'] = ''
os.environ['ALL_PROXY'] = ''

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np

# 1. 网页标题与布局设置
st.set_page_config(page_title="A股进阶量化回测", layout="wide")
st.title("📈 A股双均线 + MACD 过滤与风险回测系统")

# 2. 侧边栏：参数输入
st.sidebar.header("回测参数")
symbol = st.sidebar.text_input("输入A股代码 (例如: 600519)", "600519")
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

# 3. 核心获取数据逻辑
@st.cache_data
def fetch_a_share_data(code, start, end):
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        if not df.empty:
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"⚠️ 开发者调试信息 - 详细报错: {e}")
        return pd.DataFrame()

# 4. 执行回测
if st.sidebar.button("运行策略回测"):
    with st.spinner(f"正在拉取 {symbol} 数据并计算策略..."):
        data = fetch_a_share_data(symbol, start_date, end_date)

    if not data.empty:
        # === A. 基础数据与均线计算 ===
        data['每日收益率'] = data['收盘'].pct_change()
        data['Fast_MA'] = data['收盘'].rolling(window=fast_ma_days).mean()
        data['Slow_MA'] = data['收盘'].rolling(window=slow_ma_days).mean()
        
        # === B. MACD 指标计算 (指数移动平均 EMA) ===
        ema_short = data['收盘'].ewm(span=macd_short, adjust=False).mean()
        ema_long = data['收盘'].ewm(span=macd_long, adjust=False).mean()
        data['DIF'] = ema_short - ema_long
        data['DEA'] = data['DIF'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Hist'] = 2 * (data['DIF'] - data['DEA']) # MACD 柱状图
        
        # === C. 核心交易逻辑：双因子共振 ===
        # 条件1: 快速均线 > 慢速均线 (均线多头)
        # 条件2: MACD_Hist > 0 (动量多头)
        # 满足以上两者才持仓(1)，否则空仓(0)
        data['Signal'] = np.where((data['Fast_MA'] > data['Slow_MA']) & (data['MACD_Hist'] > 0), 1, 0)
        
        # 计算策略收益率 (用昨天的信号吃今天的涨跌幅)
        data['策略每日收益'] = data['Signal'].shift(1) * data['每日收益率']
        data = data.dropna()
        
        # === D. 计算资金净值与最大回撤 ===
        data['基准净值'] = (1 + data['每日收益率']).cumprod()
        data['策略净值'] = (1 + data['策略每日收益']).cumprod()
        
        # 计算最高水位线 (High Water Mark)
        data['High_Water_Mark'] = data['策略净值'].cummax()
        # 计算每日回撤 (当前净值 - 历史最高点) / 历史最高点
        data['Drawdown'] = (data['策略净值'] - data['High_Water_Mark']) / data['High_Water_Mark']
        
        # === E. 界面渲染与展示 ===
        st.success("回测完成！")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📊 资金净值走势：策略 vs 一直持有")
            st.line_chart(data[['基准净值', '策略净值']])
            
            st.subheader("📉 策略动态回撤曲线")
            # 画出回撤图，直观感受资金缩水情况 (乘100转换为百分比)
            st.area_chart(data['Drawdown'] * 100)
            
        with col2:
            st.subheader("🏆 风险与业绩评估")
            base_return = (data['基准净值'].iloc[-1] - 1) * 100
            strategy_return = (data['策略净值'].iloc[-1] - 1) * 100
            
            # 核心风险指标：最大回撤
            max_drawdown = data['Drawdown'].min() * 100
            
            # 胜率计算
            winning_days = len(data[data['策略每日收益'] > 0])
            total_trading_days = len(data[data['Signal'].shift(1) == 1])
            win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0
            
            st.metric(label="策略累计收益", value=f"{strategy_return:.2f}%")
            st.metric(label="基准累计收益", value=f"{base_return:.2f}%")
            # 使用红色高亮显示最大回撤风险
            st.metric(label="极限最大回撤 (MDD)", value=f"{max_drawdown:.2f}%", delta="控制风险核心指标", delta_color="inverse")
            st.metric(label="持仓日胜率", value=f"{win_rate:.1f}%")
            
        st.subheader("📋 交易信号与风控明细")
        # 展示包含了 MACD 和 回撤 的详尽数据表
        st.dataframe(data[['收盘', 'Fast_MA', 'Slow_MA', 'MACD_Hist', 'Signal', '策略净值', 'Drawdown']].tail(15))
        
    else:
        st.error("数据拉取失败。")