import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 网页标题与布局设置
st.set_page_config(page_title="全球市场量化研报", layout="wide")
st.title("🌍 专业量化回测与绩效研报系统")

# 2. 侧边栏：参数输入
st.sidebar.header("1. 基础参数")
market_choice = st.sidebar.radio("选择交易市场", ["A股 (沪深)", "美股 (NASDAQ/NYSE)"])

if market_choice == "A股 (沪深)":
    symbol_input = st.sidebar.text_input("输入A股代码 (如: 600519)", "600519")
    symbol = f"{symbol_input}.SS" if symbol_input.startswith("6") else f"{symbol_input}.SZ"
else:
    symbol = st.sidebar.text_input("输入美股代码 (如: AAPL)", "AAPL")

start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.header("2. 策略参数")
fast_ma_days = st.sidebar.number_input("快速均线 (天)", min_value=1, max_value=100, value=5)
slow_ma_days = st.sidebar.number_input("慢速均线 (天)", min_value=2, max_value=250, value=20)
macd_short = st.sidebar.number_input("MACD 短周期", value=12)
macd_long = st.sidebar.number_input("MACD 长周期", value=26)
macd_signal = st.sidebar.number_input("MACD 信号线", value=9)

st.sidebar.markdown("---")
st.sidebar.header("3. 真实环境模拟")
cost_rate_input = st.sidebar.number_input("单边交易综合成本 (千分之)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
trade_cost = cost_rate_input / 1000.0 

# 3. 数据获取引擎
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

# 核心计算引擎
def run_strategy(data_df, fast, slow, m_short, m_long, m_sig, cost):
    df = data_df.copy()
    df['每日收益率'] = df['收盘'].pct_change()
    df['Fast_MA'] = df['收盘'].rolling(window=fast).mean()
    df['Slow_MA'] = df['收盘'].rolling(window=slow).mean()
    
    ema_short = df['收盘'].ewm(span=m_short, adjust=False).mean()
    ema_long = df['收盘'].ewm(span=m_long, adjust=False).mean()
    df['DIF'] = ema_short - ema_long
    df['DEA'] = df['DIF'].ewm(span=m_sig, adjust=False).mean()
    df['MACD_Hist'] = 2 * (df['DIF'] - df['DEA']) 
    
    df['Signal'] = np.where((df['Fast_MA'] > df['Slow_MA']) & (df['MACD_Hist'] > 0), 1, 0)
    df['Trade_Action'] = df['Signal'].diff().abs().fillna(0)
    
    df['策略每日毛收益'] = df['Signal'].shift(1) * df['每日收益率']
    df['策略每日收益'] = df['策略每日毛收益'] - (df['Trade_Action'] * cost)
    df = df.dropna()
    
    if df.empty:
        return df
        
    df['基准净值'] = (1 + df['每日收益率']).cumprod()
    df['策略净值'] = (1 + df['策略每日收益']).cumprod()
    df['High_Water_Mark'] = df['策略净值'].cummax()
    df['Drawdown'] = (df['策略净值'] - df['High_Water_Mark']) / df['High_Water_Mark']
    
    return df

# 4. 界面展示：双标签页
tab1, tab2 = st.tabs(["📑 量化绩效研报", "🤖 智能参数寻优"])

data = fetch_global_data(symbol, start_date, end_date)

with tab1:
    if st.button("▶️ 生成策略研报"):
        if not data.empty:
            result_df = run_strategy(data, fast_ma_days, slow_ma_days, macd_short, macd_long, macd_signal, trade_cost)
            
            # === 研报级绩效指标计算 ===
            trading_days = len(result_df)
            years = trading_days / 252
            
            # 累计收益
            strat_ret = (result_df['策略净值'].iloc[-1] - 1)
            base_ret = (result_df['基准净值'].iloc[-1] - 1)
            
            # 年化收益 (CAGR)
            ann_strat_ret = ((1 + strat_ret) ** (1 / years) - 1) if years > 0 else 0
            
            # 夏普比率 (假设无风险利率为 0)
            daily_mean = result_df['策略每日收益'].mean()
            daily_std = result_df['策略每日收益'].std()
            sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
            
            # 索提诺比率 (仅计算亏损日的标准差)
            downside_returns = result_df[result_df['策略每日收益'] < 0]['策略每日收益']
            downside_std = downside_returns.std()
            sortino_ratio = (daily_mean / downside_std) * np.sqrt(252) if downside_std != 0 else 0
            
            max_dd = result_df['Drawdown'].min()
            total_trades = result_df['Trade_Action'].sum()
            
            st.success("研报生成完毕！")
            
            # --- 第一部分：核心指标看板 ---
            st.markdown("### 🔬 核心量化指标 (Alpha & Risk)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("策略累计净收益", f"{strat_ret*100:.2f}%", f"基准: {base_ret*100:.2f}%")
            m2.metric("年化收益率 (CAGR)", f"{ann_strat_ret*100:.2f}%")
            m3.metric("极限最大回撤", f"{max_dd*100:.2f}%", delta="风控指标", delta_color="inverse")
            m4.metric("夏普比率 (Sharpe)", f"{sharpe_ratio:.2f}", "每单位总风险收益")
            m5.metric("索提诺比率 (Sortino)", f"{sortino_ratio:.2f}", "每单位下行风险收益")
            
            st.markdown("---")
            
            # --- 第二部分：净值与回撤图 ---
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📈 资金净值走势图 (已扣除交易成本)")
                st.line_chart(result_df[['基准净值', '策略净值']])
                st.markdown("#### 📉 动态回撤面积图")
                st.area_chart(result_df['Drawdown'] * 100)
                
            with col2:
                # --- 第三部分：月度收益热力图 ---
                st.markdown("#### 📅 策略月度收益分布表")
                # 计算每月的收益率
                result_df['Year'] = result_df.index.year
                result_df['Month'] = result_df.index.month
                
                # 月度累计收益率 = 当月连乘
                monthly_ret = result_df.groupby(['Year', 'Month'])['策略每日收益'].apply(lambda x: (1 + x).prod() - 1).unstack()
                
                # 使用 Pandas Styler 生成高大上的热力图表
                styled_monthly = monthly_ret.style.format("{:.2%}", na_rep="-") \
                    .background_gradient(cmap='RdYlGn', axis=None, vmin=-0.15, vmax=0.15) \
                    .highlight_null(color='#f0f2f6')
                
                st.dataframe(styled_monthly, use_container_width=True, height=400)
                
                st.info(f"💡 **交易摘要：** 回测区间内共发生 **{int(total_trades)}** 次买卖操作。按单边 {cost_rate_input}‰ 计算，累计付出的摩擦成本约占本金的 **{total_trades * cost_rate_input:.1f}‰**。")

        else:
            st.error("获取数据失败，请检查代码。")

with tab2:
    st.markdown("### 🔍 寻找抗磨损的最优均线组合 (含夏普评估)")
    # (保留你之前的网格搜索逻辑，只需将 run_strategy 的返回值接好即可，为节约篇幅此处省略)
    st.info("如需在此页面也加上夏普比率的排序，可在循环中提取计算好的 Sharpe 字段。")