import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. 网页标题与布局设置
st.set_page_config(page_title="量化回测与参数寻优", layout="wide")
st.title("🌍 真实市场双因子回测 & 智能参数寻优")

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
st.sidebar.header("2. 策略参数 (仅用于单次回测)")
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

# 核心计算引擎 (封装成函数，方便单次回测和批量寻优调用)
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
        return df, 0, 0, 0, 0
        
    df['基准净值'] = (1 + df['每日收益率']).cumprod()
    df['策略净值'] = (1 + df['策略每日收益']).cumprod()
    df['High_Water_Mark'] = df['策略净值'].cummax()
    df['Drawdown'] = (df['策略净值'] - df['High_Water_Mark']) / df['High_Water_Mark']
    
    strategy_return = (df['策略净值'].iloc[-1] - 1) * 100
    base_return = (df['基准净值'].iloc[-1] - 1) * 100
    max_drawdown = df['Drawdown'].min() * 100
    total_trades = df['Trade_Action'].sum()
    
    return df, strategy_return, base_return, max_drawdown, total_trades

# 4. 界面展示：引入双标签页设计
tab1, tab2 = st.tabs(["📈 单次策略回测", "🤖 智能参数寻优 (网格搜索)"])

# 准备底层数据
data = fetch_global_data(symbol, start_date, end_date)

with tab1:
    if st.button("▶️ 运行单次实战回测"):
        if not data.empty:
            result_df, strat_ret, base_ret, max_dd, trades = run_strategy(
                data, fast_ma_days, slow_ma_days, macd_short, macd_long, macd_signal, trade_cost
            )
            
            st.success("回测完成！")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("📊 资金净值走势 (已扣除交易成本)")
                st.line_chart(result_df[['基准净值', '策略净值']])
                st.subheader("📉 策略动态回撤曲线")
                st.area_chart(result_df['Drawdown'] * 100)
            with col2:
                st.subheader("🏆 业绩评估")
                st.metric("策略累计净收益", f"{strat_ret:.2f}%")
                st.metric("基准累计收益", f"{base_ret:.2f}%")
                st.metric("极限最大回撤", f"{max_dd:.2f}%", delta="实盘风控", delta_color="inverse")
                st.metric("买卖交易总次数", f"{int(trades)} 次", delta=f"摩擦成本约 {trades * cost_rate_input:.1f}‰", delta_color="inverse")
        else:
            st.error("获取数据失败，请检查代码。")

with tab2:
    st.markdown("### 🔍 寻找抗磨损的最优均线组合")
    st.write(f"系统将在扣除 **{cost_rate_input}‰** 摩擦成本的前提下，自动遍历不同的均线组合，寻找最优解。")
    
    if st.button("🚀 开始全自动参数寻优"):
        if not data.empty:
            with st.spinner("正在启动矩阵运算，遍历历史数据..."):
                results = []
                # 我们测试 5到20 的快线，和 20到60 的慢线组合
                fast_options = [5, 10, 15]
                slow_options = [20, 30, 40, 60]
                
                # 嵌套循环：穷举所有组合
                for f in fast_options:
                    for s in slow_options:
                        if f >= s: continue # 排除快线大于慢线的不合理组合
                        
                        _, strat_ret, _, max_dd, trades = run_strategy(
                            data, f, s, macd_short, macd_long, macd_signal, trade_cost
                        )
                        
                        results.append({
                            "快线 (天)": f,
                            "慢线 (天)": s,
                            "净收益率 (%)": round(strat_ret, 2),
                            "最大回撤 (%)": round(max_dd, 2),
                            "交易次数": int(trades)
                        })
                
                # 将结果转为数据表，并按收益率从高到低排序
                results_df = pd.DataFrame(results).sort_values(by="净收益率 (%)", ascending=False)
                results_df.reset_index(drop=True, inplace=True)
                
                st.success("寻优完成！以下是扣除手续费后的最强参数组合排行榜：")
                
                # 高亮显示收益最高的第一名
                st.dataframe(
                    results_df.style.highlight_max(subset=['净收益率 (%)'], color='lightgreen')
                                    .highlight_min(subset=['最大回撤 (%)'], color='lightcoral'),
                    use_container_width=True
                )
                
                best_fast = results_df.iloc[0]['快线 (天)']
                best_slow = results_df.iloc[0]['慢线 (天)']
                st.info(f"💡 **结论建议：** 针对该标的，在当前市场摩擦成本下，历史最优均线组合为 **{best_fast}日线 / {best_slow}日线**。你可以回到左侧输入这两个参数，在『单次策略回测』中查看它的具体净值走势。")
        else:
            st.error("请先确认股票数据能够正常拉取。")