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
st.sidebar.header("3. 风控与仓位管理 (ATR)")
atr_period = st.sidebar.number_input("ATR 计算周期", value=14)
atr_multi = st.sidebar.number_input("ATR 止损乘数 (X倍ATR)", value=2.0, step=0.1)
max_pos = st.sidebar.slider("首次建仓比例", min_value=0.1, max_value=1.0, value=0.5, step=0.1, help="0.5表示首次只买半仓，盈利后加仓")

st.sidebar.markdown("---")
st.sidebar.header("4. 真实环境模拟") 
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
# 核心计算引擎 (已加入 ATR 止损与分批建仓)
def run_strategy(data_df, fast, slow, m_short, m_long, m_sig, cost, atr_period, atr_multi, max_pos):
    df = data_df.copy()
    df['每日收益率'] = df['收盘'].pct_change()
    df['Fast_MA'] = df['收盘'].rolling(window=fast).mean()
    df['Slow_MA'] = df['收盘'].rolling(window=slow).mean()
    
    ema_short = df['收盘'].ewm(span=m_short, adjust=False).mean()
    ema_long = df['收盘'].ewm(span=m_long, adjust=False).mean()
    df['DIF'] = ema_short - ema_long
    df['DEA'] = df['DIF'].ewm(span=m_sig, adjust=False).mean()
    df['MACD_Hist'] = 2 * (df['DIF'] - df['DEA']) 
    
    # === 新增：计算 ATR (真实波动幅度) ===
    df['Prev_Close'] = df['收盘'].shift(1)
    df['TR'] = np.maximum(df['最高'] - df['最低'],
               np.maximum(abs(df['最高'] - df['Prev_Close']),
                          abs(df['最低'] - df['Prev_Close'])))
    df['ATR'] = df['TR'].rolling(window=int(atr_period)).mean()
    
    # === 新增：状态机模拟 (仓位管理与动态止损) ===
    close_arr = df['收盘'].values
    low_arr = df['最低'].values
    fast_ma = df['Fast_MA'].values
    slow_ma = df['Slow_MA'].values
    macd_h = df['MACD_Hist'].values
    atr_arr = df['ATR'].fillna(0).values
    
    positions = np.zeros(len(df))
    entry_price = 0.0
    stop_loss = 0.0
    
    for i in range(1, len(df)):
        buy_signal = (fast_ma[i-1] > slow_ma[i-1]) and (macd_h[i-1] > 0)
        sell_signal = (fast_ma[i-1] < slow_ma[i-1])
        current_pos = positions[i-1]
        
        if current_pos == 0:
            if buy_signal:
                positions[i] = max_pos # 首次建仓指定比例
                entry_price = close_arr[i]
                stop_loss = entry_price - atr_multi * atr_arr[i-1] # 初始止损线
        elif current_pos > 0:
            # 1. 检查是否触发止损 (最低价击穿止损线)
            if low_arr[i] < stop_loss:
                positions[i] = 0 # 止损清仓
                entry_price = 0.0
            # 2. 检查是否触发常规平仓信号
            elif sell_signal:
                positions[i] = 0
                entry_price = 0.0
            # 3. 仍在场内，进行移动止损与加仓
            else:
                # 动态上移止损线 (Trailing Stop)
                new_stop = close_arr[i-1] - atr_multi * atr_arr[i-1]
                if new_stop > stop_loss:
                    stop_loss = new_stop
                
                # 盈利加仓逻辑：如果收盘价突破成本价+1倍ATR，且没满仓，则加满
                if close_arr[i] > entry_price + atr_arr[i-1] and current_pos < 1.0:
                    positions[i] = 1.0 
                    entry_price = close_arr[i] # 更新均价
                else:
                    positions[i] = current_pos
                    
    df['Position'] = positions
    df['Trade_Action'] = df['Position'].diff().abs().fillna(0)
    
    # 将原来的 df['Signal'] 替换为 df['Position'] 计算资金曲线
    df['策略每日毛收益'] = df['Position'].shift(1) * df['每日收益率']
    df['策略每日收益'] = df['策略每日毛收益'] - (df['Trade_Action'] * cost)
    df = df.dropna()
    
    if df.empty:
        return df
        
    df['基准净值'] = (1 + df['每日收益率']).cumprod()
    df['策略净值'] = (1 + df['策略每日收益']).cumprod()
    df['High_Water_Mark'] = df['策略净值'].cummax()
    df['Drawdown'] = (df['策略净值'] - df['High_Water_Mark']) / df['High_Water_Mark']
    
    return df

# ================= 新增拼接部分 1：多资产引擎 =================
@st.cache_data
def fetch_portfolio_data(symbol_list, start, end):
    data_dict = {}
    for code in symbol_list:
        try:
            ticker = yf.Ticker(code)
            df = ticker.history(start=start, end=end)
            if not df.empty:
                df.index = df.index.tz_localize(None)
                df.rename(columns={'Open': '开盘', 'High': '最高', 'Low': '最低', 'Close': '收盘', 'Volume': '成交量'}, inplace=True)
                data_dict[code] = df[['开盘', '最高', '最低', '收盘', '成交量']]
        except Exception:
            pass
    return data_dict

def run_portfolio_strategy(data_dict, fast, slow, m_short, m_long, m_sig, cost, atr_period, atr_multi, max_pos):
    """
    详尽版多资产投资组合回测引擎
    包含：完整技术指标计算、单票状态机（ATR动态止损 + 分批加仓）、等权组合汇总
    """
    strategy_returns = []
    benchmark_returns = []
    total_trades = 0
    
    # 遍历资产池中的每一只股票
    for code, df in data_dict.items():
        temp_df = df.copy()
        
        # ================= 1. 基础价格与均线指标 =================
        temp_df['每日收益率'] = temp_df['收盘'].pct_change()
        temp_df['Fast_MA'] = temp_df['收盘'].rolling(window=fast).mean()
        temp_df['Slow_MA'] = temp_df['收盘'].rolling(window=slow).mean()
        
        # ================= 2. MACD 指标计算 =================
        ema_short = temp_df['收盘'].ewm(span=m_short, adjust=False).mean()
        ema_long = temp_df['收盘'].ewm(span=m_long, adjust=False).mean()
        temp_df['DIF'] = ema_short - ema_long
        temp_df['DEA'] = temp_df['DIF'].ewm(span=m_sig, adjust=False).mean()
        temp_df['MACD_Hist'] = 2 * (temp_df['DIF'] - temp_df['DEA']) 
        
        # ================= 3. ATR (真实波动幅度) 计算 =================
        temp_df['Prev_Close'] = temp_df['收盘'].shift(1)
        temp_df['TR'] = np.maximum(temp_df['最高'] - temp_df['最低'],
                        np.maximum(abs(temp_df['最高'] - temp_df['Prev_Close']),
                                   abs(temp_df['最低'] - temp_df['Prev_Close'])))
        temp_df['ATR'] = temp_df['TR'].rolling(window=int(atr_period)).mean()
        
        # ================= 4. 状态机：仓位管理与动态止损 =================
        # 为了提高回测速度，将 pandas Series 转换为 numpy 数组进行 for 循环计算
        close_arr = temp_df['收盘'].values
        low_arr = temp_df['最低'].values
        fast_ma = temp_df['Fast_MA'].values
        slow_ma = temp_df['Slow_MA'].values
        macd_h = temp_df['MACD_Hist'].values
        atr_arr = temp_df['ATR'].fillna(0).values
        
        positions = np.zeros(len(temp_df))
        entry_price = 0.0
        stop_loss = 0.0
        
        for i in range(1, len(temp_df)):
            # 信号判定逻辑
            buy_signal = (fast_ma[i-1] > slow_ma[i-1]) and (macd_h[i-1] > 0)
            sell_signal = (fast_ma[i-1] < slow_ma[i-1])
            current_pos = positions[i-1]
            
            # --- 空仓状态 ---
            if current_pos == 0:
                if buy_signal:
                    positions[i] = max_pos  # 首次建仓指定比例 (如 0.5)
                    entry_price = close_arr[i]
                    stop_loss = entry_price - atr_multi * atr_arr[i-1] # 设定初始止损线
            
            # --- 持仓状态 ---
            elif current_pos > 0:
                # 检查1：是否被动触发止损 (盘中最低价击穿止损线)
                if low_arr[i] < stop_loss:
                    positions[i] = 0  # 止损清仓
                    entry_price = 0.0
                    
                # 检查2：是否主动触发常规平仓 (均线死叉)
                elif sell_signal:
                    positions[i] = 0  # 信号平仓
                    entry_price = 0.0
                    
                # 检查3：仍在场内，执行移动止损保护与顺势加仓
                else:
                    # 动态上移止损线 (Trailing Stop)：确保止损线只上移，不下降
                    new_stop = close_arr[i-1] - atr_multi * atr_arr[i-1]
                    if new_stop > stop_loss:
                        stop_loss = new_stop
                    
                    # 盈利加仓：如果当前收盘价脱离成本区 (超过1倍ATR)，且还有加仓空间，则加满到 1.0
                    if close_arr[i] > entry_price + atr_arr[i-1] and current_pos < 1.0:
                        positions[i] = 1.0 
                        entry_price = close_arr[i] # 动态更新持仓均价
                    else:
                        # 保持原仓位
                        positions[i] = current_pos
                        
        # 将 numpy 数组的结果写回 DataFrame
        temp_df['Position'] = positions
        temp_df['Trade_Action'] = temp_df['Position'].diff().abs().fillna(0)
        
        # ================= 5. 单票收益率计算 =================
        # 当日收益 = 前一日的持仓比例 * 标的当日涨跌幅 - 交易摩擦成本
        temp_df['策略每日收益'] = (temp_df['Position'].shift(1) * temp_df['每日收益率']) - (temp_df['Trade_Action'] * cost)
        
        # 收集该标的收益序列与换手次数
        strategy_returns.append(temp_df['策略每日收益'].rename(code))
        benchmark_returns.append(temp_df['每日收益率'].rename(code))
        total_trades += temp_df['Trade_Action'].sum()
        
    # 如果没有有效数据，直接返回空表
    if not strategy_returns: 
        return pd.DataFrame()
    
    # ================= 6. 组合层面的汇总与统计 =================
    port_df = pd.DataFrame()
    # 等权重组合：直接将多列收益率横向取平均
    port_df['投资组合每日收益'] = pd.concat(strategy_returns, axis=1).mean(axis=1)
    port_df['基准每日收益 (等权)'] = pd.concat(benchmark_returns, axis=1).mean(axis=1)
    port_df = port_df.dropna()
    
    if port_df.empty: 
        return port_df
        
    # 计算资金曲线与最大回撤
    port_df['基准组合净值'] = (1 + port_df['基准每日收益 (等权)']).cumprod()
    port_df['策略组合净值'] = (1 + port_df['投资组合每日收益']).cumprod()
    port_df['High_Water_Mark'] = port_df['策略组合净值'].cummax()
    port_df['Drawdown'] = (port_df['策略组合净值'] - port_df['High_Water_Mark']) / port_df['High_Water_Mark']
    port_df['Trade_Action_Sum'] = total_trades 
    
    return port_df

# 4. 界面展示：双标签页
tab1, tab2, tab3 = st.tabs(["📑 量化绩效研报", "🤖 智能参数寻优", "🌍 多资产投资组合"])

data = fetch_global_data(symbol, start_date, end_date)

with tab1:
    if st.button("▶️ 生成策略研报"):
        if not data.empty:
            result_df = run_strategy(data, fast_ma_days, slow_ma_days, macd_short, macd_long, macd_signal, trade_cost, atr_period, atr_multi, max_pos)
            
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
    st.write(f"系统将在扣除 **{cost_rate_input}‰** 摩擦成本的前提下，自动遍历不同的均线组合，并计算风险回报比。")
    
    if st.button("🚀 开始全自动参数寻优"):
        if not data.empty:
            with st.spinner("正在启动矩阵运算，遍历历史数据..."):
                results = []
                # 测试参数范围：快线 5-15，慢线 20-60
                fast_options = [5, 10, 15]
                slow_options = [20, 30, 40, 60]
                
                for f in fast_options:
                    for s in slow_options:
                        if f >= s: continue # 排除不合理的组合
                        
                        # 调用核心引擎跑回测
                        result_df = run_strategy(data, f, s, macd_short, macd_long, macd_signal, trade_cost, atr_period, atr_multi, max_pos)
                        
                        if not result_df.empty:
                            # 提取收益与回撤
                            strat_ret = (result_df['策略净值'].iloc[-1] - 1) * 100
                            max_dd = result_df['Drawdown'].min() * 100
                            trades = result_df['Trade_Action'].sum()
                            
                            # 计算夏普比率 (核心风控指标)
                            daily_mean = result_df['策略每日收益'].mean()
                            daily_std = result_df['策略每日收益'].std()
                            sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
                            
                            results.append({
                                "快线 (天)": f,
                                "慢线 (天)": s,
                                "净收益率 (%)": round(strat_ret, 2),
                                "最大回撤 (%)": round(max_dd, 2),
                                "夏普比率": round(sharpe, 2), # 新增夏普字段
                                "交易次数": int(trades)
                            })
                
                # 将结果转为数据表，这次按【夏普比率】从高到低排序
                results_df = pd.DataFrame(results).sort_values(by="夏普比率", ascending=False)
                results_df.reset_index(drop=True, inplace=True)
                
                st.success("寻优完成！以下是按【夏普比率】排名的最强参数组合：")
                
                # 高亮显示表现最好的数值
                st.dataframe(
                    results_df.style.highlight_max(subset=['夏普比率', '净收益率 (%)'], color='lightgreen')
                                    .highlight_min(subset=['最大回撤 (%)'], color='lightcoral'),
                    use_container_width=True
                )
                
                best_fast = results_df.iloc[0]['快线 (天)']
                best_slow = results_df.iloc[0]['慢线 (天)']
                best_sharpe = results_df.iloc[0]['夏普比率']
                
                st.info(f"💡 **结论建议：** 综合收益与抗风险能力，当前标的在扣除手续费后的历史最优搭配为 **{best_fast}日线 / {best_slow}日线** (夏普比率 {best_sharpe})。")
        else:
            st.error("请先确认左侧数据能够正常拉取。")

# ================= 部分 3：Tab3 组合管理页面 =================
with tab3:
    st.markdown("### 🌍 多资产投资组合等权重回测")
    st.write("在下方输入多只股票代码，系统将自动分配等额度资金，并对冲单一标的的极端风险。")
    
    port_symbols_input = st.text_input("输入资产池代码 (用英文逗号隔开，支持A股纯数字或美股代码)：", "AAPL, MSFT, NVDA")
    
    if st.button("▶️ 运行多资产组合回测"):
        # 智能处理输入的代码后缀
        raw_symbols = [s.strip() for s in port_symbols_input.split(",") if s.strip()]
        port_symbols = []
        for s in raw_symbols:
            if s.isdigit() and len(s) == 6:
                port_symbols.append(f"{s}.SS" if s.startswith("6") else f"{s}.SZ")
            else:
                port_symbols.append(s.upper())
                
        with st.spinner(f"正在拉取 {len(port_symbols)} 只标的数据进行矩阵运算..."):
            portfolio_data = fetch_portfolio_data(port_symbols, start_date, end_date)
            
            if portfolio_data:
                port_df = run_portfolio_strategy(portfolio_data, fast_ma_days, slow_ma_days, macd_short, macd_long, macd_signal, trade_cost, atr_period, atr_multi, max_pos)
                
                if not port_df.empty:
                    trading_days = len(port_df)
                    years = trading_days / 252
                    strat_ret = (port_df['策略组合净值'].iloc[-1] - 1)
                    base_ret = (port_df['基准组合净值'].iloc[-1] - 1)
                    
                    daily_mean = port_df['投资组合每日收益'].mean()
                    daily_std = port_df['投资组合每日收益'].std()
                    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
                    
                    st.success(f"组合回测完成！有效拉取了 {len(portfolio_data)} 只标的。")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("组合策略累计收益", f"{strat_ret*100:.2f}%", f"等权基准: {base_ret*100:.2f}%")
                    c2.metric("组合夏普比率", f"{sharpe_ratio:.2f}")
                    c3.metric("组合极限回撤", f"{port_df['Drawdown'].min()*100:.2f}%")
                    c4.metric("总交易换手次数", f"{int(port_df['Trade_Action_Sum'].iloc[0])} 次")
                    
                    st.line_chart(port_df[['基准组合净值', '策略组合净值']])
                    st.area_chart(port_df['Drawdown'] * 100)
                else:
                    st.error("数据计算异常。")
            else:
                st.error("无法获取你输入的任何标的数据，请检查代码拼写。")