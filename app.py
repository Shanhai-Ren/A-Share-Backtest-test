import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures # 引入并行库
import time

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

# === 交易胜率与盈亏比核心统计算法 ===
def calculate_trade_stats(result_df):
    """
    通过仓位(Position)变化，提取每一次完整交易的收益率，
    计算胜率、盈亏比和最大连续亏损次数。
    """
    # 检查是否有仓位列（安全校验）
    if 'Position' not in result_df.columns:
        return 0.0, 0.0, 0, 0, 0.0, 0.0

    # 1. 识别每一次完整交易（从空仓到有仓位算作一次新交易开始）
    is_invested = result_df['Position'] > 0
    trade_starts = (result_df['Position'] > 0) & (result_df['Position'].shift(1) == 0)
    trade_id = trade_starts.cumsum() # 给每次交易打上独立ID
    
    active_trades = result_df[is_invested]
    
    if active_trades.empty:
        return 0.0, 0.0, 0, 0, 0.0, 0.0
        
    # 2. 按交易ID分组，计算单次完整交易的累计净收益率（含手续费）
    trade_returns = active_trades.groupby(trade_id.loc[active_trades.index])['策略每日收益'].apply(lambda x: (1 + x).prod() - 1)
    
    # 3. 统计胜亏数据
    win_trades = trade_returns[trade_returns > 0]
    lose_trades = trade_returns[trade_returns <= 0] # 没赚钱或亏钱都算亏损
    
    total_rounds = len(trade_returns)
    win_rate = len(win_trades) / total_rounds if total_rounds > 0 else 0
    
    avg_win = win_trades.mean() if not win_trades.empty else 0
    avg_loss = lose_trades.mean() if not lose_trades.empty else 0
    
    # 4. 盈亏比 (平均盈利 / 平均亏损的绝对值)
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else (99.99 if avg_win > 0 else 0)
    
    # 5. 计算最大连续亏损次数（极为关键的心理抗压指标）
    is_loss = (trade_returns <= 0).astype(int)
    max_cons_losses = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).sum().max()
    
    return win_rate, pl_ratio, int(max_cons_losses), total_rounds, avg_win, avg_loss

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

def run_portfolio_strategy(data_dict, fast, slow, m_short, m_long, m_sig, cost, atr_period, atr_multi, max_pos, weight_method="风险平价"):
    """
    详尽版多资产投资组合回测引擎 (已加入 风险平价/波动率倒数 动态权重)
    """
    strategy_returns = []
    benchmark_returns = []
    volatilities = [] # 新增：用于收集每只标的的滚动波动率
    total_trades = 0
    
    for code, df in data_dict.items():
        temp_df = df.copy()
        
        # 1-3. 基础指标、MACD、ATR计算 (与之前保持一致)
        temp_df['每日收益率'] = temp_df['收盘'].pct_change()
        temp_df['Fast_MA'] = temp_df['收盘'].rolling(window=fast).mean()
        temp_df['Slow_MA'] = temp_df['收盘'].rolling(window=slow).mean()
        
        ema_short = temp_df['收盘'].ewm(span=m_short, adjust=False).mean()
        ema_long = temp_df['收盘'].ewm(span=m_long, adjust=False).mean()
        temp_df['DIF'] = ema_short - ema_long
        temp_df['DEA'] = temp_df['DIF'].ewm(span=m_sig, adjust=False).mean()
        temp_df['MACD_Hist'] = 2 * (temp_df['DIF'] - temp_df['DEA']) 
        
        temp_df['Prev_Close'] = temp_df['收盘'].shift(1)
        temp_df['TR'] = np.maximum(temp_df['最高'] - temp_df['最低'],
                        np.maximum(abs(temp_df['最高'] - temp_df['Prev_Close']),
                                   abs(temp_df['最低'] - temp_df['Prev_Close'])))
        temp_df['ATR'] = temp_df['TR'].rolling(window=int(atr_period)).mean()
        
        # === 计算该资产的20日滚动波动率 (年化标准差) ===
        # 最小周期设为5，防止初期全为NaN
        temp_df['Volatility'] = temp_df['每日收益率'].rolling(window=20, min_periods=5).std() 
        
        # 4. 状态机：仓位管理与动态止损 (与之前完全一致)
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
            buy_signal = (fast_ma[i-1] > slow_ma[i-1]) and (macd_h[i-1] > 0)
            sell_signal = (fast_ma[i-1] < slow_ma[i-1])
            current_pos = positions[i-1]
            
            if current_pos == 0:
                if buy_signal:
                    positions[i] = max_pos
                    entry_price = close_arr[i]
                    stop_loss = entry_price - atr_multi * atr_arr[i-1]
            elif current_pos > 0:
                if low_arr[i] < stop_loss:
                    positions[i] = 0 
                    entry_price = 0.0
                elif sell_signal:
                    positions[i] = 0 
                    entry_price = 0.0
                else:
                    new_stop = close_arr[i-1] - atr_multi * atr_arr[i-1]
                    if new_stop > stop_loss: stop_loss = new_stop
                    if close_arr[i] > entry_price + atr_arr[i-1] and current_pos < 1.0:
                        positions[i] = 1.0 
                        entry_price = close_arr[i]
                    else:
                        positions[i] = current_pos
                        
        temp_df['Position'] = positions
        temp_df['Trade_Action'] = temp_df['Position'].diff().abs().fillna(0)
        temp_df['策略每日收益'] = (temp_df['Position'].shift(1) * temp_df['每日收益率']) - (temp_df['Trade_Action'] * cost)
        
        # 收集数据进行聚合
        strategy_returns.append(temp_df['策略每日收益'].rename(code))
        benchmark_returns.append(temp_df['每日收益率'].rename(code))
        volatilities.append(temp_df['Volatility'].rename(code)) # 收集波动率
        total_trades += temp_df['Trade_Action'].sum()
        
    if not strategy_returns: 
        return pd.DataFrame()
    
    # ================= 6. 组合层面的汇总与动态权重计算 =================
    returns_df = pd.concat(strategy_returns, axis=1)
    bench_returns_df = pd.concat(benchmark_returns, axis=1)
    
    if weight_method == "风险平价":
        vol_df = pd.concat(volatilities, axis=1)
        # 防止除以0引发报错，将极小波动设为底层阈值
        vol_df = vol_df.replace(0, 1e-8) 
        
        # 核心：取波动的倒数，并且必须 shift(1)！用昨天的波动决定今天的权重
        inv_vol_df = (1.0 / vol_df).shift(1)
        
        # 回测前几日没有波动率数据，默认按等权重(1.0)填充
        inv_vol_df = inv_vol_df.fillna(1.0) 
        
        # 归一化：各项倒数 / 总倒数之和 = 每日动态权重百分比
        weights_df = inv_vol_df.div(inv_vol_df.sum(axis=1), axis=0)
    else:
        # 传统等权重：每天每只票都是 1 / 标的数量
        weights_df = pd.DataFrame(1.0 / len(data_dict), index=returns_df.index, columns=returns_df.columns)

    port_df = pd.DataFrame()
    # 策略组合收益 = sum(各标的收益 * 各标的动态权重)
    port_df['投资组合每日收益'] = (returns_df * weights_df).sum(axis=1)
    # 无论策略怎么配，基准始终保持等权重，方便做 Alpha 对比
    port_df['基准每日收益 (等权)'] = bench_returns_df.mean(axis=1) 
    
    port_df = port_df.dropna()
    if port_df.empty: return port_df
        
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
            
            # === 新增：交易灵魂统计看板 ===
            win_rate, pl_ratio, max_cons_losses, total_rounds, avg_win, avg_loss = calculate_trade_stats(result_df)
            
            st.markdown("### 🎯 核心交易统计 (Trade Statistics)")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("策略胜率 (Win Rate)", f"{win_rate*100:.1f}%", "趋势跟踪通常胜率<50%")
            t2.metric("盈亏比 (P/L Ratio)", f"{pl_ratio:.2f}", f"均赚 {avg_win*100:.1f}% / 均亏 {avg_loss*100:.1f}%")
            t3.metric("最大连亏次数", f"{max_cons_losses} 次", delta="实盘心理抗压阀值", delta_color="inverse")
            t4.metric("完整交易轮次", f"{total_rounds} 轮", f"换手操作: {int(total_trades)} 次")
            
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

# 定义一个用于并行的包装函数 (必须放在全局，方便子进程序列化)
def worker_backtest(args):
    """
    单次参数组合的并行任务
    args 包含: data, f, s, m_short, m_long, m_sig, cost, atr_p, atr_m, m_pos
    """
    data, f, s, m_short, m_long, m_sig, cost, atr_p, atr_m, m_pos = args
    # 调用你之前写好的详尽版单资产引擎 run_strategy
    res_df = run_strategy(data, f, s, m_short, m_long, m_sig, cost, atr_p, atr_m, m_pos)
    
    if res_df.empty:
        return None
    
    # 提取核心指标返回，减少子进程到主进程的数据传输量
    strat_ret = (res_df['策略净值'].iloc[-1] - 1) * 100
    max_dd = res_df['Drawdown'].min() * 100
    daily_mean = res_df['策略每日收益'].mean()
    daily_std = res_df['策略每日收益'].std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
    trades = res_df['Trade_Action'].sum()
    
    return {
        "快线 (天)": f,
        "慢线 (天)": s,
        "净收益率 (%)": round(strat_ret, 2),
        "最大回撤 (%)": round(max_dd, 2),
        "夏普比率": round(sharpe, 2),
        "交易次数": int(trades)
    }

with tab2:
    st.markdown("### 🔍 参数矩阵全核心寻优 (多进程加速)")
    st.info("系统将启动多进程并行计算，利用 CPU 所有核心同时回测。")

    # 1. 寻优范围配置
    c1, c2 = st.columns(2)
    with c1:
        fast_range = st.slider("快线搜索范围", 3, 30, (5, 15))
        slow_range = st.slider("慢线搜索范围", 20, 120, (20, 60))
    with c2:
        step = st.number_input("搜索步长 (天)", min_value=1, value=2)

    if st.button("🚀 启动全速寻优"):
        if not data.empty:
            # 构建待测试的参数列表
            param_list = []
            for f in range(fast_range[0], fast_range[1] + 1, step):
                for s in range(slow_range[0], slow_range[1] + 1, step):
                    if f >= s: continue
                    # 打包所有参数
                    param_list.append((data, f, s, macd_short, macd_long, macd_signal, 
                                      trade_cost, atr_period, atr_multi, max_pos))
            
            total_tasks = len(param_list)
            st.write(f"📊 总计待计算组合数: **{total_tasks}**")
            
            # 2. 初始化进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            final_results = []
            
            # 3. 使用多进程池
            # max_workers 留一个核心给系统，其他全开
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # 提交所有任务
                futures = [executor.submit(worker_backtest, p) for p in param_list]
                
                # 实时获取结果更新进度
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    res = future.result()
                    if res:
                        final_results.append(res)
                    
                    # 更新进度条
                    pct = int((i + 1) / total_tasks * 100)
                    progress_bar.progress(pct)
                    status_text.text(f"已完成: {i+1}/{total_tasks} 组合 | 耗时: {time.time()-start_time:.1f}s")

            # 4. 展示结果
            if final_results:
                results_df = pd.DataFrame(final_results).sort_values(by="夏普比率", ascending=False)
                results_df.reset_index(drop=True, inplace=True)
                
                st.success(f"⚡ 寻优完成！总耗时: {time.time()-start_time:.1f} 秒")
                st.dataframe(
                    results_df.style.highlight_max(subset=['夏普比率', '净收益率 (%)'], color='lightgreen')
                                    .highlight_min(subset=['最大回撤 (%)'], color='lightcoral'),
                    use_container_width=True
                )
                
                # 给出最优建议
                best = results_df.iloc[0]
                st.balloons()
                st.info(f"💡 **最优配置：** 快线 {int(best['快线 (天)'])} / 慢线 {int(best['慢线 (天)'])} | 夏普: {best['夏普比率']}")
        else:
            st.error("请先确认左侧数据能够正常拉取。")

# ================= 部分 3：Tab3 组合管理页面 =================
with tab3:
    st.markdown("### 🌍 多资产投资组合回测 (支持风险平价)")
    st.write("在下方输入多只股票代码，系统将自动分配资金，并对冲单一标的的极端风险。")
    
    # === 新增：资金分配权重方式选择 ===
    weight_choice = st.radio("资金权重分配模型", ["风险平价 (推荐: 根据波动率倒数动态调仓)", "等权重 (传统: 各分配 1/N 资金)"], horizontal=True)
    weight_method_arg = "风险平价" if "风险平价" in weight_choice else "等权重"
    
    port_symbols_input = st.text_input("输入资产池代码 (用英文逗号隔开)：", "AAPL, MSFT, NVDA, TSLA")
    
    if st.button("▶️ 运行多资产组合回测"):
        raw_symbols = [s.strip() for s in port_symbols_input.split(",") if s.strip()]
        port_symbols = []
        for s in raw_symbols:
            if s.isdigit() and len(s) == 6:
                port_symbols.append(f"{s}.SS" if s.startswith("6") else f"{s}.SZ")
            else:
                port_symbols.append(s.upper())
                
        with st.spinner(f"正在拉取 {len(port_symbols)} 只标的数据并执行 {weight_method_arg} 矩阵运算..."):
            portfolio_data = fetch_portfolio_data(port_symbols, start_date, end_date)
            
            if portfolio_data:
                # === 注意这里：把 weight_method_arg 传进去 ===
                port_df = run_portfolio_strategy(portfolio_data, fast_ma_days, slow_ma_days, macd_short, macd_long, macd_signal, trade_cost, atr_period, atr_multi, max_pos, weight_method_arg)
                
                if not port_df.empty:
                    strat_ret = (port_df['策略组合净值'].iloc[-1] - 1)
                    base_ret = (port_df['基准组合净值'].iloc[-1] - 1)
                    
                    daily_mean = port_df['投资组合每日收益'].mean()
                    daily_std = port_df['投资组合每日收益'].std()
                    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
                    
                    st.success(f"组合回测完成！有效拉取了 {len(portfolio_data)} 只标的。")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"组合累计收益 ({weight_method_arg})", f"{strat_ret*100:.2f}%", f"等权基准: {base_ret*100:.2f}%")
                    c2.metric("组合夏普比率", f"{sharpe_ratio:.2f}")
                    c3.metric("组合极限回撤", f"{port_df['Drawdown'].min()*100:.2f}%")
                    c4.metric("总交易换手次数", f"{int(port_df['Trade_Action_Sum'].iloc[0])} 次")
                    
                    st.line_chart(port_df[['基准组合净值', '策略组合净值']])
                    st.area_chart(port_df['Drawdown'] * 100)
                else:
                    st.error("数据计算异常。")
            else:
                st.error("无法获取你输入的任何标的数据，请检查代码拼写。")