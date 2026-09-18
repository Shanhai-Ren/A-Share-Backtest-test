# -*- coding: utf-8 -*-
"""
================================================================================
 全球市场量化回测与绩效研报系统  v2.0
--------------------------------------------------------------------------------
 相比 v1 的升级：
   1. 数据层     —— 多源取数（yfinance / akshare）、复权、交易日历对齐、数据质量体检
   2. 信号层     —— 4 类策略（趋势跟踪 / 均值回归 / 时序动量 / RSI 反转）+ 信号融合
   3. 执行层     —— 佣金 + 滑点（ATR 比例）+ 印花税；波动率目标仓位；时间止损；分批建仓
   4. 绩效层     —— 30+ 指标、逐笔交易台账（含退出原因）、滚动指标、月度热力图
   5. 统计检验层 —— PSR / DSR（多重检验校正）、Block Bootstrap 置信区间、
                    蒙特卡洛交易重排（回撤分布 / 破产概率）、White Reality Check
   6. 组合层     —— 等权 / 逆波动 / ERC 风险平价（协方差迭代求解）/ 最小方差 /
                    最大分散化 / 目标波动率；Ledoit-Wolf 收缩；再平衡与漂移阈值；
                    风险贡献分解与相关性矩阵
   7. 寻优层     —— 并行网格 + 参数高原热力图 + 邻域稳健性评分 + Walk-Forward 样本外
   8. AI 研报层  —— 把指标与检验结论打包交给大模型写研报（无 Key 时用本地规则模板）

 设计原则：任何一个"锦上添花"的依赖（scipy / akshare / 大模型 API）缺失时，
           系统都必须降级运行而不是报错退出。
================================================================================
"""

import io
import json
import time
import math
import warnings
import concurrent.futures
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")                     # 云端无显示设备，必须在 import pyplot 前设定
import matplotlib.font_manager
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- 可选依赖：缺了就降级，不让整个 app 崩掉 ----------------------------------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

try:
    import akshare as ak
    HAS_AK = True
except Exception:
    HAS_AK = False

try:
    from scipy.optimize import minimize
    from scipy import stats as sps
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


# ==============================================================================
# 0. 全局配置
# ==============================================================================
TRADING_DAYS = 252

st.set_page_config(page_title="全球市场量化研报 v2", layout="wide",
                   initial_sidebar_state="expanded")


def _setup_matplotlib_font():
    """
    Streamlit Cloud 的容器里通常没有中文字体，matplotlib 画中文会变成方框。
    这里做两件事：能找到中文字体就用；找不到就退回英文标签（图内文字一律用英文）。
    """
    for name in ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                 "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


CJK_FONT_OK = _setup_matplotlib_font()


def init_state():
    defaults = {
        "history_reports": {},       # {报告名: {"result": df, "cfg": dict, "stats": dict}}
        "current_report_key": None,
        "opt_results": None,         # 参数寻优结果
        "wfa_results": None,         # walk-forward 结果
        "portfolio_result": None,    # 组合回测结果
        "llm_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ==============================================================================
# 1. 回测配置对象
# ==============================================================================
@dataclass
class BTConfig:
    """一次回测的全部可调参数。做成对象是为了让寻优、WFA、组合三处共用同一套语义。"""
    # --- 信号 ---
    strategy: str = "趋势跟踪 (MA+MACD)"
    fast_ma: int = 5
    slow_ma: int = 20
    macd_short: int = 12
    macd_long: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_buy: int = 30
    rsi_sell: int = 70
    boll_period: int = 20
    boll_std: float = 2.0
    mom_lookback: int = 60
    adx_period: int = 14
    adx_min: float = 20.0
    blend_mode: str = "多数投票"      # 信号融合方式
    blend_members: tuple = ()

    # --- 风控与仓位 ---
    atr_period: int = 14
    atr_multi: float = 2.0
    init_pos: float = 0.5             # 首次建仓比例
    add_on_atr: float = 1.0           # 浮盈超过 N 倍 ATR 后加满
    time_stop: int = 0                # 持仓超过 N 个交易日强制离场，0 = 关闭
    vol_target: float = 0.0           # 年化目标波动率，0 = 关闭波动率目标仓位
    vol_window: int = 20
    max_leverage: float = 1.0

    # --- 交易成本 ---
    commission_bps: float = 1.0       # 单边佣金（bp，万分之一）
    slippage_mode: str = "ATR 比例"
    slippage_bps: float = 2.0         # 固定滑点（bp）
    slippage_atr: float = 0.05        # 按 ATR 的比例滑点
    stamp_duty_bps: float = 5.0       # 印花税，仅卖出方向，A 股默认 5bp
    is_a_share: bool = True

    # --- 其他 ---
    rf_annual: float = 0.0            # 无风险利率（年化）

    def key(self) -> str:
        return f"{self.strategy}|MA{self.fast_ma}-{self.slow_ma}|ATR{self.atr_period}x{self.atr_multi}"


# ==============================================================================
# 2. 数据层
# ==============================================================================
def normalize_symbol(raw: str, market: str) -> str:
    """把用户输入统一成 yfinance 能认的代码。"""
    s = raw.strip().upper()
    if market.startswith("A股"):
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) == 6:
            # 6/9 开头为沪市，其余（0/3）为深市；北交所 8/4 开头 yfinance 基本没有数据
            return f"{digits}.SS" if digits[0] in ("6", "9") else f"{digits}.SZ"
        return s
    return s


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ohlcv(code: str, start, end, source: str = "自动") -> pd.DataFrame:
    """
    取 OHLCV。策略：
      - "自动"：A 股优先用 akshare（前复权质量更好），失败再退 yfinance；美股直接 yfinance
      - 统一列名为中文，index 去时区，按日期升序
    """
    cols = ["开盘", "最高", "最低", "收盘", "成交量"]

    def _from_yf(c):
        if not HAS_YF:
            return pd.DataFrame()
        df = yf.Ticker(c).history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        df = df.rename(columns={"Open": "开盘", "High": "最高", "Low": "最低",
                                "Close": "收盘", "Volume": "成交量"})
        return df[cols]

    def _from_ak(c):
        if not HAS_AK:
            return pd.DataFrame()
        digits = "".join(ch for ch in c if ch.isdigit())
        if len(digits) != 6:
            return pd.DataFrame()
        df = ak.stock_zh_a_hist(symbol=digits, period="daily",
                                start_date=pd.to_datetime(start).strftime("%Y%m%d"),
                                end_date=pd.to_datetime(end).strftime("%Y%m%d"),
                                adjust="qfq")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"日期": "date", "开盘": "开盘", "最高": "最高",
                                "最低": "最低", "收盘": "收盘", "成交量": "成交量"})
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[cols]

    is_a = code.endswith(".SS") or code.endswith(".SZ")
    order = []
    if source == "yfinance":
        order = [_from_yf]
    elif source == "akshare":
        order = [_from_ak]
    else:
        order = [_from_ak, _from_yf] if is_a else [_from_yf, _from_ak]

    for fn in order:
        try:
            df = fn(code)
            if df is not None and not df.empty:
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]
                return df.astype(float)
        except Exception:
            continue
    return pd.DataFrame()


def data_quality_report(df: pd.DataFrame) -> dict:
    """
    数据体检。回测最常见的坑不是模型写错，而是数据有洞自己没发现。
    """
    if df.empty:
        return {"状态": "无数据"}
    ret = df["收盘"].pct_change()
    suspended = int((df["成交量"] <= 0).sum())
    gaps = int((ret.abs() > 0.11).sum())          # A 股涨跌停 10%，超过 11% 多半是数据问题
    idx = pd.DatetimeIndex(df.index)
    span_days = (idx[-1] - idx[0]).days
    expected = span_days * TRADING_DAYS / 365.25 if span_days > 0 else len(df)
    return {
        "状态": "正常",
        "样本区间": f"{idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}",
        "交易日数": len(df),
        "预期交易日": int(expected),
        "覆盖率": f"{len(df) / expected * 100:.1f}%" if expected else "-",
        "缺失值": int(df.isna().sum().sum()),
        "疑似停牌(零成交)": suspended,
        "异常跳空(>11%)": gaps,
        "年化波动": f"{ret.std() * np.sqrt(TRADING_DAYS) * 100:.1f}%",
    }


# ==============================================================================
# 3. 指标层
# ==============================================================================
def add_indicators(df: pd.DataFrame, cfg: BTConfig) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["收盘"], d["最高"], d["最低"]

    d["每日收益率"] = c.pct_change()

    # 均线
    d["Fast_MA"] = c.rolling(cfg.fast_ma).mean()
    d["Slow_MA"] = c.rolling(cfg.slow_ma).mean()

    # MACD
    ema_s = c.ewm(span=cfg.macd_short, adjust=False).mean()
    ema_l = c.ewm(span=cfg.macd_long, adjust=False).mean()
    d["DIF"] = ema_s - ema_l
    d["DEA"] = d["DIF"].ewm(span=cfg.macd_signal, adjust=False).mean()
    d["MACD_Hist"] = 2 * (d["DIF"] - d["DEA"])

    # ATR（Wilder 平滑，比简单均值更标准）
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["TR"] = tr
    d["ATR"] = tr.ewm(alpha=1 / max(cfg.atr_period, 1), adjust=False).mean()

    # RSI（Wilder）
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / max(cfg.rsi_period, 1), adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / max(cfg.rsi_period, 1), adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)
    d["RSI"] = d["RSI"].fillna(50)

    # 布林带
    mid = c.rolling(cfg.boll_period).mean()
    sd = c.rolling(cfg.boll_period).std()
    d["BOLL_MID"] = mid
    d["BOLL_UP"] = mid + cfg.boll_std * sd
    d["BOLL_DN"] = mid - cfg.boll_std * sd
    d["BOLL_PCTB"] = (c - d["BOLL_DN"]) / (d["BOLL_UP"] - d["BOLL_DN"]).replace(0, np.nan)

    # 时序动量
    d["MOM"] = c.pct_change(cfg.mom_lookback)

    # ADX（判断"有没有趋势"，用来过滤震荡市里的假信号）
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_n = tr.ewm(alpha=1 / max(cfg.adx_period, 1), adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(
        alpha=1 / max(cfg.adx_period, 1), adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(
        alpha=1 / max(cfg.adx_period, 1), adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1 / max(cfg.adx_period, 1), adjust=False).mean().fillna(0)

    # 滚动波动率（给波动率目标仓位用）
    d["RealizedVol"] = d["每日收益率"].rolling(cfg.vol_window, min_periods=5).std() * np.sqrt(TRADING_DAYS)

    return d


# ==============================================================================
# 4. 信号层
# ==============================================================================
def _sig_trend(d: pd.DataFrame, cfg: BTConfig):
    """趋势跟踪：快线上穿慢线 + MACD 柱为正才进；快线下穿慢线出。"""
    entry = (d["Fast_MA"] > d["Slow_MA"]) & (d["MACD_Hist"] > 0)
    exit_ = (d["Fast_MA"] < d["Slow_MA"])
    return entry, exit_


def _sig_meanrev(d: pd.DataFrame, cfg: BTConfig):
    """均值回归：跌破布林下轨进，回到中轨出。"""
    entry = d["收盘"] < d["BOLL_DN"]
    exit_ = d["收盘"] > d["BOLL_MID"]
    return entry, exit_


def _sig_momentum(d: pd.DataFrame, cfg: BTConfig):
    """时序动量 + ADX 过滤：过去 N 日涨且确实处在趋势里才进。"""
    entry = (d["MOM"] > 0) & (d["ADX"] > cfg.adx_min)
    exit_ = (d["MOM"] < 0)
    return entry, exit_


def _sig_rsi(d: pd.DataFrame, cfg: BTConfig):
    """RSI 反转：超卖进、超买出。"""
    entry = d["RSI"] < cfg.rsi_buy
    exit_ = d["RSI"] > cfg.rsi_sell
    return entry, exit_


SIGNAL_LIB = {
    "趋势跟踪 (MA+MACD)": _sig_trend,
    "均值回归 (布林带)": _sig_meanrev,
    "时序动量 (MOM+ADX)": _sig_momentum,
    "RSI 反转": _sig_rsi,
}


def build_signals(d: pd.DataFrame, cfg: BTConfig):
    """
    产出两个布尔序列：entry_raw / exit_raw。
    注意：这里只表达"当日收盘后看到的状态"，真正下单在执行层统一延后一根 K 线，
    避免用当天的收盘价做当天的决策（前视偏差）。
    """
    if cfg.strategy != "信号融合":
        fn = SIGNAL_LIB.get(cfg.strategy, _sig_trend)
        return fn(d, cfg)

    members = list(cfg.blend_members) or list(SIGNAL_LIB.keys())
    entries, exits = [], []
    for m in members:
        e, x = SIGNAL_LIB[m](d, cfg)
        entries.append(e.astype(int))
        exits.append(x.astype(int))
    e_sum = pd.concat(entries, axis=1).sum(axis=1)
    x_sum = pd.concat(exits, axis=1).sum(axis=1)
    n = len(members)

    if cfg.blend_mode == "全票通过":
        return e_sum == n, x_sum >= 1
    if cfg.blend_mode == "任一触发":
        return e_sum >= 1, x_sum == n
    return e_sum > n / 2, x_sum > n / 2       # 多数投票

# ==============================================================================
# 5. 执行层：状态机回测
# ==============================================================================
def run_backtest(df: pd.DataFrame, cfg: BTConfig) -> pd.DataFrame:
    """
    单标的回测引擎。

    与 v1 的关键差别：
      1. 信号一律用 shift(1) 后的值 —— 今天的仓位只能由昨天收盘可见的信息决定；
      2. 成本拆成 佣金 + 滑点 + 印花税（卖出方向），滑点可按 ATR 比例动态计算；
      3. 仓位可叠加波动率目标（vol targeting），高波动时自动缩量；
      4. 记录每次离场的原因，供交易台账做归因。
    """
    d = add_indicators(df, cfg)
    entry_raw, exit_raw = build_signals(d, cfg)

    # 全部延后一根 K 线，杜绝前视偏差
    d["Entry"] = entry_raw.shift(1).fillna(False).astype(bool)
    d["Exit"] = exit_raw.shift(1).fillna(False).astype(bool)

    n = len(d)
    close = d["收盘"].values
    low = d["最低"].values
    atr = d["ATR"].fillna(0).values
    entry_arr = d["Entry"].values
    exit_arr = d["Exit"].values
    rvol = d["RealizedVol"].values

    # 波动率目标：目标波动 / 已实现波动，上限为 max_leverage
    if cfg.vol_target and cfg.vol_target > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rvol > 1e-8, cfg.vol_target / rvol, 1.0)
        scale = np.clip(np.nan_to_num(scale, nan=1.0), 0.0, cfg.max_leverage)
    else:
        scale = np.ones(n)

    pos = np.zeros(n)
    stop_line = np.zeros(n)
    exit_reason = np.array([""] * n, dtype=object)

    entry_price = 0.0
    stop_loss = 0.0
    hold_days = 0

    for i in range(1, n):
        prev = pos[i - 1]
        cur_scale = scale[i] if scale[i] > 0 else 1.0

        if prev == 0:
            if entry_arr[i]:
                pos[i] = min(cfg.init_pos * cur_scale, cfg.max_leverage)
                entry_price = close[i]
                stop_loss = entry_price - cfg.atr_multi * atr[i - 1]
                hold_days = 1
            else:
                pos[i] = 0.0
        else:
            hold_days += 1
            # 1) 止损：当日最低价击穿止损线
            if low[i] < stop_loss:
                pos[i] = 0.0
                exit_reason[i] = "移动止损" if stop_loss > entry_price else "初始止损"
                entry_price, stop_loss, hold_days = 0.0, 0.0, 0
            # 2) 信号离场
            elif exit_arr[i]:
                pos[i] = 0.0
                exit_reason[i] = "信号离场"
                entry_price, stop_loss, hold_days = 0.0, 0.0, 0
            # 3) 时间止损
            elif cfg.time_stop and hold_days > cfg.time_stop:
                pos[i] = 0.0
                exit_reason[i] = "时间止损"
                entry_price, stop_loss, hold_days = 0.0, 0.0, 0
            else:
                # 移动止损只上移不下移
                new_stop = close[i - 1] - cfg.atr_multi * atr[i - 1]
                if new_stop > stop_loss:
                    stop_loss = new_stop
                # 浮盈超过 N 倍 ATR 后加满
                target = min(1.0 * cur_scale, cfg.max_leverage)
                if close[i] > entry_price + cfg.add_on_atr * atr[i - 1] and prev < target:
                    pos[i] = target
                    entry_price = close[i]
                else:
                    pos[i] = prev
        stop_line[i] = stop_loss

    d["Position"] = pos
    d["StopLine"] = np.where(d["Position"] > 0, stop_line, np.nan)
    d["ExitReason"] = exit_reason
    d["PosChange"] = d["Position"].diff().fillna(d["Position"])
    d["Turnover"] = d["PosChange"].abs()

    # ---- 成本模型 -------------------------------------------------------------
    commission = cfg.commission_bps / 10000.0
    if cfg.slippage_mode == "ATR 比例":
        slip = (cfg.slippage_atr * d["ATR"] / d["收盘"]).fillna(0).clip(0, 0.05)
    else:
        slip = pd.Series(cfg.slippage_bps / 10000.0, index=d.index)
    stamp = (cfg.stamp_duty_bps / 10000.0) if cfg.is_a_share else 0.0

    sell_turnover = (-d["PosChange"]).clip(lower=0)      # 只有减仓才交印花税
    d["成本"] = d["Turnover"] * (commission + slip) + sell_turnover * stamp

    d["策略每日毛收益"] = d["Position"].shift(1).fillna(0) * d["每日收益率"]
    d["策略每日收益"] = d["策略每日毛收益"] - d["成本"]

    d = d.dropna(subset=["每日收益率"])
    if d.empty:
        return d

    d["基准净值"] = (1 + d["每日收益率"]).cumprod()
    d["策略净值"] = (1 + d["策略每日收益"]).cumprod()
    d["HWM"] = d["策略净值"].cummax()
    d["Drawdown"] = d["策略净值"] / d["HWM"] - 1
    d["基准HWM"] = d["基准净值"].cummax()
    d["基准Drawdown"] = d["基准净值"] / d["基准HWM"] - 1
    return d


# ==============================================================================
# 6. 绩效层
# ==============================================================================
def _max_dd_duration(equity: pd.Series) -> int:
    """最长回撤持续天数：净值从上一个高点到重新创新高，中间隔了多久。"""
    hwm = equity.cummax()
    under = equity < hwm
    if not under.any():
        return 0
    longest = cur = 0
    for flag in under.values:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return int(longest)


def performance_stats(d: pd.DataFrame, cfg: BTConfig) -> dict:
    """一次算完研报需要的全部指标。返回纯 float，方便后面喂给大模型。"""
    if d.empty:
        return {}
    r = d["策略每日收益"]
    b = d["每日收益率"]
    eq = d["策略净值"]
    years = len(d) / TRADING_DAYS
    rf_daily = cfg.rf_annual / TRADING_DAYS

    total = eq.iloc[-1] - 1
    bench_total = d["基准净值"].iloc[-1] - 1
    cagr = (eq.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (d["基准净值"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0

    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < rf_daily]
    dvol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    excess = r - rf_daily
    sharpe = excess.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
    sortino = excess.mean() / downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 and downside.std() > 0 else 0.0
    mdd = d["Drawdown"].min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    # Omega：以无风险为阈值，上行面积 / 下行面积
    up = (r - rf_daily).clip(lower=0).sum()
    dn = (rf_daily - r).clip(lower=0).sum()
    omega = up / dn if dn > 0 else np.nan

    # 相对基准：beta / alpha / 信息比率 / 跟踪误差
    if b.std() > 0:
        beta = np.cov(r.fillna(0), b.fillna(0))[0, 1] / b.var()
    else:
        beta = np.nan
    alpha = (r.mean() - rf_daily - beta * (b.mean() - rf_daily)) * TRADING_DAYS if beta == beta else np.nan
    active = r - b
    te = active.std() * np.sqrt(TRADING_DAYS)
    ir = active.mean() / active.std() * np.sqrt(TRADING_DAYS) if active.std() > 0 else np.nan

    # 尾部
    var95 = np.percentile(r.dropna(), 5)
    cvar95 = r[r <= var95].mean() if (r <= var95).any() else np.nan

    exposure = (d["Position"] > 0).mean()
    turnover_yr = d["Turnover"].sum() / years if years > 0 else np.nan
    cost_total = d["成本"].sum()

    return {
        "累计收益": float(total),
        "基准累计收益": float(bench_total),
        "超额收益": float(total - bench_total),
        "年化收益": float(cagr),
        "基准年化": float(bench_cagr),
        "年化波动": float(vol),
        "下行波动": float(dvol) if dvol == dvol else np.nan,
        "夏普比率": float(sharpe),
        "索提诺比率": float(sortino),
        "卡玛比率": float(calmar) if calmar == calmar else np.nan,
        "Omega": float(omega) if omega == omega else np.nan,
        "最大回撤": float(mdd),
        "基准最大回撤": float(d["基准Drawdown"].min()),
        "最长回撤天数": _max_dd_duration(eq),
        "Beta": float(beta) if beta == beta else np.nan,
        "年化Alpha": float(alpha) if alpha == alpha else np.nan,
        "跟踪误差": float(te),
        "信息比率": float(ir) if ir == ir else np.nan,
        "日VaR95": float(var95),
        "日CVaR95": float(cvar95) if cvar95 == cvar95 else np.nan,
        "偏度": float(r.skew()),
        "峰度": float(r.kurtosis()),
        "在场比例": float(exposure),
        "年换手率": float(turnover_yr) if turnover_yr == turnover_yr else np.nan,
        "累计成本占比": float(cost_total),
        "交易日数": int(len(d)),
        "回测年数": float(years),
    }


def trade_ledger(d: pd.DataFrame) -> pd.DataFrame:
    """
    把仓位序列还原成一笔一笔的交易，带入场/出场日期、持有天数、收益、离场原因。
    这是把"回测结果"变成"可复盘的交易记录"的关键一步。
    """
    if d.empty or "Position" not in d:
        return pd.DataFrame()
    pos = d["Position"].values
    idx = d.index
    rows = []
    in_pos = False
    start_i = 0
    for i in range(len(d)):
        if not in_pos and pos[i] > 0:
            in_pos, start_i = True, i
        elif in_pos and pos[i] == 0:
            seg = d.iloc[start_i:i]
            ret = (1 + seg["策略每日收益"]).prod() - 1
            rows.append({
                "入场日": idx[start_i].date(),
                "出场日": idx[i].date(),
                "持有天数": i - start_i,
                "入场价": round(float(d["收盘"].iloc[start_i]), 4),
                "出场价": round(float(d["收盘"].iloc[i]), 4),
                "最大仓位": round(float(seg["Position"].max()), 2),
                "区间收益": ret,
                "区间成本": float(seg["成本"].sum()),
                "离场原因": d["ExitReason"].iloc[i] or "信号离场",
            })
            in_pos = False
    if in_pos:                                   # 回测结束时还在场内
        seg = d.iloc[start_i:]
        rows.append({
            "入场日": idx[start_i].date(),
            "出场日": idx[-1].date(),
            "持有天数": len(d) - start_i,
            "入场价": round(float(d["收盘"].iloc[start_i]), 4),
            "出场价": round(float(d["收盘"].iloc[-1]), 4),
            "最大仓位": round(float(seg["Position"].max()), 2),
            "区间收益": (1 + seg["策略每日收益"]).prod() - 1,
            "区间成本": float(seg["成本"].sum()),
            "离场原因": "尚未平仓",
        })
    return pd.DataFrame(rows)


def trade_stats(ledger: pd.DataFrame) -> dict:
    """基于交易台账（而不是日频收益）统计胜率、盈亏比、连亏、期望值。"""
    if ledger.empty:
        return {}
    rets = ledger["区间收益"]
    wins, losses = rets[rets > 0], rets[rets <= 0]
    wr = len(wins) / len(rets)
    avg_w = wins.mean() if len(wins) else 0.0
    avg_l = losses.mean() if len(losses) else 0.0
    pl = abs(avg_w / avg_l) if avg_l != 0 else np.nan
    is_loss = (rets <= 0).astype(int)
    max_cons = int(is_loss.groupby((is_loss != is_loss.shift()).cumsum()).sum().max()) if len(rets) else 0
    return {
        "交易笔数": int(len(rets)),
        "胜率": float(wr),
        "平均盈利": float(avg_w),
        "平均亏损": float(avg_l),
        "盈亏比": float(pl) if pl == pl else np.nan,
        "单笔期望": float(rets.mean()),
        "最大连亏次数": max_cons,
        "平均持有天数": float(ledger["持有天数"].mean()),
        "最佳单笔": float(rets.max()),
        "最差单笔": float(rets.min()),
    }


def monthly_return_table(d: pd.DataFrame) -> pd.DataFrame:
    tmp = d.copy()
    tmp["Y"] = tmp.index.year
    tmp["M"] = tmp.index.month
    m = tmp.groupby(["Y", "M"])["策略每日收益"].apply(lambda x: (1 + x).prod() - 1).unstack()
    m.columns = [f"{c}月" for c in m.columns]
    m["全年"] = (1 + m.fillna(0)).prod(axis=1) - 1
    return m


def rolling_metrics(d: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    r = d["策略每日收益"]
    out = pd.DataFrame(index=d.index)
    out["滚动夏普"] = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(TRADING_DAYS)
    out["滚动年化波动"] = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    out["滚动超额"] = (r - d["每日收益率"]).rolling(window).mean() * TRADING_DAYS
    return out.dropna()


# ==============================================================================
# 7. 统计检验层  —— 这一层回答的是："这个结果是真本事，还是撞大运？"
# ==============================================================================
def probabilistic_sharpe_ratio(sr: float, n: int, skew: float, kurt: float,
                               sr_benchmark: float = 0.0) -> float:
    """
    PSR（Bailey & López de Prado）：在收益分布有偏、有肥尾的前提下，
    观测到的夏普真的高于基准夏普的概率。
    sr / sr_benchmark 均为年化值，内部转成日频。
    """
    if n < 10:
        return np.nan
    sr_d = sr / np.sqrt(TRADING_DAYS)
    sr_b = sr_benchmark / np.sqrt(TRADING_DAYS)
    denom = 1 - skew * sr_d + (kurt + 2) / 4.0 * sr_d ** 2   # kurt 为超额峰度
    if denom <= 0:
        return np.nan
    z = (sr_d - sr_b) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(_norm_cdf(z))


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurt: float,
                          n_trials: int, sr_variance: float) -> float:
    """
    DSR：把"你一共试了多少组参数"这件事算进去。
    试 500 组参数挑出来的最高夏普，本来就该比单次测试高一截；
    DSR 先算出这个"运气门槛"（期望最大夏普），再问你是否显著超过它。
    """
    if n_trials < 2 or sr_variance <= 0 or n < 10:
        return np.nan
    e = 0.5772156649015329                       # Euler-Mascheroni
    z1 = _norm_ppf(1 - 1.0 / n_trials)
    z2 = _norm_ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(sr_variance) * ((1 - e) * z1 + e * z2)     # 期望最大夏普（年化口径）
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)


def _norm_cdf(x):
    if HAS_SCIPY:
        return sps.norm.cdf(x)
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    if HAS_SCIPY:
        return sps.norm.ppf(p)
    # Acklam 有理逼近，精度足够画结论
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def block_bootstrap(returns: np.ndarray, n_boot: int = 500, block: int = 20, seed: int = 42):
    """
    平稳区块自助法。日收益有自相关和波动聚集，直接逐日重抽会把这些结构抹掉，
    所以按区块抽（每块保留块内的时序结构）。
    返回：每次重抽的 (年化收益, 年化夏普, 最大回撤)。
    """
    rng = np.random.default_rng(seed)
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n < block * 2:
        return np.empty((0, 3))
    n_blocks = int(np.ceil(n / block))
    out = np.zeros((n_boot, 3))
    for k in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        eq = np.cumprod(1 + sample)
        years = n / TRADING_DAYS
        cagr = eq[-1] ** (1 / years) - 1 if years > 0 else 0.0
        sd = sample.std()
        sharpe = sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        out[k] = [cagr, sharpe, mdd]
    return out


def monte_carlo_trades(trade_returns: np.ndarray, n_sim: int = 2000,
                       ruin_threshold: float = -0.30, seed: int = 7,
                       mode: str = "有放回重抽"):
    """
    蒙特卡洛交易重排，两种模式：

    - 「随机排列」：只打乱历史这几笔交易的先后顺序。注意乘法可交换，终值是恒定的，
      它唯一能回答的问题是「同样的交易，换个顺序，路径能有多难看」——也就是回撤分布。
    - 「有放回重抽」（默认）：把历史交易当成一个分布，重新抽同样多笔。
      胜率和盈亏比在期望上不变，但终值会变，因此还能回答「亏钱的概率有多大」。

    实盘真正要准备的是这里的 5% 最坏情形，而不是历史上恰好发生的那一条路径。
    """
    r = trade_returns[~np.isnan(trade_returns)]
    if len(r) < 5:
        return {}
    rng = np.random.default_rng(seed)
    m = len(r)
    finals, mdds = np.zeros(n_sim), np.zeros(n_sim)
    for k in range(n_sim):
        if mode == "随机排列":
            sample = rng.permutation(r)
        else:
            sample = rng.choice(r, size=m, replace=True)
        eq = np.cumprod(1 + sample)
        finals[k] = eq[-1] - 1
        mdds[k] = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {
        "模式": mode,
        "终值中位数": float(np.median(finals)),
        "终值5%分位": float(np.percentile(finals, 5)),
        "终值95%分位": float(np.percentile(finals, 95)),
        "亏损概率": float((finals < 0).mean()),
        "回撤中位数": float(np.median(mdds)),
        "回撤95%分位(最坏)": float(np.percentile(mdds, 5)),
        f"回撤超过{abs(ruin_threshold):.0%}的概率": float((mdds < ruin_threshold).mean()),
        "_finals": finals,
        "_mdds": mdds,
    }


def reality_check(strategy_ret: np.ndarray, bench_ret: np.ndarray,
                  n_boot: int = 500, block: int = 20, seed: int = 11) -> float:
    """
    White's Reality Check 的简化版：在"策略相对基准无超额"的原假设下，
    用区块自助重抽算出超额收益均值的分布，看实际观测值落在什么位置。
    返回 p 值，越小越说明超额不是噪声。
    """
    d = (strategy_ret - bench_ret)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < block * 2:
        return np.nan
    observed = d.mean()
    centered = d - observed                      # 强制满足原假设
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boot_means = np.zeros(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        s = np.concatenate([centered[i:i + block] for i in starts])[:n]
        boot_means[k] = s.mean()
    return float((boot_means >= observed).mean())

# ==============================================================================
# 8. 组合层
# ==============================================================================
def ledoit_wolf_shrink(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf 式收缩（常相关目标）。样本协方差在标的多、样本短时噪声极大，
    收缩到一个结构化目标能让最小方差/ERC 这类"吃协方差矩阵"的模型稳很多。
    """
    X = returns.dropna().values
    t, n = X.shape
    if t < 5 or n < 2:
        return np.cov(returns.dropna().values, rowvar=False)
    S = np.cov(X, rowvar=False)
    var = np.diag(S)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    off = corr[~np.eye(n, dtype=bool)]
    r_bar = np.nanmean(off) if off.size else 0.0
    F = r_bar * np.outer(std, std)               # 目标矩阵：所有相关系数都等于平均值
    np.fill_diagonal(F, var)
    # 收缩强度用简化估计：样本越短、标的越多，越靠目标
    delta = float(np.clip(n / max(t, 1) * 0.5, 0.0, 1.0))
    return (1 - delta) * S + delta * F


def _risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.zeros_like(w)
    mrc = cov @ w                                 # 边际风险贡献
    return w * mrc / np.sqrt(port_var)            # 风险贡献


def solve_weights(cov: np.ndarray, method: str) -> np.ndarray:
    """
    按不同风险模型求权重。全部做多、权重和为 1。
    scipy 不可用时自动退回解析解或逆波动近似，保证永远有结果。
    """
    n = cov.shape[0]
    eq = np.ones(n) / n
    if n == 1:
        return np.ones(1)

    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))

    if method == "等权重":
        return eq
    if method == "逆波动率":
        inv = 1.0 / vol
        return inv / inv.sum()

    if not HAS_SCIPY:
        # 没有 scipy 时，ERC 用逆波动近似，最小方差/最大分散化退回逆波动
        inv = 1.0 / vol
        return inv / inv.sum()

    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = eq

    if method == "ERC 风险平价":
        target = np.ones(n) / n

        def obj(w):
            rc = _risk_contribution(w, cov)
            tot = rc.sum()
            if tot <= 0:
                return 1e6
            return float(((rc / tot - target) ** 2).sum())
    elif method == "最小方差":
        def obj(w):
            return float(w @ cov @ w)
    elif method == "最大分散化":
        def obj(w):
            num = float(w @ vol)
            den = float(np.sqrt(w @ cov @ w))
            return -num / den if den > 0 else 1e6
    else:
        return eq

    try:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-10})
        w = np.clip(res.x, 0, None)
        s = w.sum()
        return w / s if s > 0 else eq
    except Exception:
        return eq


def run_portfolio(data_dict: dict, cfg: BTConfig, weight_method: str,
                  lookback: int = 60, rebalance: str = "每月",
                  drift_threshold: float = 0.0,
                  port_vol_target: float = 0.0,
                  dd_cutoff: float = 0.0) -> dict:
    """
    多资产组合回测。

    流程：先对每只标的独立跑一遍择时策略拿到各自的日收益，
          再在组合层按选定的风险模型分配资金，并叠加组合级风控。

    组合级风控两条：
      - 目标波动率：组合近期波动高于目标就整体缩仓；
      - 回撤熔断：组合回撤突破阈值后强制空仓，等净值修复再回来。
    """
    strat_rets, bench_rets, per_asset = {}, {}, {}
    for code, raw in data_dict.items():
        res = run_backtest(raw, cfg)
        if res.empty:
            continue
        strat_rets[code] = res["策略每日收益"]
        bench_rets[code] = res["每日收益率"]
        per_asset[code] = res

    if not strat_rets:
        return {}

    R = pd.DataFrame(strat_rets).dropna(how="all").fillna(0.0)
    B = pd.DataFrame(bench_rets).reindex(R.index).fillna(0.0)
    codes = list(R.columns)
    n = len(codes)

    # ---- 再平衡日 ----
    if rebalance == "每日":
        rb_mask = pd.Series(True, index=R.index)
    elif rebalance == "每周":
        rb_mask = R.index.to_series().dt.isocalendar().week.diff().fillna(1) != 0
    elif rebalance == "每季":
        rb_mask = R.index.to_series().dt.quarter.diff().fillna(1) != 0
    else:                                        # 每月
        rb_mask = R.index.to_series().dt.month.diff().fillna(1) != 0

    W = pd.DataFrame(index=R.index, columns=codes, dtype=float)
    cur_w = np.ones(n) / n
    rc_log = {}

    for i, dt in enumerate(R.index):
        if i >= lookback and bool(rb_mask.iloc[i]):
            window = R.iloc[i - lookback:i]
            cov = ledoit_wolf_shrink(window)
            new_w = solve_weights(cov, weight_method)
            # 漂移阈值：新老权重差得不够多就不动，省下这一次的换手成本
            if not (drift_threshold > 0 and np.abs(new_w - cur_w).max() < drift_threshold):
                cur_w = new_w
                rc_log[dt] = _risk_contribution(cur_w, cov)
        W.iloc[i] = cur_w

    W = W.shift(1).fillna(1.0 / n)               # 权重同样延后一日生效

    gross = (R * W).sum(axis=1)

    # 组合层再平衡成本：每次调仓都要按换手额付佣金和滑点，这一笔不能漏
    rebal_turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    rebal_cost = rebal_turnover * ((cfg.commission_bps + cfg.slippage_bps) / 10000.0)
    gross = gross - rebal_cost

    # ---- 组合级目标波动率 ----
    if port_vol_target and port_vol_target > 0:
        realized = gross.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)
        scale = (port_vol_target / realized).clip(0, cfg.max_leverage).fillna(1.0).shift(1).fillna(1.0)
    else:
        scale = pd.Series(1.0, index=gross.index)
    port_ret = gross * scale

    # ---- 组合级回撤熔断 ----
    if dd_cutoff and dd_cutoff > 0:
        eq, hwm, live = 1.0, 1.0, 1.0
        adj = np.zeros(len(port_ret))
        vals = port_ret.values
        for i in range(len(vals)):
            adj[i] = vals[i] * live
            eq *= (1 + adj[i])
            hwm = max(hwm, eq)
            live = 0.0 if (eq / hwm - 1) < -dd_cutoff else 1.0
            if live == 0.0 and eq / hwm - 1 > -dd_cutoff * 0.5:
                live = 1.0                        # 修复一半回撤后再入场
        port_ret = pd.Series(adj, index=port_ret.index)

    out = pd.DataFrame(index=R.index)
    out["组合每日收益"] = port_ret
    out["基准每日收益"] = B.mean(axis=1)          # 基准固定等权买入持有
    out["组合净值"] = (1 + out["组合每日收益"]).cumprod()
    out["基准净值"] = (1 + out["基准每日收益"]).cumprod()
    out["HWM"] = out["组合净值"].cummax()
    out["Drawdown"] = out["组合净值"] / out["HWM"] - 1

    # 分资产收益归因：各标的贡献 = 该标的收益 × 权重，再累加
    contrib = (R * W).sum(axis=0).sort_values(ascending=False)

    last_cov = ledoit_wolf_shrink(R.iloc[-min(lookback, len(R)):])
    last_w = W.iloc[-1].values.astype(float)
    rc = _risk_contribution(last_w, last_cov)
    rc_pct = rc / rc.sum() if rc.sum() != 0 else rc

    out["再平衡成本"] = rebal_cost

    return {
        "curve": out,
        "rebal_cost_total": float(rebal_cost.sum()),
        "weights": W,
        "returns": R,
        "codes": codes,
        "contribution": contrib,
        "risk_contribution": pd.Series(rc_pct, index=codes),
        "corr": R.corr(),
        "per_asset": per_asset,
    }


# ==============================================================================
# 9. 寻优层：网格 + 参数高原 + Walk-Forward
# ==============================================================================
def _worker(args):
    """子进程任务：跑一组参数，只把标量指标传回主进程（减少序列化开销）。"""
    raw, cfg_dict, f, s = args
    cfg = BTConfig(**cfg_dict)
    cfg.fast_ma, cfg.slow_ma = f, s
    res = run_backtest(raw, cfg)
    if res.empty or len(res) < 30:
        return None
    r = res["策略每日收益"]
    years = len(res) / TRADING_DAYS
    eq = res["策略净值"].iloc[-1]
    sd = r.std()
    return {
        "快线": f, "慢线": s,
        "年化收益": (eq ** (1 / years) - 1) if years > 0 else 0.0,
        "夏普比率": (r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "最大回撤": float(res["Drawdown"].min()),
        "卡玛比率": ((eq ** (1 / years) - 1) / abs(res["Drawdown"].min())) if res["Drawdown"].min() < 0 and years > 0 else np.nan,
        "交易次数": int((res["Position"].diff().abs() > 0).sum()),
        "在场比例": float((res["Position"] > 0).mean()),
    }


def parallel_grid(raw: pd.DataFrame, cfg: BTConfig, fast_range, slow_range, step,
                  progress_cb=None, use_process=True):
    """
    并行网格。Streamlit Cloud 常常只有 1 核且禁 fork，所以多进程失败要能优雅退回串行。
    """
    params = []
    base = asdict(cfg)
    for f in range(fast_range[0], fast_range[1] + 1, step):
        for s in range(slow_range[0], slow_range[1] + 1, step):
            if f >= s:
                continue
            params.append((raw, base, f, s))
    if not params:
        return pd.DataFrame()

    results, total = [], len(params)
    done = 0
    if use_process:
        try:
            with concurrent.futures.ProcessPoolExecutor() as ex:
                futures = [ex.submit(_worker, p) for p in params]
                for fut in concurrent.futures.as_completed(futures):
                    r = fut.result()
                    if r:
                        results.append(r)
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)
            return pd.DataFrame(results)
        except Exception:
            results, done = [], 0                # 多进程不可用，退回串行

    for p in params:
        r = _worker(p)
        if r:
            results.append(r)
        done += 1
        if progress_cb:
            progress_cb(done, total)
    return pd.DataFrame(results)


def plateau_score(grid: pd.DataFrame, metric: str = "夏普比率") -> pd.DataFrame:
    """
    参数高原评分。单看最高夏普那一个点非常危险——它很可能是网格里的一根毛刺。
    这里给每组参数算"邻域均值 - 邻域标准差"：既要自己好，也要周围一圈都好。
    """
    if grid.empty:
        return grid
    g = grid.copy()
    pivot = g.pivot_table(index="快线", columns="慢线", values=metric)
    scores = []
    for _, row in g.iterrows():
        f, s = row["快线"], row["慢线"]
        fi = list(pivot.index)
        si = list(pivot.columns)
        try:
            a, b = fi.index(f), si.index(s)
        except ValueError:
            scores.append(np.nan)
            continue
        block = pivot.iloc[max(0, a - 1):a + 2, max(0, b - 1):b + 2].values.flatten()
        block = block[~np.isnan(block)]
        scores.append(block.mean() - block.std() if len(block) >= 3 else np.nan)
    g["邻域稳健分"] = scores
    return g.sort_values("邻域稳健分", ascending=False)


def walk_forward(raw: pd.DataFrame, cfg: BTConfig, fast_range, slow_range, step,
                 n_splits: int = 5, train_ratio: float = 0.7, metric: str = "夏普比率",
                 progress_cb=None):
    """
    Walk-Forward 样本外验证 —— 整个系统里最重要的一块。

    做法：把样本切成若干段，每段前 train_ratio 用来选参数（样本内），
          后面那截直接用选出来的参数跑（样本外），再把所有样本外片段拼成一条净值。
    产出 WFE（Walk-Forward Efficiency）= 样本外年化 / 样本内年化：
          接近或高于 1 说明参数是真的；远低于 1 说明只是过拟合了历史。
    """
    n = len(raw)
    if n < 300:
        return {}
    seg = n // n_splits
    oos_pieces, rows = [], []

    for k in range(n_splits):
        lo = k * seg
        hi = n if k == n_splits - 1 else (k + 1) * seg
        block = raw.iloc[lo:hi]
        cut = int(len(block) * train_ratio)
        if cut < 60 or len(block) - cut < 30:
            continue
        train, test = block.iloc[:cut], block.iloc[cut:]

        grid = parallel_grid(train, cfg, fast_range, slow_range, step, use_process=False)
        if grid.empty:
            continue
        grid = plateau_score(grid, metric)
        best = grid.dropna(subset=["邻域稳健分"]).head(1)
        if best.empty:
            best = grid.sort_values(metric, ascending=False).head(1)
        bf, bs = int(best["快线"].iloc[0]), int(best["慢线"].iloc[0])

        c2 = BTConfig(**asdict(cfg))
        c2.fast_ma, c2.slow_ma = bf, bs
        is_res = run_backtest(train, c2)
        oos_res = run_backtest(test, c2)
        if oos_res.empty:
            continue

        def _cagr(dfx):
            y = len(dfx) / TRADING_DAYS
            return (dfx["策略净值"].iloc[-1] ** (1 / y) - 1) if y > 0 else np.nan

        rows.append({
            "段": k + 1,
            "训练区间": f"{train.index[0]:%y-%m}~{train.index[-1]:%y-%m}",
            "测试区间": f"{test.index[0]:%y-%m}~{test.index[-1]:%y-%m}",
            "选中快线": bf, "选中慢线": bs,
            "样本内年化": _cagr(is_res),
            "样本外年化": _cagr(oos_res),
            "样本外回撤": float(oos_res["Drawdown"].min()),
        })
        oos_pieces.append(oos_res["策略每日收益"])
        if progress_cb:
            progress_cb(k + 1, n_splits)

    if not rows:
        return {}

    table = pd.DataFrame(rows)
    oos = pd.concat(oos_pieces).sort_index()
    eq = (1 + oos).cumprod()
    years = len(oos) / TRADING_DAYS
    is_mean = table["样本内年化"].mean()
    oos_mean = table["样本外年化"].mean()
    return {
        "table": table,
        "oos_returns": oos,
        "oos_equity": eq,
        "OOS年化": (eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "OOS夏普": (oos.mean() / oos.std() * np.sqrt(TRADING_DAYS)) if oos.std() > 0 else np.nan,
        "OOS最大回撤": float((eq / eq.cummax() - 1).min()),
        "WFE": (oos_mean / is_mean) if is_mean and is_mean == is_mean and is_mean != 0 else np.nan,
        "样本外为正比例": float((table["样本外年化"] > 0).mean()),
    }


# ==============================================================================
# 10. AI 研报层
# ==============================================================================
def build_evidence(stats: dict, tstats: dict, tests: dict, cfg: BTConfig, symbol: str) -> dict:
    """把所有数值打包成一份"证据包"。给大模型的是结构化事实，不是让它自由发挥。"""
    def rnd(d):
        return {k: (round(v, 4) if isinstance(v, (int, float)) and v == v else v)
                for k, v in d.items() if not str(k).startswith("_")}
    return {
        "标的": symbol,
        "策略配置": {k: v for k, v in asdict(cfg).items() if not isinstance(v, tuple)},
        "绩效指标": rnd(stats),
        "交易统计": rnd(tstats),
        "统计检验": rnd(tests),
    }


LLM_SYSTEM_PROMPT = """你是一名卖方量化研究员，正在为投研团队撰写一份策略回测评估报告。

硬性要求：
1. 只使用我给你的 JSON 数值，不得编造任何未出现的数字；引用数字时必须带单位。
2. 结论必须区分「统计上站得住」和「只是历史上好看」。重点看 PSR、DSR、Reality Check p 值、
   Walk-Forward 效率（WFE）与样本外表现，而不是只看累计收益。
3. 如果证据不足以支持某个判断，直接写「数据不足以判断」，不要含糊其辞。
4. 语气克制，不吹不贬。不要给投资建议，只做策略质量评估。

输出结构（Markdown，总长 600-900 字）：
## 一、结论
## 二、收益与风险
## 三、这个结果可信吗（统计检验解读）
## 四、主要风险与失效场景
## 五、下一步改进建议
"""


def local_report(ev: dict) -> str:
    """
    没有 API Key 时的本地规则模板。宁可用确定性的规则写，也不留空白。
    """
    s, t, k = ev["绩效指标"], ev["交易统计"], ev["统计检验"]

    def g(d, key, default=np.nan):
        v = d.get(key, default)
        return v if isinstance(v, (int, float)) and v == v else None

    sharpe, mdd = g(s, "夏普比率"), g(s, "最大回撤")
    cagr, excess = g(s, "年化收益"), g(s, "超额收益")
    psr, p_rc, wfe = g(k, "PSR"), g(k, "RealityCheck_p"), g(k, "WFE")

    verdicts = []
    if psr is not None:
        verdicts.append("PSR {:.0%}，{}".format(
            psr, "夏普显著为正的把握较高" if psr > 0.95 else
                 "尚不足以排除运气成分" if psr > 0.75 else "统计上很可能是噪声"))
    if p_rc is not None:
        verdicts.append("Reality Check p = {:.3f}，{}".format(
            p_rc, "超额收益难以用随机性解释" if p_rc < 0.05 else "超额收益无法与随机区分"))
    if wfe is not None:
        verdicts.append("Walk-Forward 效率 {:.2f}，{}".format(
            wfe, "样本外基本延续了样本内表现" if wfe > 0.6 else "样本外明显衰减，过拟合嫌疑大"))

    lines = [
        "## 一、结论", "",
        "标的 **{}**，策略 **{}**。".format(ev["标的"], ev["策略配置"].get("strategy", "-")),
        "回测区间年化 {}、最大回撤 {}、夏普 {}。".format(
            f"{cagr:.2%}" if cagr is not None else "-",
            f"{mdd:.2%}" if mdd is not None else "-",
            f"{sharpe:.2f}" if sharpe is not None else "-"),
        ("相对买入持有的超额收益为 {:.2%}。".format(excess) if excess is not None else ""),
        "", "## 二、收益与风险", "",
    ]
    for key in ["年化收益", "年化波动", "夏普比率", "索提诺比率", "卡玛比率",
                "最大回撤", "最长回撤天数", "在场比例", "年换手率"]:
        v = s.get(key)
        if isinstance(v, (int, float)) and v == v:
            if key in ("年化收益", "年化波动", "最大回撤", "在场比例"):
                fmt = f"{v:.2%}"
            elif key == "最长回撤天数":
                fmt = f"{int(v)} 天"
            else:
                fmt = f"{v:.2f}"
            lines.append(f"- {key}：{fmt}")
    if t:
        lines += ["", "交易层面：共 {} 笔，胜率 {:.1%}，盈亏比 {}，最大连亏 {} 次，平均持有 {:.0f} 天。".format(
            t.get("交易笔数", 0), t.get("胜率", 0) or 0,
            f"{t.get('盈亏比'):.2f}" if isinstance(t.get("盈亏比"), (int, float)) and t.get("盈亏比") == t.get("盈亏比") else "-",
            t.get("最大连亏次数", 0), t.get("平均持有天数", 0) or 0)]
    lines += ["", "## 三、这个结果可信吗", ""]
    lines += [f"- {v}" for v in verdicts] or ["- 未运行统计检验，无法判断。"]
    lines += [
        "", "## 四、主要风险与失效场景", "",
        "- 趋势类策略在震荡市中会被反复止损，最大连亏次数是实盘最难扛的部分。",
        "- 交易成本假设（佣金 {} bp、印花税 {} bp）若与实际券商费率不符，净值会明显偏移。".format(
            ev["策略配置"].get("commission_bps"), ev["策略配置"].get("stamp_duty_bps")),
        "- 单标的回测不含选股环节，换一只标的结论未必成立。",
        "", "## 五、下一步改进建议", "",
        "- 用 Walk-Forward 而不是全样本寻优来确定参数。",
        "- 扩大到多标的组合，用风险平价分散单一标的的尾部风险。",
        "- 引入更真实的滑点模型（按成交量或买卖价差），并做成本敏感性分析。",
        "", "---", "*本节由本地规则模板生成（未配置大模型 API Key）。*",
    ]
    return "\n".join([x for x in lines if x is not None])


def llm_report(ev: dict, api_key: str, base_url: str, model: str, timeout: int = 90) -> str:
    """
    调用 OpenAI 兼容接口（DeepSeek / Moonshot / 本地 vLLM 都能用同一套协议）。
    失败一律退回本地模板，绝不让页面报错。
    """
    if not (api_key and HAS_REQUESTS):
        return local_report(ev)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": "以下是回测证据包（JSON）：\n\n"
                                        + json.dumps(ev, ensure_ascii=False, indent=2)},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}",
                                           "Content-Type": "application/json"},
                             json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text + "\n\n---\n*由 {} 生成，数值全部来自本地回测证据包。*".format(model)
    except Exception as e:
        return local_report(ev) + f"\n\n> ⚠️ 大模型调用失败（{type(e).__name__}），已回退本地模板。"


# ==============================================================================
# 11. 绘图小工具（图内一律英文标签，避免云端缺中文字体变方框）
# ==============================================================================
def heatmap_fig(pivot: pd.DataFrame, title: str, xlabel: str, ylabel: str, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def hist_fig(values: np.ndarray, observed: float, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(values, bins=50, color="#4C78A8", alpha=0.85)
    ax.axvline(observed, color="#E45756", linestyle="--", linewidth=2,
               label=f"observed = {observed:.3f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def corr_fig(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Asset Correlation", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def fmt_pct(x, digits=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x*100:.{digits}f}%"


def fmt_num(x, digits=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{digits}f}"

# ==============================================================================
# 12. 侧边栏：全局参数
# ==============================================================================
st.title("🌍 全球市场量化回测与绩效研报系统")
st.caption("v2.0 · 信号融合 / 真实成本 / 统计显著性检验 / Walk-Forward 样本外 / 组合风险预算 / AI 研报")

with st.sidebar:
    st.header("① 样本与数据源")
    c_a, c_b = st.columns(2)
    start_date = c_a.date_input("开始日期", pd.to_datetime("2019-01-01"))
    end_date = c_b.date_input("结束日期", pd.to_datetime("today"))
    data_source = st.selectbox("数据源", ["自动", "yfinance", "akshare"],
                               help="A 股优先走 akshare 取前复权；失败自动退回 yfinance")

    st.header("② 策略与信号")
    strategy = st.selectbox("策略类型",
                            list(SIGNAL_LIB.keys()) + ["信号融合"])
    blend_members, blend_mode = (), "多数投票"
    if strategy == "信号融合":
        blend_members = tuple(st.multiselect("参与融合的子策略", list(SIGNAL_LIB.keys()),
                                             default=list(SIGNAL_LIB.keys())[:3]))
        blend_mode = st.radio("融合规则", ["多数投票", "全票通过", "任一触发"], horizontal=True)

    with st.expander("信号参数", expanded=(strategy in ("趋势跟踪 (MA+MACD)", "信号融合"))):
        c1, c2 = st.columns(2)
        fast_ma = c1.number_input("快速均线", 1, 100, 5)
        slow_ma = c2.number_input("慢速均线", 2, 250, 20)
        c1, c2, c3 = st.columns(3)
        macd_short = c1.number_input("MACD 短", 2, 50, 12)
        macd_long = c2.number_input("MACD 长", 3, 100, 26)
        macd_signal = c3.number_input("MACD 信号", 2, 50, 9)
        c1, c2 = st.columns(2)
        boll_period = c1.number_input("布林周期", 5, 120, 20)
        boll_std = c2.number_input("布林倍数", 0.5, 4.0, 2.0, 0.1)
        c1, c2, c3 = st.columns(3)
        rsi_period = c1.number_input("RSI 周期", 2, 60, 14)
        rsi_buy = c2.number_input("RSI 超卖", 5, 50, 30)
        rsi_sell = c3.number_input("RSI 超买", 50, 95, 70)
        c1, c2, c3 = st.columns(3)
        mom_lookback = c1.number_input("动量回看", 5, 250, 60)
        adx_period = c2.number_input("ADX 周期", 5, 60, 14)
        adx_min = c3.number_input("ADX 门槛", 0.0, 60.0, 20.0, 1.0)

    st.header("③ 风控与仓位")
    c1, c2 = st.columns(2)
    atr_period = c1.number_input("ATR 周期", 2, 60, 14)
    atr_multi = c2.number_input("ATR 止损倍数", 0.5, 10.0, 2.0, 0.1)
    init_pos = st.slider("首次建仓比例", 0.1, 1.0, 0.5, 0.1)
    add_on_atr = st.number_input("加仓触发（N 倍 ATR 浮盈）", 0.0, 5.0, 1.0, 0.1)
    time_stop = st.number_input("时间止损（交易日，0=关闭）", 0, 250, 0)
    vol_target = st.number_input("波动率目标仓位（年化，0=关闭）", 0.0, 1.0, 0.0, 0.05,
                                 help="开启后：目标波动 ÷ 近期已实现波动 = 仓位缩放系数")
    max_leverage = st.number_input("最大仓位上限", 0.1, 2.0, 1.0, 0.1)

    st.header("④ 交易成本")
    is_a_share = st.checkbox("按 A 股规则计税（卖出收印花税）", value=True)
    c1, c2 = st.columns(2)
    commission_bps = c1.number_input("单边佣金 (bp)", 0.0, 50.0, 1.0, 0.1)
    stamp_duty_bps = c2.number_input("印花税 (bp)", 0.0, 50.0, 5.0, 0.5)
    slippage_mode = st.radio("滑点模型", ["ATR 比例", "固定 bp"], horizontal=True)
    if slippage_mode == "ATR 比例":
        slippage_atr = st.number_input("滑点 = N × ATR", 0.0, 1.0, 0.05, 0.01)
        slippage_bps = 2.0
    else:
        slippage_bps = st.number_input("固定滑点 (bp)", 0.0, 100.0, 2.0, 0.5)
        slippage_atr = 0.05
    rf_annual = st.number_input("无风险利率（年化）", 0.0, 0.10, 0.0, 0.005)

    st.markdown("---")
    st.header("🗂️ 研报档案库")
    if st.session_state["history_reports"]:
        keys = list(st.session_state["history_reports"].keys())
        cur = st.session_state["current_report_key"]
        idx = keys.index(cur) if cur in keys else 0
        sel = st.selectbox("查看历史研报", keys, index=idx)
        if sel != st.session_state["current_report_key"]:
            st.session_state["current_report_key"] = sel
            st.rerun()
        if st.button("🗑️ 清空档案库"):
            st.session_state["history_reports"] = {}
            st.session_state["current_report_key"] = None
            st.rerun()
    else:
        st.info("暂无缓存，先到右侧生成一份研报。")

    st.markdown("---")
    with st.expander("🤖 AI 研报设置"):
        llm_key = st.text_input("API Key", type="password",
                                help="留空则用本地规则模板生成研报，功能不受影响")
        llm_base = st.text_input("Base URL", "https://api.deepseek.com/v1")
        llm_model = st.text_input("模型", "deepseek-chat")

CFG = BTConfig(
    strategy=strategy, fast_ma=int(fast_ma), slow_ma=int(slow_ma),
    macd_short=int(macd_short), macd_long=int(macd_long), macd_signal=int(macd_signal),
    rsi_period=int(rsi_period), rsi_buy=int(rsi_buy), rsi_sell=int(rsi_sell),
    boll_period=int(boll_period), boll_std=float(boll_std),
    mom_lookback=int(mom_lookback), adx_period=int(adx_period), adx_min=float(adx_min),
    blend_mode=blend_mode, blend_members=tuple(blend_members),
    atr_period=int(atr_period), atr_multi=float(atr_multi), init_pos=float(init_pos),
    add_on_atr=float(add_on_atr), time_stop=int(time_stop),
    vol_target=float(vol_target), max_leverage=float(max_leverage),
    commission_bps=float(commission_bps), slippage_mode=slippage_mode,
    slippage_bps=float(slippage_bps), slippage_atr=float(slippage_atr),
    stamp_duty_bps=float(stamp_duty_bps), is_a_share=bool(is_a_share),
    rf_annual=float(rf_annual),
)


def symbol_input(key: str, default_a="600519", default_us="AAPL"):
    c1, c2 = st.columns([1, 2])
    mkt = c1.selectbox("市场", ["A股 (沪深)", "美股 (NASDAQ/NYSE)"], key=f"{key}_m")
    raw = c2.text_input("代码", default_a if mkt.startswith("A股") else default_us, key=f"{key}_s")
    return normalize_symbol(raw, mkt)


def current_report():
    k = st.session_state["current_report_key"]
    if k and k in st.session_state["history_reports"]:
        return st.session_state["history_reports"][k]
    return None


# ==============================================================================
# 13. 页面
# ==============================================================================
TABS = st.tabs([
    "📑 单标的研报", "🧾 交易台账与归因", "🔬 稳健性与显著性",
    "🤖 参数寻优与高原", "🧪 Walk-Forward", "🌍 组合与风险预算", "📝 AI 研报导出",
])

# ------------------------------------------------------------------ Tab 1
with TABS[0]:
    st.subheader("生成策略研报")
    sym1 = symbol_input("t1")
    if st.button("▶️ 运行回测", type="primary"):
        with st.spinner("取数并回测中…"):
            raw = fetch_ohlcv(sym1, start_date, end_date, data_source)
            if raw.empty:
                st.error("没有取到数据，检查代码拼写或换一个数据源。")
            else:
                res = run_backtest(raw, CFG)
                if res.empty:
                    st.error("样本太短，无法回测。")
                else:
                    stats = performance_stats(res, CFG)
                    ledger = trade_ledger(res)
                    name = f"{sym1} | {CFG.strategy} | MA{CFG.fast_ma}-{CFG.slow_ma} | {time.strftime('%H:%M:%S')}"
                    st.session_state["history_reports"][name] = {
                        "symbol": sym1, "raw": raw, "result": res, "cfg": CFG,
                        "stats": stats, "ledger": ledger,
                        "tstats": trade_stats(ledger), "quality": data_quality_report(raw),
                    }
                    st.session_state["current_report_key"] = name
                    st.session_state["llm_report"] = None

    rep = current_report()
    if rep is None:
        st.info("先点上面的「运行回测」。左侧栏可以调策略、风控和成本假设。")
    else:
        res, stats, cfg = rep["result"], rep["stats"], rep["cfg"]
        st.success(f"当前研报：**{st.session_state['current_report_key']}**")

        with st.expander("🩺 数据质量体检（回测前先看这里）"):
            q = rep["quality"]
            cols = st.columns(len(q))
            for (k, v), col in zip(q.items(), cols):
                col.metric(k, v)
            if q.get("疑似停牌(零成交)", 0) > 0 or q.get("异常跳空(>11%)", 0) > 0:
                st.warning("样本里存在停牌日或异常跳空，回测结果可能被这几天放大，建议核对原始数据。")

        st.markdown("### 核心绩效")
        m = st.columns(6)
        m[0].metric("累计收益", fmt_pct(stats["累计收益"]), f"基准 {fmt_pct(stats['基准累计收益'])}")
        m[1].metric("年化收益", fmt_pct(stats["年化收益"]))
        m[2].metric("年化波动", fmt_pct(stats["年化波动"]))
        m[3].metric("最大回撤", fmt_pct(stats["最大回撤"]), "风控", delta_color="inverse")
        m[4].metric("夏普", fmt_num(stats["夏普比率"]))
        m[5].metric("卡玛", fmt_num(stats["卡玛比率"]))

        m = st.columns(6)
        m[0].metric("索提诺", fmt_num(stats["索提诺比率"]))
        m[1].metric("Omega", fmt_num(stats["Omega"]))
        m[2].metric("信息比率", fmt_num(stats["信息比率"]))
        m[3].metric("年化 Alpha", fmt_pct(stats["年化Alpha"]))
        m[4].metric("Beta", fmt_num(stats["Beta"]))
        m[5].metric("最长回撤", f"{stats['最长回撤天数']} 天")

        m = st.columns(6)
        m[0].metric("日 VaR95", fmt_pct(stats["日VaR95"]))
        m[1].metric("日 CVaR95", fmt_pct(stats["日CVaR95"]))
        m[2].metric("偏度", fmt_num(stats["偏度"]))
        m[3].metric("峰度", fmt_num(stats["峰度"]))
        m[4].metric("在场比例", fmt_pct(stats["在场比例"]))
        m[5].metric("年换手", fmt_num(stats["年换手率"]))

        st.caption(f"⚠️ 累计交易成本吃掉了 {fmt_pct(stats['累计成本占比'])} 的本金"
                   f"（佣金 {cfg.commission_bps}bp + {cfg.slippage_mode} 滑点"
                   f"{' + 印花税 ' + str(cfg.stamp_duty_bps) + 'bp' if cfg.is_a_share else ''}）。"
                   "成本假设是回测里最容易自欺欺人的地方，改一改这里再看结论稳不稳。")

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### 净值走势（已扣成本）")
            st.line_chart(res[["基准净值", "策略净值"]])
            st.markdown("#### 回撤")
            st.area_chart(res[["Drawdown", "基准Drawdown"]] * 100)
            st.markdown("#### 仓位变化（0 = 空仓）")
            st.line_chart(res[["Position"]])
        with c2:
            st.markdown("#### 月度收益")
            mt = monthly_return_table(res)
            st.dataframe(mt.style.format("{:.2%}", na_rep="-")
                         .background_gradient(cmap="RdYlGn", axis=None, vmin=-0.12, vmax=0.12),
                         use_container_width=True, height=380)
            st.markdown("#### 滚动指标（126 日）")
            rm = rolling_metrics(res, 126)
            if not rm.empty:
                st.line_chart(rm[["滚动夏普"]])

# ------------------------------------------------------------------ Tab 2
with TABS[1]:
    st.subheader("逐笔交易台账")
    rep = current_report()
    if rep is None:
        st.info("先在「单标的研报」里跑一次回测。")
    else:
        ledger, tstats = rep["ledger"], rep["tstats"]
        if ledger.empty:
            st.warning("回测区间内没有产生任何完整交易 —— 信号太严或样本太短。")
        else:
            m = st.columns(6)
            m[0].metric("交易笔数", tstats["交易笔数"])
            m[1].metric("胜率", fmt_pct(tstats["胜率"], 1))
            m[2].metric("盈亏比", fmt_num(tstats["盈亏比"]))
            m[3].metric("单笔期望", fmt_pct(tstats["单笔期望"]))
            m[4].metric("最大连亏", f"{tstats['最大连亏次数']} 次", "心理抗压阈值", delta_color="inverse")
            m[5].metric("平均持有", f"{tstats['平均持有天数']:.0f} 天")

            st.markdown("#### 离场原因归因")
            grp = ledger.groupby("离场原因").agg(
                笔数=("区间收益", "size"),
                平均收益=("区间收益", "mean"),
                合计贡献=("区间收益", "sum"),
                平均持有天数=("持有天数", "mean"),
            ).sort_values("笔数", ascending=False)
            st.dataframe(grp.style.format({"平均收益": "{:.2%}", "合计贡献": "{:.2%}",
                                           "平均持有天数": "{:.0f}"}),
                         use_container_width=True)
            st.caption("如果「初始止损」占了大多数笔数且合计贡献显著为负，说明入场信号太钝或止损设得太紧，"
                       "应该先改信号而不是继续调止损倍数。")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 单笔收益分布")
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.hist(ledger["区间收益"] * 100, bins=30, color="#4C78A8", alpha=0.85)
                ax.axvline(0, color="#333", linewidth=1)
                ax.set_xlabel("Trade return (%)", fontsize=9)
                ax.set_title("Per-trade Return Distribution", fontsize=11)
                fig.tight_layout()
                st.pyplot(fig)
            with c2:
                st.markdown("#### 收益 vs 持有天数")
                fig, ax = plt.subplots(figsize=(6, 3.4))
                colors = np.where(ledger["区间收益"] > 0, "#59A14F", "#E45756")
                ax.scatter(ledger["持有天数"], ledger["区间收益"] * 100, c=colors, alpha=0.75)
                ax.axhline(0, color="#333", linewidth=1)
                ax.set_xlabel("Holding days", fontsize=9)
                ax.set_ylabel("Return (%)", fontsize=9)
                ax.set_title("Return vs Holding Period", fontsize=11)
                fig.tight_layout()
                st.pyplot(fig)

            st.markdown("#### 明细")
            st.dataframe(ledger.style.format({"区间收益": "{:.2%}", "区间成本": "{:.4%}"}),
                         use_container_width=True, height=360)
            st.download_button("⬇️ 下载交易台账 CSV",
                               ledger.to_csv(index=False).encode("utf-8-sig"),
                               file_name="trade_ledger.csv", mime="text/csv")

# ------------------------------------------------------------------ Tab 3
with TABS[2]:
    st.subheader("这个结果是真本事，还是撞大运？")
    st.caption("回测最大的谎言不是算错，而是算对了但没意义。这一页用统计检验回答：观测到的表现能否与随机区分。")
    rep = current_report()
    if rep is None:
        st.info("先在「单标的研报」里跑一次回测。")
    else:
        res, stats = rep["result"], rep["stats"]
        c1, c2, c3 = st.columns(3)
        n_boot = c1.number_input("Bootstrap 次数", 100, 3000, 500, 100)
        block = c2.number_input("区块长度（交易日）", 5, 120, 20, 5,
                                help="区块自助法保留块内的自相关与波动聚集，比逐日重抽更贴近真实")
        n_sim = c3.number_input("蒙特卡洛模拟次数", 200, 10000, 2000, 200)
        mc_mode = st.radio("蒙特卡洛模式", ["有放回重抽", "随机排列"], horizontal=True,
                           help="随机排列只打乱顺序，终值恒定（乘法可交换），只能看回撤分布；"
                                "有放回重抽把历史交易当分布重新抽样，终值与亏损概率才有意义")
        n_trials = st.number_input("此前一共试过多少组参数（用于 DSR 多重检验校正）",
                                   1, 100000, 1, 1,
                                   help="跑完「参数寻优」后回来填网格组合数，DSR 才有意义")

        if st.button("🔬 运行全部检验", type="primary"):
            r = res["策略每日收益"].dropna().values
            b = res["每日收益率"].reindex(res["策略每日收益"].dropna().index).fillna(0).values
            with st.spinner("Bootstrap / 蒙特卡洛 / Reality Check 计算中…"):
                boot = block_bootstrap(r, int(n_boot), int(block))
                mc = monte_carlo_trades(rep["ledger"]["区间收益"].values if not rep["ledger"].empty
                                        else np.array([]), int(n_sim), mode=mc_mode)
                p_rc = reality_check(r, b, int(n_boot), int(block))
                psr = probabilistic_sharpe_ratio(stats["夏普比率"], len(r),
                                                 stats["偏度"], stats["峰度"])
                sr_var = float(np.var(boot[:, 1])) if len(boot) else 0.0
                dsr = deflated_sharpe_ratio(stats["夏普比率"], len(r), stats["偏度"],
                                            stats["峰度"], int(n_trials), sr_var)
                rep["tests"] = {
                    "PSR": psr, "DSR": dsr, "RealityCheck_p": p_rc,
                    "Bootstrap_CAGR_5%": float(np.percentile(boot[:, 0], 5)) if len(boot) else np.nan,
                    "Bootstrap_CAGR_95%": float(np.percentile(boot[:, 0], 95)) if len(boot) else np.nan,
                    "Bootstrap_Sharpe_5%": float(np.percentile(boot[:, 1], 5)) if len(boot) else np.nan,
                    "Bootstrap_Sharpe_95%": float(np.percentile(boot[:, 1], 95)) if len(boot) else np.nan,
                    "Bootstrap_MaxDD_5%": float(np.percentile(boot[:, 2], 5)) if len(boot) else np.nan,
                    "参数试验次数": int(n_trials),
                    **{k: v for k, v in mc.items() if not k.startswith("_")},
                }
                rep["_boot"] = boot
                rep["_mc"] = mc

        tests = rep.get("tests")
        if tests:
            m = st.columns(4)
            psr, dsr, p_rc = tests.get("PSR"), tests.get("DSR"), tests.get("RealityCheck_p")
            m[0].metric("PSR 概率夏普", fmt_pct(psr, 1) if psr == psr else "-",
                        "夏普>0 的把握")
            m[1].metric("DSR 收缩夏普", fmt_pct(dsr, 1) if dsr and dsr == dsr else "-",
                        f"已按 {tests.get('参数试验次数')} 次试验校正")
            m[2].metric("Reality Check p", fmt_num(p_rc, 3) if p_rc == p_rc else "-",
                        "越小越像真超额")
            m[3].metric("蒙特卡洛亏损概率", fmt_pct(tests.get("亏损概率"), 1)
                        if tests.get("亏损概率") is not None else "-",
                        tests.get("模式", ""))

            verdict = []
            if psr == psr:
                verdict.append("✅ PSR {:.1%}：夏普显著为正".format(psr) if psr > 0.95
                               else "⚠️ PSR {:.1%}：还不足以排除运气".format(psr) if psr > 0.75
                               else "❌ PSR {:.1%}：统计上更像噪声".format(psr))
            if dsr and dsr == dsr:
                verdict.append("✅ DSR {:.1%}：扣掉多重检验红利后依然显著".format(dsr) if dsr > 0.95
                               else "❌ DSR {:.1%}：把「试了很多组参数」算进来后，优势基本消失".format(dsr))
            if p_rc == p_rc:
                verdict.append("✅ Reality Check p={:.3f}：超额收益难以用随机解释".format(p_rc) if p_rc < 0.05
                               else "❌ Reality Check p={:.3f}：超额收益与随机不可区分".format(p_rc))
            for v in verdict:
                st.markdown(f"- {v}")

            st.markdown("#### Bootstrap 置信区间（500 次区块重抽）")
            ci = pd.DataFrame({
                "指标": ["年化收益", "夏普比率", "最大回撤"],
                "点估计": [stats["年化收益"], stats["夏普比率"], stats["最大回撤"]],
                "5% 分位": [tests["Bootstrap_CAGR_5%"], tests["Bootstrap_Sharpe_5%"], tests["Bootstrap_MaxDD_5%"]],
                "95% 分位": [tests["Bootstrap_CAGR_95%"], tests["Bootstrap_Sharpe_95%"], np.nan],
            })
            st.dataframe(ci.style.format({"点估计": "{:.3f}", "5% 分位": "{:.3f}", "95% 分位": "{:.3f}"}),
                         use_container_width=True)

            boot = rep.get("_boot")
            mc = rep.get("_mc")
            c1, c2 = st.columns(2)
            if boot is not None and len(boot):
                with c1:
                    st.pyplot(hist_fig(boot[:, 1], stats["夏普比率"],
                                       "Bootstrapped Sharpe Distribution", "Annualised Sharpe"))
            if mc:
                with c2:
                    st.pyplot(hist_fig(mc["_mdds"] * 100, stats["最大回撤"] * 100,
                                       "Monte-Carlo Max Drawdown Distribution", "Max drawdown (%)"))
                st.warning(
                    "蒙特卡洛把同样这些交易打乱顺序重跑 {} 次：最坏 5% 的情形下回撤会到 {}，"
                    "而历史上只出现了 {}。实盘要按前者准备心理和风控预案，而不是后者。".format(
                        int(n_sim), fmt_pct(mc["回撤95%分位(最坏)"]), fmt_pct(stats["最大回撤"])))

# ------------------------------------------------------------------ Tab 4
with TABS[3]:
    st.subheader("参数寻优：不要找最高的那个点，要找最宽的那片高原")
    sym4 = symbol_input("t4")
    c1, c2, c3 = st.columns(3)
    fr = c1.slider("快线搜索范围", 2, 40, (3, 15))
    sr_ = c2.slider("慢线搜索范围", 10, 150, (20, 60))
    step = c3.number_input("步长", 1, 10, 2)
    use_mp = st.checkbox("启用多进程（云端单核时会自动退回串行）", value=True)

    if st.button("🚀 启动网格寻优", type="primary"):
        raw = fetch_ohlcv(sym4, start_date, end_date, data_source)
        if raw.empty:
            st.error("取数失败。")
        else:
            bar = st.progress(0)
            txt = st.empty()
            t0 = time.time()

            def cb(done, total):
                bar.progress(min(int(done / total * 100), 100))
                txt.text(f"已完成 {done}/{total} 组 | 耗时 {time.time()-t0:.1f}s")

            grid = parallel_grid(raw, CFG, fr, sr_, int(step), cb, use_mp)
            if grid.empty:
                st.error("没有有效结果，放宽一下搜索范围。")
            else:
                st.session_state["opt_results"] = plateau_score(grid, "夏普比率")
                st.success(f"完成 {len(grid)} 组，耗时 {time.time()-t0:.1f}s")

    grid = st.session_state.get("opt_results")
    if grid is not None and not grid.empty:
        best_raw = grid.sort_values("夏普比率", ascending=False).iloc[0]
        best_rob = grid.dropna(subset=["邻域稳健分"]).iloc[0] if grid["邻域稳健分"].notna().any() else best_raw
        c1, c2 = st.columns(2)
        c1.info(f"**最高夏普点**：快线 {int(best_raw['快线'])} / 慢线 {int(best_raw['慢线'])}，"
                f"夏普 {best_raw['夏普比率']:.2f}，回撤 {best_raw['最大回撤']:.2%}")
        c2.success(f"**邻域最稳点（推荐）**：快线 {int(best_rob['快线'])} / 慢线 {int(best_rob['慢线'])}，"
                   f"夏普 {best_rob['夏普比率']:.2f}，稳健分 {best_rob['邻域稳健分']:.2f}")
        st.caption("两者不一致时优先选后者：最高点常常是网格上的一根毛刺，换个样本就塌了；"
                   "邻域稳健分 = 周围 3×3 参数的夏普均值 − 标准差，衡量的是「这一片」而不是「这一点」。")

        metric_pick = st.selectbox("热力图指标", ["夏普比率", "年化收益", "最大回撤", "卡玛比率", "邻域稳健分"])
        pivot = grid.pivot_table(index="快线", columns="慢线", values=metric_pick)
        st.pyplot(heatmap_fig(pivot, f"Parameter Surface — {metric_pick}",
                              "Slow MA", "Fast MA",
                              cmap="RdYlGn" if metric_pick != "最大回撤" else "RdYlGn_r"))

        st.dataframe(grid.head(30).style.format({
            "年化收益": "{:.2%}", "夏普比率": "{:.2f}", "最大回撤": "{:.2%}",
            "卡玛比率": "{:.2f}", "在场比例": "{:.1%}", "邻域稳健分": "{:.2f}"}),
            use_container_width=True, height=340)
        st.download_button("⬇️ 下载寻优结果 CSV",
                           grid.to_csv(index=False).encode("utf-8-sig"),
                           file_name="grid_search.csv", mime="text/csv")
        st.info(f"本次共试了 **{len(grid)}** 组参数。把这个数字填到「稳健性与显著性」页的 "
                f"DSR 试验次数里，才能知道最优夏普里有多少是多重检验白捡的。")

# ------------------------------------------------------------------ Tab 5
with TABS[4]:
    st.subheader("Walk-Forward：只用过去的信息选参数，再拿它赌未来")
    st.caption("全样本寻优的结果永远好看，因为参数已经看过答案。WFA 把「选参数」和「验证」在时间上彻底隔开。")
    sym5 = symbol_input("t5")
    c1, c2, c3, c4 = st.columns(4)
    n_splits = c1.number_input("切分段数", 3, 10, 5)
    train_ratio = c2.slider("训练占比", 0.5, 0.9, 0.7, 0.05)
    fr5 = c3.slider("快线范围", 2, 40, (3, 15), key="wf_f")
    sr5 = c4.slider("慢线范围", 10, 150, (20, 60), key="wf_s")
    step5 = st.number_input("步长", 1, 10, 3, key="wf_step")

    if st.button("🧪 运行 Walk-Forward", type="primary"):
        raw = fetch_ohlcv(sym5, start_date, end_date, data_source)
        if raw.empty:
            st.error("取数失败。")
        elif len(raw) < 300:
            st.error("样本不足 300 个交易日，WFA 没有意义，拉长回测区间。")
        else:
            bar = st.progress(0)
            with st.spinner("逐段优化并做样本外验证…"):
                wfa = walk_forward(raw, CFG, fr5, sr5, int(step5), int(n_splits),
                                   float(train_ratio),
                                   progress_cb=lambda d, t: bar.progress(int(d / t * 100)))
            if not wfa:
                st.error("没有产生有效分段，减少段数或拉长样本。")
            else:
                st.session_state["wfa_results"] = wfa

    wfa = st.session_state.get("wfa_results")
    if wfa:
        m = st.columns(5)
        m[0].metric("样本外年化", fmt_pct(wfa["OOS年化"]))
        m[1].metric("样本外夏普", fmt_num(wfa["OOS夏普"]))
        m[2].metric("样本外最大回撤", fmt_pct(wfa["OOS最大回撤"]))
        m[3].metric("WFE 效率", fmt_num(wfa["WFE"]), "样本外/样本内")
        m[4].metric("样本外为正比例", fmt_pct(wfa["样本外为正比例"], 0))

        wfe = wfa["WFE"]
        if wfe == wfe:
            if wfe > 0.8:
                st.success("WFE {:.2f}：样本外几乎完整保留了样本内表现，参数是稳的。".format(wfe))
            elif wfe > 0.4:
                st.warning("WFE {:.2f}：样本外衰减明显但仍为正，可以用，但要下调收益预期。".format(wfe))
            else:
                st.error("WFE {:.2f}：样本外几乎没有延续性，当前参数高度依赖历史，不建议直接上。".format(wfe))

        st.markdown("#### 样本外拼接净值")
        st.line_chart(wfa["oos_equity"])
        st.markdown("#### 分段明细")
        st.dataframe(wfa["table"].style.format({
            "样本内年化": "{:.2%}", "样本外年化": "{:.2%}", "样本外回撤": "{:.2%}"}),
            use_container_width=True)
        st.caption("看「选中快线/慢线」这两列：如果每一段选出来的参数跳来跳去，说明参数本身没有稳定含义，"
                   "即使 WFE 还不错也要谨慎。")

# ------------------------------------------------------------------ Tab 6
with TABS[5]:
    st.subheader("多资产组合：把钱按风险分，而不是按金额分")
    codes_raw = st.text_input("资产池（逗号分隔，A 股填 6 位代码）", "AAPL, MSFT, NVDA, TSLA, GLD")
    c1, c2, c3 = st.columns(3)
    weight_method = c1.selectbox("权重模型",
                                 ["ERC 风险平价", "逆波动率", "最小方差", "最大分散化", "等权重"])
    rebalance = c2.selectbox("再平衡频率", ["每月", "每周", "每季", "每日"])
    lookback = c3.number_input("协方差回看窗口", 20, 250, 60, 10)
    c1, c2, c3 = st.columns(3)
    drift_th = c1.number_input("权重漂移阈值（低于则不调仓）", 0.0, 0.5, 0.0, 0.01)
    port_vol_target = c2.number_input("组合目标波动率（年化，0=关闭）", 0.0, 0.5, 0.0, 0.01)
    dd_cutoff = c3.number_input("回撤熔断阈值（0=关闭）", 0.0, 0.6, 0.0, 0.05)

    if st.button("▶️ 运行组合回测", type="primary"):
        syms = []
        for s in codes_raw.split(","):
            s = s.strip()
            if not s:
                continue
            digits = "".join(ch for ch in s if ch.isdigit())
            syms.append(normalize_symbol(s, "A股" if len(digits) == 6 and len(s) <= 7 else "美股"))
        with st.spinner(f"拉取 {len(syms)} 只标的并执行 {weight_method}…"):
            data = {}
            for s in syms:
                df = fetch_ohlcv(s, start_date, end_date, data_source)
                if not df.empty:
                    data[s] = df
            if len(data) < 2:
                st.error("至少要有 2 只标的成功取数。")
            else:
                st.session_state["portfolio_result"] = run_portfolio(
                    data, CFG, weight_method, int(lookback), rebalance,
                    float(drift_th), float(port_vol_target), float(dd_cutoff))

    pr = st.session_state.get("portfolio_result")
    if pr:
        curve = pr["curve"]
        r = curve["组合每日收益"]
        years = len(curve) / TRADING_DAYS
        cagr = curve["组合净值"].iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
        m = st.columns(5)
        m[0].metric("组合累计", fmt_pct(curve["组合净值"].iloc[-1] - 1),
                    f"等权基准 {fmt_pct(curve['基准净值'].iloc[-1] - 1)}")
        m[1].metric("组合年化", fmt_pct(cagr))
        m[2].metric("组合夏普", fmt_num(sharpe))
        m[3].metric("组合最大回撤", fmt_pct(curve["Drawdown"].min()))
        m[4].metric("再平衡成本累计", fmt_pct(pr.get("rebal_cost_total", 0.0)),
                    f"{len(pr['codes'])} 只标的")

        st.line_chart(curve[["基准净值", "组合净值"]])
        st.area_chart(curve[["Drawdown"]] * 100)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 权重演化")
            st.area_chart(pr["weights"])
            st.markdown("#### 收益贡献分解")
            st.bar_chart(pr["contribution"])
        with c2:
            st.markdown("#### 期末风险贡献占比")
            rcdf = pr["risk_contribution"].to_frame("风险贡献占比")
            st.dataframe(rcdf.style.format("{:.1%}"), use_container_width=True)
            st.caption("ERC 的目标就是让这一列各行尽量相等。如果仍然一家独大，"
                       "说明回看窗口太短或该标的与其他资产高度相关，分散不掉。")
            st.markdown("#### 相关性矩阵")
            st.pyplot(corr_fig(pr["corr"]))

# ------------------------------------------------------------------ Tab 7
with TABS[6]:
    st.subheader("AI 研报生成")
    st.caption("大模型只负责组织语言和解释统计含义，所有数字都来自本地回测证据包，它不能也不会自己编数。")
    rep = current_report()
    if rep is None:
        st.info("先跑一次回测。建议先把「稳健性与显著性」也跑完，研报的含金量主要在那一段。")
    else:
        tests = dict(rep.get("tests") or {})
        wfa = st.session_state.get("wfa_results")
        if wfa:
            tests["WFE"] = wfa.get("WFE")
            tests["OOS年化"] = wfa.get("OOS年化")
            tests["OOS夏普"] = wfa.get("OOS夏普")
        ev = build_evidence(rep["stats"], rep["tstats"], tests, rep["cfg"], rep["symbol"])

        with st.expander("📦 查看交给模型的证据包（JSON）"):
            st.json(ev)

        c1, c2 = st.columns([1, 1])
        if c1.button("📝 生成研报", type="primary"):
            with st.spinner("撰写中…"):
                st.session_state["llm_report"] = llm_report(ev, llm_key, llm_base, llm_model)
        if c2.button("🧱 只用本地模板生成"):
            st.session_state["llm_report"] = local_report(ev)

        text = st.session_state.get("llm_report")
        if text:
            st.markdown("---")
            st.markdown(text)
            st.download_button("⬇️ 下载研报 Markdown", text.encode("utf-8"),
                               file_name=f"report_{rep['symbol']}.md", mime="text/markdown")
            buf = io.StringIO()
            buf.write(json.dumps(ev, ensure_ascii=False, indent=2))
            st.download_button("⬇️ 下载证据包 JSON", buf.getvalue().encode("utf-8"),
                               file_name=f"evidence_{rep['symbol']}.json", mime="application/json")

st.markdown("---")
st.caption(
    "依赖状态：yfinance {} · akshare {} · scipy {} · requests {} · 中文字体 {}".format(
        "✅" if HAS_YF else "❌", "✅" if HAS_AK else "❌",
        "✅" if HAS_SCIPY else "❌（ERC/最小方差已降级为逆波动近似）",
        "✅" if HAS_REQUESTS else "❌", "✅" if CJK_FONT_OK else "图内使用英文标签"))
