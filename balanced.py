"""Stock and bond ETF allocation, using same-currency market-price histories."""

import pandas as pd
import streamlit as st

from balanced_engine import run_balanced
from ui_text import bilingual as L

# Only these same-currency pairs are offered; no implicit FX conversion.
EQUITIES = {"USD": ("SPY", "VTI"), "CNY": ("510300.SS",)}
BONDS = {"USD": ("BND", "SHY", "IEF", "TLT"), "CNY": ("511010.SS", "511260.SS")}
FUNDS = {
    "SPY": ("S&P 500", "标普 500"),
    "VTI": ("US total stock market", "美国全市场股票"),
    "BND": ("US investment-grade aggregate bonds", "美国投资级综合债券"),
    "SHY": ("1–3 year US Treasuries", "美国 1–3 年期国债"),
    "IEF": ("7–10 year US Treasuries", "美国 7–10 年期国债"),
    "TLT": ("20+ year US Treasuries", "美国 20 年以上国债"),
    "510300.SS": ("Huatai-PineBridge CSI 300", "华泰柏瑞沪深 300 ETF"),
    "511010.SS": ("Guotai SSE 5-year government bonds", "国泰上证 5 年期国债 ETF"),
    "511260.SS": ("Guotai SSE 10-year government bonds", "国泰上证 10 年期国债 ETF"),
}
VANGUARD = "https://workplace.vanguard.com/assets/corp/fund_communications/pdf_publish/us-products/fact-sheet/F0928.pdf"
DURATION = {
    "BND": (5.7, "2026-09-30", VANGUARD),
    "SHY": (1.79, "2026-11-17", "https://www.ishares.com/us/products/239452/ishares-13-year-treasury-bond-etf"),
    "IEF": (
        6.78,
        "2026-11-17",
        "https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf",
    ),
    "TLT": (
        14.70,
        "2026-11-17",
        "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf",
    ),
}
GS_METHOD = "https://github.com/goldmansachs/gs-quant/blob/master/gs_quant/timeseries/measures_reports.py"
QL_METHOD = "https://github.com/lballabio/QuantLib/blob/master/ql/cashflows/cashflows.hpp"


def fund_name(symbol):
    return symbol.split(".")[0] + " · " + L(*FUNDS[symbol])


def completed_end(end, currency):
    zone, close = ("America/New_York", (16, 15)) if currency == "USD" else ("Asia/Shanghai", (15, 15))
    now = pd.Timestamp.now(tz=zone)
    last = (now - pd.Timedelta(days=int((now.hour, now.minute) < close))).date()
    return min(end, last)


ETF_OHLCV = ["开盘", "最高", "最低", "收盘", "成交量"]


@st.cache_data(ttl=3600, show_spinner=False)
def _china_etf_daily(symbol, start, end):
    import akshare as ak

    frame = ak.fund_etf_hist_em(
        symbol=symbol.split(".")[0],
        period="daily",
        start_date=pd.Timestamp(start).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end).strftime("%Y%m%d"),
        adjust="hfq",
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.set_index("日期")


def _etf_history(fetch, symbol, start, end, source, columns):
    """Load genuine adjusted bars, or just adjusted closes for allocation runs."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".SH"):
        symbol = symbol[:-3] + ".SS"
    if len(symbol) == 6 and symbol.isdigit():
        if symbol.startswith("5"):
            symbol += ".SS"
        elif symbol.startswith("1"):
            symbol += ".SZ"
    is_china = symbol.endswith((".SS", ".SZ"))
    first = pd.Timestamp(start).date()
    last = completed_end(pd.Timestamp(end).date(), "CNY" if is_china else "USD")
    empty = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]), dtype=float)
    if first > last:
        return empty
    sources = ("yfinance", "akshare") if source == "自动" else (source,)
    for provider in sources:
        try:
            if provider == "akshare" and is_china:
                frame = _china_etf_daily(symbol, first, last)
            elif provider == "yfinance":
                frame = fetch(symbol, first, last, "yfinance", adjusted=True)
            else:
                continue
            if frame is None or frame.empty or not set(columns).issubset(frame.columns):
                continue
            bars = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
            # Preserve the provider's local session date when dropping its timezone.
            bars.index = pd.DatetimeIndex(pd.to_datetime(bars.index, errors="coerce")).tz_localize(None).normalize()
            bars = bars[bars.index.notna()].sort_index(kind="stable").loc[pd.Timestamp(first):pd.Timestamp(last)]
            bars = bars.replace([float("inf"), -float("inf")], float("nan")).dropna()
            bars = bars[bars["收盘"] > 0]
            if "最低" in columns:
                valid = ((bars["最低"] > 0) & (bars["成交量"] >= 0)
                         & (bars["最低"] <= bars[["开盘", "收盘"]].min(axis=1))
                         & (bars["最高"] >= bars[["开盘", "收盘"]].max(axis=1)))
                bars = bars[valid]
            if not bars.empty:
                bars = bars[~bars.index.duplicated(keep="last")]
                bars.attrs["provider"] = provider
                return bars
        except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError):
            continue
    return empty


def bond_etf_ohlcv(fetch, symbol, start, end, source):
    return _etf_history(fetch, symbol, start, end, source, ETF_OHLCV)


def prices_for(fetch, symbol, start, end, source):
    frame = _etf_history(fetch, symbol, start, end, source, ["收盘"])
    return frame["收盘"], frame.attrs.get("provider")


def allocation_name(method):
    return L("Fixed mix", "固定比例") if method == "fixed" else L("Inverse volatility", "逆波动率")


def stress_view(result):
    curve = result["curve"]
    bond = result["bond"]
    ew, bw = curve[["equity_weight", "bond_weight"]].iloc[-1]
    st.markdown(L("#### What if stocks fall and yields rise?", "#### 如果股票下跌、收益率上升"))
    st.caption(
        L(
            f"End-of-sample holdings: {ew:.1%} equity and {bw:.1%} bonds. Change either shock to reprice these holdings.",
            f"按期末实际持仓测算：股票 {ew:.1%}，债券 {bw:.1%}。调整下列冲击可查看组合损益。",
        )
    )
    a, b, c = st.columns(3)
    move = a.number_input(
        L("Equity shock (%)", "股票涨跌（%）"), -100.0, 100.0, -10.0, 1.0, key="sb_shock_equity"
    )
    rate = b.number_input(L("Yield shock (bp)", "收益率变动（基点）"), -300, 300, 100, 25, key="sb_shock_yield")
    default = DURATION.get(bond)
    dur = c.number_input(
        L("Assumed duration (years)", "假设久期（年）"),
        min_value=0.0,
        max_value=40.0,
        value=default[0] if default else None,
        step=0.1,
        format="%.2f",
        key="sb_duration_" + bond,
        help=L(
            "Use an effective or modified duration appropriate to the fund and valuation date.",
            "可填入与基金及估值日相匹配的有效久期或修正久期。",
        ),
    )
    if default:
        _, date, link = default
        st.caption(
            L(
                f"The editable {bond} starting assumption is from the issuer on {date}; the backtest weights use only historical prices. [Issuer data]({link})",
                f"{bond} 的初始久期取自发行人 {date} 的披露，可修改。历史回测权重仅使用历史价格。[发行人资料]({link})",
            )
        )
    else:
        st.caption(
            L(
                "Enter a duration from a fund disclosure or your own scenario. The maturity in the ETF name describes the tracked bonds.",
                "请填写基金披露的久期或你的情景假设。ETF 名称中的年限描述其跟踪债券的期限。",
            )
        )
    if dur is not None:
        stock_pnl = ew * move / 100
        bond_pnl = -bw * dur * rate / 10000
        a, b, c = st.columns(3)
        a.metric(L("Portfolio impact", "组合损益"), f"{stock_pnl+bond_pnl:+.2%}")
        b.metric(L("From equities", "股票贡献"), f"{stock_pnl*100:+.2f} pp")
        c.metric(L("From bonds", "债券贡献"), f"{bond_pnl*100:+.2f} pp")
    st.caption(
        L(
            "First-order estimate for a parallel yield shift: bond price change ≈ −duration × yield change. Credit-spread moves, convexity and income during the shock are excluded.",
            "平行移位的一阶估算：债券价格变动约等于 −久期 × 收益率变动。未计信用利差、凸性及冲击期间的票息。",
        )
    )


def show_balanced(result):
    curve = result["curve"]
    sym, bond = result["equity"], result["bond"]
    strategy = allocation_name(result["method"])
    fixed = result["method"] == "fixed"
    if fixed:
        mix = result["target_equity"]
        strategy += f" ({mix:.0%}/{1-mix:.0%})"
    bench = L("60/40 benchmark", "60/40 基准")
    # Distinguish the two legend entries even for a 60/40 selected mix.
    title = f"{sym.split('.')[0]} + {bond.split('.')[0]} · {result['currency']}"
    st.markdown("### " + title)
    freq = L("monthly", "每月") if result["rebalance"] == "monthly" else L("quarterly", "每季")
    st.caption(
        L(
            f"Last run · {strategy} · {freq} · {result['cost_bps']:g} bp one way · {result['start']:%Y-%m-%d} to {result['end']:%Y-%m-%d} · {result['sessions']} holding intervals.",
            f"上次运行 · {strategy} · {freq}再平衡 · 单边成本 {result['cost_bps']:g} bp · {result['start']:%Y-%m-%d} 至 {result['end']:%Y-%m-%d} · {result['sessions']} 个持有区间。",
        )
    )
    m = result["metrics"].rename(index={"strategy": strategy, "benchmark": bench})
    columns = {
        "total_return": L("Net return", "净累计收益"),
        "cagr": L("Annual return", "年化收益"),
        "volatility": L("Annual volatility", "年化波动"),
        "max_drawdown": L("Max drawdown", "最大回撤"),
        "cost": L("Fees / starting capital", "成本 / 初始本金"),
    }
    st.dataframe(m[list(columns)].rename(columns=columns).style.format("{:.2%}", na_rep="—"), width="stretch")
    st.line_chart(
        curve[["strategy_nav", "benchmark_nav"]].rename(
            columns={"strategy_nav": strategy, "benchmark_nav": bench}
        )
    )
    if fixed and result["target_equity"] == 0.6:
        st.caption(
            L(
                "Both curves coincide for a fixed 60/40 mix. Change the equity target or use inverse volatility to compare a different allocation.",
                "当前固定配比与 60/40 基准相同，两条净值曲线重合。修改股票占比或选择逆波动率即可比较另一种配置。",
            )
        )
    with st.expander(L("Holdings and return attribution", "持仓与收益归因")):
        st.area_chart(
            curve[["equity_weight", "bond_weight"]].rename(columns={"equity_weight": sym, "bond_weight": bond})
        )
        att = result["attribution"]
        a, b, c = st.columns(3)
        for column, label, key in (
            (a, sym, "equity"),
            (b, bond, "bond"),
            (c, L("Trading costs", "交易成本"), "cost"),
        ):
            column.metric(label, f"{att[key]*100:+.2f} pp")
        st.caption(
            L(
                "Contributions are measured against initial capital and add up to net cumulative return.",
                "各项损益以初始本金为分母，加总即组合净累计收益。",
            )
        )
        st.line_chart(
            curve[["rolling_corr"]].rename(
                columns={"rolling_corr": L("Rolling equity–bond correlation", "滚动股债相关性")}
            )
        )
    stress_view(result)
    replay = result.get("replay")
    if replay:
        start, end, pnl = replay
        st.markdown(L("#### Historical replay at ending weights", "#### 用期末仓位重放历史行情"))
        st.write(
            L(
                f"The weakest 21-session window for a buy-and-hold replay was {start:%Y-%m-%d} to {end:%Y-%m-%d}: {pnl:+.2%}. Each historical window starts with the same final equity and bond weights.",
                f"将期末股债权重作为每个历史窗口的起点，持有 21 个共同交易日；最差窗口为 {start:%Y-%m-%d} 至 {end:%Y-%m-%d}，收益 {pnl:+.2%}。",
            )
        )
        st.caption(
            L(
                "Replay uses observed adjusted market prices. It excludes rebalancing and new costs within each 21-session window.",
                "重放使用实际复权行情，每个窗口内持仓不再平衡，未额外计入新交易成本。",
            )
        )
    with st.expander(L("Data, assumptions and research references", "数据、假设与研究参考")):
        s1, s2 = result["providers"]
        st.write(
            L(
                f"Adjusted market closes: {sym} via {s1}, {bond} via {s2}. Both legs use {result['currency']}. Distributions reflected in adjusted prices are counted once. Fund NAV, withholding tax and FX hedging are outside this calculation.",
                f"复权市场收盘价来源：{sym} 使用 {s1}，{bond} 使用 {s2}。两只 ETF 均以 {result['currency']} 计价，行情中已复权的分配计入一次。计算不含基金净值、预扣税或汇率对冲。",
            )
        )
        st.write(
            L(
                f"{result['shared_prices']} shared closes. Unmatched dates excluded: {sym} {result['equity_only_dates']}, {bond} {result['bond_only_dates']}. Both policies use {result['lookback']} past returns for the common warm-up. A rebalance uses available history; positions drift between trades. The benchmark holds the same ETFs with 60% equity and the same schedule and costs.",
                f"共有 {result['shared_prices']} 个共同收盘日，剔除单独有价日期：{sym} 为 {result['equity_only_dates']} 天，{bond} 为 {result['bond_only_dates']} 天。策略和基准统一预热 {result['lookback']} 个收益观察期，再平衡只取当时可见数据，期间权重自然漂移。基准以同一对 ETF 配置 60% 股票，调仓频率与成本相同。",
            )
        )
        st.markdown(
            L(
                f"Method references: [Goldman Sachs GS Quant · historical simulation]({GS_METHOD}) · [QuantLib · duration and convexity]({QL_METHOD}).",
                f"方法参考：[高盛 GS Quant · 历史模拟]({GS_METHOD}) · [QuantLib · 久期与凸性]({QL_METHOD})。",
            )
        )
    export = curve.copy()
    export.index.name = "date"
    st.download_button(
        L("Download portfolio history (CSV)", "下载组合历史数据（CSV）"),
        export.to_csv().encode("utf-8-sig"),
        file_name=f"{sym}_{bond}_allocation.csv",
        mime="text/csv",
        key="balanced_export",
        help=L(
            "Download daily equity curves, holdings, contributions and risk measures.",
            "下载每日净值、持仓、损益贡献与风险指标。",
        ),
    )


def render_balanced(fetch, start, end, source):
    st.subheader(L("Stock–bond allocation", "股债资产配置"))
    st.caption(
        L(
            "Pair an equity ETF with a bond ETF in one currency. Use the sidebar sample dates and compare the mix with a costed 60/40 benchmark.",
            "在同一币种下组合股票 ETF 和债券 ETF，沿用侧栏日期，与扣费后的同标的 60/40 基准比较。",
        )
    )
    currency = st.radio(
        L("Market and currency", "市场与计价币种"),
        ["USD", "CNY"],
        horizontal=True,
        format_func={"USD": L("US · USD", "美国 · 美元"), "CNY": L("China · CNY", "中国 · 人民币")}.get,
        key="sb_currency",
    )
    display_names = {symbol: fund_name(symbol) for symbol in FUNDS}
    rule_names = {key: allocation_name(key) for key in ("fixed", "inverse_vol")}
    left, right = st.columns(2)
    equity = left.selectbox(
        L("Equity ETF", "股票 ETF"),
        EQUITIES[currency],
        format_func=display_names.get,
        key="sb_equity_" + currency,
    )
    bond = right.selectbox(
        L("Bond ETF", "债券 ETF"), BONDS[currency], format_func=display_names.get, key="sb_bond_" + currency
    )
    method = st.radio(
        L("Allocation rule", "配置规则"),
        ["fixed", "inverse_vol"],
        horizontal=True,
        format_func=rule_names.get,
        key="sb_method",
    )
    if method == "fixed":
        wt = st.slider(L("Equity target (%)", "股票目标占比（%）"), 0, 100, 60, 5, key="sb_weight") / 100
        st.caption(
            L(
                f"At each rebalance: {wt:.0%} stocks, {1-wt:.0%} bonds.",
                f"每次再平衡：股票 {wt:.0%}，债券 {1-wt:.0%}。",
            )
        )
    else:
        wt = 0.6
        st.caption(
            L(
                "Allocate in inverse proportion to each ETF’s trailing daily volatility. The less volatile sleeve receives more capital.",
                "按两只 ETF 历史日收益波动率的倒数分配权重，波动较小的一侧获得更多资金。",
            )
        )
    a, b, c = st.columns(3)
    rebalance = a.selectbox(
        L("Rebalance", "再平衡频率"),
        ["monthly", "quarterly"],
        format_func={"monthly": L("Monthly", "每月"), "quarterly": L("Quarterly", "每季")}.get,
        key="sb_freq",
    )
    lookback = b.selectbox(
        L("Risk window (sessions)", "风险窗口（交易日）"), [21, 63, 126], index=1, key="sb_window"
    )
    cost = c.number_input(
        L("One-way cost (bp)", "单边成本（基点）"),
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        key="sb_cost",
        help=L(
            "Combined trading fee and slippage per amount purchased or sold.",
            "每单位买入或卖出金额承担的手续费与滑点合计。",
        ),
    )
    if st.button(
        L("Run stock–bond backtest", "运行股债回测"),
        key="run_balanced",
        type="primary",
        help=L(
            "Load adjusted ETF prices and simulate both the selected portfolio and its benchmark.",
            "获取复权行情，模拟当前组合及对应基准。",
        ),
    ):
        st.session_state.pop("balanced_result", None)
        cutoff = completed_end(end, currency)
        if start >= cutoff:
            st.warning(
                L(
                    "Choose a start date earlier than the last completed session.",
                    "起始日期须早于最近一个已收盘交易日。",
                )
            )
        elif source == "akshare" and currency == "USD":
            st.warning(
                L(
                    "For US ETFs, choose Automatic or yfinance in the sidebar.",
                    "美股 ETF 请在侧栏选择自动或 yfinance。",
                )
            )
        else:
            with st.spinner(
                L(
                    "Loading ETF histories and applying the allocation rules…",
                    "正在获取 ETF 行情并计算股债配置…",
                )
            ):
                eq, ep = prices_for(fetch, equity, start, cutoff, source)
                bd, bp = prices_for(fetch, bond, start, cutoff, source)
                missing = [name for name, data in ((equity, eq), (bond, bd)) if data.empty]
                if missing:
                    st.warning(
                        L("Adjusted prices unavailable for ", "以下 ETF 缺少复权行情：")
                        + ", ".join(missing)
                        + L(". Check dates or change the data source.", "。请检查日期或更换行情源。")
                    )
                else:
                    try:
                        result = run_balanced(
                            eq,
                            bd,
                            equity_weight=wt,
                            method=method,
                            rebalance=rebalance,
                            lookback=lookback,
                            cost_bps=cost,
                        )
                    except ValueError as err:
                        if str(err) == "short_history":
                            st.warning(
                                L(
                                    "Too few shared prices for this risk window. Extend the date range or shorten the risk window.",
                                    "共同交易日不足。请拉长日期区间或缩短风险窗口。",
                                )
                            )
                        else:
                            st.warning(
                                L(
                                    "Unable to form a portfolio from these prices and settings.",
                                    "当前行情或设置无法形成有效组合。",
                                )
                            )
                    else:
                        result.update(
                            equity=equity,
                            bond=bond,
                            currency=currency,
                            method=method,
                            target_equity=wt,
                            rebalance=rebalance,
                            cost_bps=cost,
                            lookback=lookback,
                            providers=(ep, bp),
                        )
                        prices = (
                            pd.concat([eq.rename("equity"), bd.rename("bond")], axis=1, join="inner")
                            .dropna()
                            .sort_index()
                            .loc[result["start"] : result["end"]]
                        )
                        weights = result["curve"][["equity_weight", "bond_weight"]].iloc[-1].to_numpy()
                        returns = prices.pct_change(21, fill_method=None).dropna()
                        if len(returns):
                            path = returns @ weights
                            last = path.idxmin()
                            first = prices.index[prices.index.get_loc(last) - 21]
                            result["replay"] = (first, last, float(path.loc[last]))
                        st.session_state["balanced_result"] = result
    if "balanced_result" in st.session_state:
        show_balanced(st.session_state["balanced_result"])
