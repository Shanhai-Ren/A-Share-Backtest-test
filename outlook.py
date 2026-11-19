import re
import numpy as np
import pandas as pd
import streamlit as st
from stock_model import csv_closes, leader_vs_pool, rank_universe
from ui_text import bilingual as L

PRESETS = {
    "us": "AAPL, MSFT, NVDA, JPM, XOM, JNJ",
    "cn": "600519, 000001, 600036, 000333, 601318, 300750",
}
FEATURE_NAMES = {
    "ret5": ("5-day return", "5日收益"),
    "ret21": ("21-day return", "21日收益"),
    "ret63": ("63-day return", "63日收益"),
    "trend": ("63-day trend", "63日均线偏离"),
    "vol21": ("21-day volatility", "21日波动率"),
}


def render_outlook(fetch, normalize, start, end, source):
    st.subheader(L("Stock outlook", "个股前瞻"))
    st.caption(
        L(
            "Rank a watchlist by estimated forward return. Each stock has its own ridge model, fitted to adjusted closing prices.",
            "用复权收盘价为每只股票拟合独立的岭回归模型，按未来区间的估计收益率排列观察名单。",
        )
    )
    left, right = st.columns(2)
    market = left.radio(
        L("Price universe", "行情范围"),
        ["us", "cn", "csv"],
        horizontal=True,
        format_func={"us": L("US stocks", "美股"), "cn": L("China A-shares", "沪深A股"), "csv": "CSV"}.get,
        key="outlook_market",
    )
    h = right.selectbox(
        L("Forecast horizon (trading sessions)", "预测期限（交易日）"),
        [5, 20, 60],
        index=1,
        key="outlook_horizon",
    )
    upload = None
    symbols = []
    if market == "csv":
        upload = st.file_uploader(
            L("Adjusted daily closes (CSV)", "每日复权收盘价（CSV）"),
            type="csv",
            key="outlook_csv",
            help=L(
                "Use columns date, symbol, close. Include at least two stocks trading on the same calendar.",
                "列名为 date、symbol、close。请提供至少两只交易日历相同的股票。",
            ),
        )
    else:
        pool = st.text_input(
            L("Stocks to compare", "待比较股票"), PRESETS[market], key="outlook_symbols_" + market
        )
        symbols = list(dict.fromkeys(s.strip().upper() for s in re.split("[,，]", pool) if s.strip()))
    st.caption(
        L(
            f"The sample dates in the sidebar apply. Allow at least {223+7*h} shared trading sessions for this horizon.",
            f"采用侧栏的样本起止日期。当前期限至少需要 {223+7*h} 个共同交易日。",
        )
    )
    if st.button(
        L("Build outlook", "生成预测排名"),
        key="run_outlook",
        type="primary",
        help=L(
            "Fit the models, evaluate past forecasts and rank the current watchlist.",
            "拟合模型，检验历史预测，再对当前股票池排序。",
        ),
    ):
        st.session_state.pop("stock_outlook", None)
        try:
            if start > end:
                raise ValueError("dates")
            with st.spinner(
                L("Loading prices and evaluating historical forecasts…", "获取行情并检验历史预测…")
            ):
                if market == "csv":
                    if upload is None:
                        raise ValueError("upload")
                    upload.seek(0)
                    prices = csv_closes(upload, start, end)
                    origin = "CSV"
                else:
                    if not 2 <= len(symbols) <= 25:
                        raise ValueError("universe")
                    pat = r"[A-Z][A-Z0-9.-]{0,10}" if market == "us" else r"[0-9]{6}(?:\.(?:SS|SZ))?"
                    if not all(re.fullmatch(pat, s) for s in symbols):
                        raise ValueError("symbols")
                    zone, cutoff = (
                        ("America/New_York", (16, 15)) if market == "us" else ("Asia/Shanghai", (15, 15))
                    )
                    now = pd.Timestamp.now(tz=zone)
                    last_close = (now - pd.Timedelta(days=int((now.hour, now.minute) < cutoff))).date()
                    price_end = min(end, last_close)
                    prices = {}
                    for name in symbols:
                        code = normalize(name, "A股" if market == "cn" else "美股")
                        frame = fetch(code, start, price_end, source, adjusted=True)
                        close = frame["收盘"] if not frame.empty else pd.Series(dtype=float)
                        prices[code] = close.loc[: pd.Timestamp(price_end)]
                    origin = source
                result = rank_universe(prices, h)
                result.update(source=origin, requested_start=str(start), requested_end=str(end))
                st.session_state["stock_outlook"] = result
        except (ValueError, pd.errors.ParserError, UnicodeError) as err:
            messages = {
                "dates": ("Choose an end date on or after the start date.", "结束日期须晚于或等于开始日期。"),
                "universe": ("Enter between 2 and 25 distinct symbols.", "请输入 2 至 25 个不同的代码。"),
                "symbols": ("Check the symbol format for this market.", "请检查代码格式是否与市场一致。"),
                "upload": ("Choose a CSV file first.", "请先选择 CSV 文件。"),
                "no_prices": (
                    "No prices returned. Check the dates, switch provider or use the CSV option.",
                    "未取得行情。请检查日期、更换行情源或使用 CSV。",
                ),
                "short_history": (
                    "At least two stocks need enough shared history ending on the same date. Extend the period or revise the universe.",
                    "至少两只股票须有足量的共同历史行情，并有相同的最新日期。请拉长区间或调整股票池。",
                ),
            }
            st.warning(
                L(
                    *messages.get(
                        str(err),
                        (
                            "Check CSV columns date, symbol, close. Use unique dates per stock and positive prices.",
                            "请核对 CSV 的 date、symbol、close 三列，每只股票每天一条记录，价格为正。",
                        ),
                    )
                )
            )
    result = st.session_state.get("stock_outlook")
    if result is None:
        return
    scores = result["scores"].copy()
    date = result["asof"].strftime("%Y-%m-%d")
    h = result["horizon"]
    st.markdown(L(f"#### {h}-session outlook · close of {date}", f"#### 未来 {h} 个交易日 · 行情截至 {date}"))
    st.caption(
        L("Last completed run", "最近一次运行")
        + " · "
        + str(result["requested_start"])
        + " – "
        + str(result["requested_end"])
        + " · "
        + str(result["source"]).replace("自动", L("Automatic", "自动"))
    )
    if result["excluded"]:
        st.caption(
            L("Excluded for missing, short or stale prices: ", "因缺失、样本不足或最新日期不一致而排除：")
            + ", ".join(result["excluded"])
        )
    best_estimate = scores.estimate.max()
    if best_estimate > 0:
        leaders = ", ".join(scores.index[scores.estimate.eq(best_estimate)])
        st.write(
            L(
                f"The highest estimated return in this watchlist is {best_estimate:+.1%} ({leaders}).",
                f"本次股票池最高的估计区间收益为 {best_estimate:+.1%}（{leaders}）。",
            )
        )
    else:
        st.write(
            L(
                "All estimates are flat or negative at this close.",
                "截至该收盘日，股票池内所有估计收益均为零或负值。",
            )
        )
    scores["driver"] = scores.driver.map(lambda key: L(*FEATURE_NAMES[key]))
    labels = dict(
        symbol=L("Stock", "股票"),
        estimate=L("Est. return", "估计收益"),
        low=L("Lower band", "区间下沿"),
        high=L("Upper band", "区间上沿"),
        mae=L("Historical MAE", "历史平均绝对误差"),
        driver=L("Largest driver", "首要驱动"),
        impact=L("Contribution vs. mean", "相对均值贡献"),
        baseline=L("Training mean", "训练均值"),
        train_n=L("Training rows", "训练样本数"),
    )
    scores.index.name = labels["symbol"]
    cols = ["estimate", "low", "high", "mae", "driver", "impact", "baseline", "train_n"]
    view = scores[cols].rename(columns=labels)
    fmts = {labels[k]: "{:.2%}" for k in ("estimate", "low", "high", "mae", "impact", "baseline")}
    st.dataframe(view.style.format(fmts), width="stretch")
    st.markdown(L("#### Walk-forward evidence", "#### 滚动样本外检验"))
    c = st.columns(4)
    c[0].metric(L("Model MAE", "模型平均绝对误差"), f'{result["mae"]:.2%}')
    c[1].metric(L("Historical-mean MAE", "历史均值基线误差"), f'{result["baseline_mae"]:.2%}')
    c[2].metric(L("Directional accuracy", "方向命中率"), f'{result["hit"]:.1%}')
    ic = result["rank_ic"]
    c[3].metric(L("Mean rank IC", "平均秩相关系数"), f"{ic:.2f}" if np.isfinite(ic) else "—")
    if result["mae"] >= result["baseline_mae"]:
        st.info(
            L(
                "The historical mean had equal or lower error. Treat the current ranking as exploratory.",
                "历史均值基线的误差更低或持平。本次排序仅适合作为探索性线索。",
            )
        )
    n = result["origins"]
    st.caption(
        L(
            f"Evaluated on {n} non-overlapping forecast dates. Bands add each stock’s 10th and 90th percentile historical errors to the estimate. Realized returns can fall outside them.",
            f"检验涵盖 {n} 个不重叠预测期。上下沿为估计值加上该股票历史误差的第 10 和第 90 百分位；实际收益可能超出区间。",
        )
    )
    periods = leader_vs_pool(result["validation"])
    if not periods.empty:
        st.markdown(L("#### Did the forecast leaders beat the watchlist?", "#### 预测排名首位的股票跑赢了股票池吗？"))
        mean_spread = periods.spread.mean() * 100
        wins = int(periods.spread.gt(0).sum())
        left, right = st.columns(2)
        left.metric(
            L("Mean excess return", "平均超额收益"),
            L(f"{mean_spread:+.2f} pp", f"{mean_spread:+.2f} 个百分点"),
        )
        right.metric(
            L("Periods ahead of the pool", "跑赢股票池的期数占比"),
            f"{wins / len(periods):.0%} ({wins}/{len(periods)})",
        )
        st.caption(
            L(
                f"Across {len(periods)} non-overlapping periods, we compare the next {h}-session return of the forecast leaders with the equal-weight watchlist. Tied leaders share equal weight. Returns are gross of trading costs.",
                f"共 {len(periods)} 个不重叠预测期。每期比较预测首位股票与等权股票池随后 {h} 个交易日的实际收益。并列首位时等权分配，收益均未扣交易成本。",
            )
        )
    with st.expander(L("Model and validation details", "模型与验证明细")):
        st.write(
            L(
                "At each close the model uses 5-, 21- and 63-session returns, distance to the 63-session moving average and 21-session volatility. Each fit standardizes features using up to 756 training rows with fully observed outcomes. Rank IC compares forecast and realized ranks on dates with at least three stocks. Universe selection can introduce survivorship bias. Estimates exclude execution costs, news and fundamentals.",
                "特征包括 5、21、63 日收益，价格相对 63 日均线的偏离，以及 21 日波动率。每次拟合仅使用届时已实现标签的最近至多 756 条记录进行标准化和训练。秩相关系数比较至少三只股票在同一期的预测与实际排序。股票池选择可能带来幸存者偏差。预测未计交易成本、新闻或公司基本面。",
            )
        )
        st.markdown(
            L(
                "The comparison follows Qlib’s approach of evaluating the top forecasts against the same-date universe. Each period has equal weight in the mean. [Qlib evaluation methods](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py)",
                "参照 Qlib 比较高排名股票与同期股票池实际收益的方法，各预测期等权计算平均超额收益。[Qlib 评估方法](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py)",
            )
        )
        if not periods.empty:
            history = periods.copy()
            history["date"] = history.date.dt.date
            history["spread"] *= 100
            history.columns = [
                L("Forecast date", "预测日"),
                L("Forecast leaders", "预测首位股票"),
                L("Leaders’ return", "首位组实际收益"),
                L("Watchlist return", "股票池实际收益"),
                L("Excess (pp)", "超额（百分点）"),
            ]
            formats = {history.columns[2]: "{:.2%}", history.columns[3]: "{:.2%}", history.columns[4]: "{:+.2f}"}
            st.dataframe(history.style.format(formats), hide_index=True, width="stretch")
        v = result["validation"][["date", "symbol", "forecast", "baseline", "realized"]].copy()
        v.columns = [
            L("Origin", "预测日"),
            labels["symbol"],
            labels["estimate"],
            labels["baseline"],
            L("Realized", "实际收益"),
        ]
        st.dataframe(v.style.format({k: "{:.2%}" for k in v.columns[2:]}), hide_index=True, width="stretch")
