"""Evidence and bilingual sell-side-style research notes for the active backtest."""

import math
import numpy as np
import pandas as pd
from ui_text import english_label

SYSTEM_PROMPT = """Write an original institutional quantitative strategy note for an investment committee.
Use the requested language throughout. Use concise, technical research prose with an investment thesis,
benchmark-relative performance, beta and residual alpha, tracking error and information ratio, conditional
upside/downside capture, execution and trade economics, a doubled-cost stress, drawdown and tail risk,
and statistical evidence (including walk-forward only if supplied). Develop the economic mechanism and
at least one competing explanation. State what observation would weaken the thesis. Use a small table
where it makes figures easier to compare. Explain units and distinguish cumulative returns, annual rates,
ratios, percentage-point spreads and trading days. A missing metric is unavailable, never zero.
All numeric facts must come from the supplied evidence or transparent arithmetic on it. Do not invent
holdings, fundamental exposures, catalysts, analyst credentials, price targets or institutional affiliations.
Conclude with specific falsification tests. Write about the active backtest only. Use 650–950 English words
or 1100–1600 Chinese characters when enough evidence is available. Keep the note shorter for thin samples."""


def extend_evidence(ev, result, ledger):
    bench, net = result["每日收益率"], result["策略每日收益"]

    def capture(mask):
        ratio = net[mask].mean() / bench[mask].mean() if mask.any() else np.nan
        return float(ratio) if np.isfinite(ratio) else None

    scenario = (1 + net - result["成本"]).prod() - 1
    ev["路径归因"] = {
        "起日": str(result.index[0].date()),
        "止日": str(result.index[-1].date()),
        "上涨日捕获率": capture(bench > 0),
        "下跌日捕获率": capture(bench < 0),
        "上涨日数": int((bench > 0).sum()),
        "下跌日数": int((bench < 0).sum()),
        "加倍成本累计收益": float(scenario),
    }
    if not ledger.empty:
        exits = ledger.loc[ledger["离场原因"] != "尚未平仓", "离场原因"].value_counts()
        if not exits.empty:
            ev["路径归因"].update(
                {
                    "主要离场原因": str(exits.index[0]),
                    "该原因笔数": int(exits.iloc[0]),
                    "已平仓笔数": int(exits.sum()),
                }
            )
    return ev


def clean_evidence(value):
    if isinstance(value, dict):
        return {k: clean_evidence(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_note(ev, english):
    def L(en, zh):
        return en if english else zh

    def fmt(data, key, pattern=".2%"):
        val = data.get(key)
        return format(val, pattern) if isinstance(val, (float, int)) and math.isfinite(val) else "—"

    s, tr, tests = ev["绩效指标"], ev["交易统计"], ev["统计检验"]
    cfg, path = ev["策略配置"], ev["路径归因"]
    symbol = ev["标的"]
    strategy = english_label(cfg["strategy"]) if english else cfg["strategy"]
    total, bench = fmt(s, "累计收益"), fmt(s, "基准累计收益")
    active = s.get("超额收益")
    lead = active is not None and active > 0
    mdd, bench_mdd = fmt(s, "最大回撤"), fmt(s, "基准最大回撤")

    def pct_points(value):
        return format(value * 100, "+.2f") if value is not None else "—"

    reasons = {
        "趋势回调再入场": L(
            "This rule buys a short-horizon RSI recovery inside a rising slow-MA regime. Repeated breaks of that average weaken the pullback thesis.",
            "该策略在慢均线向上时捕捉短期超卖修复。价格反复跌穿慢均线会削弱回调再入场的逻辑。",
        ),
        "波动过滤突破": L(
            "The breakout sleeve admits a new price high only when normalized range risk clears its volatility filter. Fast reversals after breakouts are its principal failure mode.",
            "突破策略要求价格创新高且相对波幅满足过滤阈值。若突破后迅速回落，趋势延续假设会失效。",
        ),
    }
    thesis = reasons.get(
        cfg["strategy"],
        L(
            "Entry and exit timing vary daily exposure to the underlying. The relevant competing explanation for lower drawdown is time spent in cash; benchmark capture helps separate this effect from signal selection.",
            "择时规则改变组合对标的的逐日敞口。回撤收窄也可能来自现金仓位，需结合上下行捕获率区分降仓效应与信号选择贡献。",
        ),
    )
    horizon = f'{path["起日"]} — {path["止日"]}'
    headings = [
        L("Research view", "核心判断"),
        L("Benchmark-relative attribution", "相对收益与风险归因"),
        L("Regime dependence", "市场环境与收益非对称性"),
        L("Implementation and cost sensitivity", "交易结构与成本敏感性"),
        L("Tail risk and validation", "尾部风险与统计验证"),
        L("Falsification agenda", "后续验证与失效条件"),
    ]
    spread = pct_points(active)
    lines = [
        f"# {symbol} · {strategy}",
        f"{horizon} · " + L("Quantitative strategy research", "量化策略研究"),
        "## " + headings[0],
        L(
            f"The strategy {'outperformed' if lead else 'did not outperform'} buy-and-hold over the sample. Net cumulative return was {total} versus {bench}, a spread of {spread} percentage points. Annualized Sharpe was {fmt(s,'夏普比率','.2f')} with market exposure on {fmt(s,'在场比例')} of sessions.",
            f"样本期内策略{'跑赢' if lead else '未跑赢'}买入持有基准，扣费后累计收益 {total}，同期基准 {bench}，累计收益差 {spread} 个百分点。年化夏普为 {fmt(s,'夏普比率','.2f')}，持仓交易日占比 {fmt(s,'在场比例')}。",
        ),
        thesis,
        "## " + headings[1],
        L("| Metric | Strategy | Buy-and-hold |", "| 指标 | 策略 | 买入持有 |")
        + "\n|---|---:|---:|\n"
        + L("| Annualized return", "| 年化收益")
        + f" | {fmt(s,'年化收益')} | {fmt(s,'基准年化')} |\n"
        + L("| Maximum drawdown", "| 最大回撤")
        + f" | {mdd} | {bench_mdd} |",
        L(
            f"Estimated beta to the underlying was {fmt(s,'Beta','.2f')}; the annualized single-benchmark alpha estimate was {fmt(s,'年化Alpha')}. Tracking error of {fmt(s,'跟踪误差')} accompanied an information ratio of {fmt(s,'信息比率','.2f')}. The latter scales average active return by the variability of active returns. A single-benchmark regression leaves style and regime exposures in the residual.",
            f"相对标的的 Beta 为 {fmt(s,'Beta','.2f')}，单基准口径的年化 Alpha 估计为 {fmt(s,'年化Alpha')}。年化跟踪误差 {fmt(s,'跟踪误差')}，信息比率 {fmt(s,'信息比率','.2f')}，衡量主动收益对跟踪风险的补偿。单基准回归的残差仍可能包含风格及行情阶段暴露。",
        ),
        "## " + headings[2],
        L(
            f"On {path['上涨日数']} positive benchmark sessions, upside capture was {fmt(path,'上涨日捕获率')}. On {path['下跌日数']} negative sessions, downside capture was {fmt(path,'下跌日捕获率')}. These are ratios of conditional mean daily returns. A downside capture below 100% implies the strategy lost less on average on those benchmark-down days.",
            f"基准上涨的 {path['上涨日数']} 个交易日中，上行捕获率为 {fmt(path,'上涨日捕获率')}；基准下跌的 {path['下跌日数']} 日中，下行捕获率为 {fmt(path,'下跌日捕获率')}。两项指标按条件日均收益之比计算，下行捕获率低于 100% 表示基准下跌日的平均损失较小。",
        ),
        "## " + headings[3],
    ]
    lines.append(
        L(
            f"Annual turnover was {fmt(s,'年换手率','.2f')}×. Holding the position path fixed and charging each daily execution cost twice gives a cumulative return of {fmt(path,'加倍成本累计收益')}, compared with the base case of {total}. This stress captures fee and slippage sensitivity with unchanged fills. Market impact and available capacity require order-size and liquidity data.",
            f"年换手为 {fmt(s,'年换手率','.2f')} 倍。固定持仓路径，将每日实际计入的交易成本加倍，累计收益为 {fmt(path,'加倍成本累计收益')}，基准情景为 {total}。该压力情景衡量成交路径相同时的费用与滑点敏感性；冲击成本和策略容量仍需委托规模及流动性数据验证。",
        )
    )
    if tr:
        lines.append(
            L(
                f"The ledger contains {tr.get('交易笔数',0)} trades, with a {fmt(tr,'胜率')} win rate, a {fmt(tr,'盈亏比','.2f')} payoff ratio and {fmt(tr,'单笔期望')} mean return per trade.",
                f"交易台账共 {tr.get('交易笔数',0)} 笔，胜率 {fmt(tr,'胜率')}，盈亏比 {fmt(tr,'盈亏比','.2f')}，单笔期望收益 {fmt(tr,'单笔期望')}。",
            )
        )
    if path.get("已平仓笔数"):
        reason = path["主要离场原因"]
        lines.append(
            L(
                f"Among completed trades, the most frequent exit was {english_label(reason)} ({path['该原因笔数']} of {path['已平仓笔数']}).",
                f"已平仓交易中最常见的退出原因为{reason}，共 {path['该原因笔数']} / {path['已平仓笔数']} 笔。",
            )
        )
    lines += [
        "## " + headings[4],
        L(
            f"Daily historical 95% expected shortfall was {fmt(s,'日CVaR95')}. The longest period below the prior equity high was {fmt(s,'最长回撤天数','.0f')} sessions. Tail loss describes loss severity; recovery time describes capital lock-up.",
            f"历史日度 95% 条件尾部收益为 {fmt(s,'日CVaR95')}，净值低于前高的最长持续期为 {fmt(s,'最长回撤天数','.0f')} 个交易日。前者刻画尾部损失强度，后者反映净值修复所需时间。",
        ),
    ]
    if "PSR" in tests:
        lines.append(
            L(
                f"PSR {fmt(tests,'PSR')} and DSR {fmt(tests,'DSR')} incorporate different assumptions about selection. Reality Check p = {fmt(tests,'RealityCheck_p','.3f')}. The bootstrap annual-return interval (5th–95th percentiles) spans {fmt(tests,'Bootstrap_CAGR_5%')} to {fmt(tests,'Bootstrap_CAGR_95%')}. DSR depends on the reported trial count ({tests.get('参数试验次数','—')}).",
                f"概率夏普 PSR 为 {fmt(tests,'PSR')}，经多重试验调整的 DSR 为 {fmt(tests,'DSR')}，Reality Check 的 p 值为 {fmt(tests,'RealityCheck_p','.3f')}。区块重抽得到的年化收益第 5 至 95 百分位为 {fmt(tests,'Bootstrap_CAGR_5%')} 至 {fmt(tests,'Bootstrap_CAGR_95%')}。DSR 的解释依赖所填报的试验次数（{tests.get('参数试验次数','—')}）。",
            )
        )
    else:
        lines.append(
            L(
                "Bootstrap and multiple-testing diagnostics have not been run for this backtest. Statistical significance is undetermined.",
                "本次回测尚未运行区块重抽和多重试验校正，统计显著性待检验。",
            )
        )
    if "WFE" in tests:
        lines.append(
            L(
                f"Matched walk-forward run: OOS annual return {fmt(tests,'OOS年化')}, OOS Sharpe {fmt(tests,'OOS夏普','.2f')}, efficiency {fmt(tests,'WFE','.2f')}.",
                f"同标的、日期、数据源和策略设定的滚动检验中，样本外年化收益 {fmt(tests,'OOS年化')}，样本外夏普 {fmt(tests,'OOS夏普','.2f')}，滚动效率 {fmt(tests,'WFE','.2f')}。",
            )
        )
    else:
        lines.append(
            L(
                "A matched walk-forward run is unavailable for this sample.",
                "当前样本尚无匹配的滚动验证结果。",
            )
        )
    lines += [
        "## " + headings[5],
        L(
            "Freeze the trading rule and execution assumptions before the next evaluation window. Compare upside and downside capture on that window and rerun the doubled-cost stress. A reversal of the active-return spread, concentrated losses or persistent deterioration against the benchmark would weaken the case for this specification.",
            "在下一观察窗口开始前固定信号与成交假设，复核样本外上下行捕获率，并重做成本加倍压力测试。若主动收益差转负、亏损集中在少数交易，或相对基准持续恶化，应重新评估当前设定。",
        ),
        L(
            "*Generated locally from the active backtest evidence.*",
            "*本报告由本次回测的结构化证据在本地生成。*",
        ),
    ]
    return "\n\n".join(lines)
