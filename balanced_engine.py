"""Two-sleeve, long-only backtest on common daily adjusted closes (unit starting NAV)."""

import numpy as np
import pandas as pd


def _daily_closes(series):
    if not isinstance(series, pd.Series) or pd.api.types.is_numeric_dtype(series.index.dtype):
        raise ValueError("invalid_prices")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(series.index, errors="raise"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid_prices") from exc
    # The label is the exchange's trading date, not a UTC calendar date.
    dates = dates.tz_localize(None).normalize()
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    prices = pd.Series(values, index=dates)
    prices = prices[prices.index.notna() & np.isfinite(prices) & (prices > 0)]
    if prices.index.has_duplicates:
        raise ValueError("invalid_prices")
    return prices.sort_index()


def _rebalance(holdings, nav, stock_weight, fee_rate, opening):
    """Charge the fee on actual traded notional, including its own financing."""
    if opening:
        # The initial purchase and its fee must fit in the starting cash.
        traded = nav / (1 + fee_rate)
        fee = traded * fee_rate
    else:
        change = stock_weight * nav - holdings[0]
        # The two sleeves trade in opposite directions. Post-trade targets are
        # fractions of NAV *after* the fee. Solving fee = rate * gross traded
        # yields this expression for either the buy or sell direction.
        traded = 2 * abs(change) / (1 + fee_rate * np.sign(change) * (2 * stock_weight - 1))
        fee = traded * fee_rate
    after_fee = nav - fee
    return np.array([stock_weight * after_fee, (1 - stock_weight) * after_fee]), fee, traded


def _simulate(returns, trade_days, targets, fee_rate):
    count = len(returns)
    nav = np.empty(count)
    weights = np.empty(count)
    contributions = np.zeros((count, 3))  # stock, bond, trading fee
    turnover = 0.0
    holdings = np.zeros(2)
    for i in range(count):
        if i:
            contributions[i, :2] = holdings * returns[i]
            holdings += contributions[i, :2]
        pre_trade = holdings.sum() if i else 1.0
        if trade_days[i]:
            holdings, fee, traded = _rebalance(holdings, pre_trade, targets[i], fee_rate, i == 0)
            contributions[i, 2] = -fee
            turnover += traded / pre_trade
        nav[i] = holdings.sum()
        weights[i] = holdings[0] / nav[i]
    peaks = np.maximum.accumulate(np.maximum(1.0, nav))
    return nav, weights, contributions, turnover, nav / peaks - 1


def _metrics(nav, drawdown, pnl, turnover, dates):
    daily = nav[1:] / nav[:-1] - 1
    daily[0] = nav[1] - 1  # Fold the entry fee into the first holding interval.
    volatility = daily.std(ddof=1) * np.sqrt(252)
    years = (dates[-1] - dates[0]).days / 365.25
    return {
        "total_return": nav[-1] - 1,
        "cagr": nav[-1] ** (1 / years) - 1,
        "volatility": volatility,
        "max_drawdown": drawdown.min(),
        "cost": -pnl[:, 2].sum(),
        "turnover": turnover,
        "sharpe": daily.mean() * 252 / volatility if volatility > 1e-12 else np.nan,
    }


def run_balanced(
    equity_close, bond_close, *, equity_weight=0.60, method="fixed",
    rebalance="monthly", lookback=63, cost_bps=5.0,
):
    """Compare selected allocation and 60/40 on identical dates and costs.

    Execution is at the first observed close of each month or quarter; holdings
    drift in between. A trailing inverse-volatility target uses only returns
    known at the previous close. When both volatilities are zero it uses 50/50.
    Warmup uses lookback returns for every rule, including the benchmark.
    CAGR uses elapsed calendar time; volatility and zero-rate Sharpe use 252
    sessions/year. Cost is in initial-capital units; turnover sums each trade's
    gross two-sided notional divided by its pre-trade NAV, including entry.
    """
    if (method not in ("fixed", "inverse_vol") or rebalance not in ("monthly", "quarterly")
            or not isinstance(lookback, (int, np.integer)) or lookback < 2
            or not 0 <= equity_weight <= 1 or not 0 <= cost_bps < 10_000):
        raise ValueError("invalid_settings")
    stock, bond = _daily_closes(equity_close), _daily_closes(bond_close)
    prices = pd.concat([stock.rename("equity"), bond.rename("bond")], axis=1, join="inner")
    prices = prices.sort_index()
    if len(prices) < lookback + 4:
        raise ValueError("short_history")
    returns = prices.pct_change(fill_method=None)
    first = lookback + 1
    dates = prices.index[first:]
    periods = dates.to_period("M" if rebalance == "monthly" else "Q")
    trade_days = np.r_[True, periods[1:] != periods[:-1]]
    targets = np.full(len(dates), equity_weight, dtype=float)
    if method == "inverse_vol":
        vols = returns.rolling(lookback).std().shift(1).iloc[first:]
        vols = vols.mask(vols < 1e-12, 0.0)
        total_vol = vols["equity"] + vols["bond"]
        targets = (vols["bond"] / total_vol.where(total_vol > 0)).fillna(0.5).to_numpy()
    daily = returns.iloc[first:].to_numpy()
    fee_rate = cost_bps / 10_000
    nav, stock_weight, pnl, turnover, dd = _simulate(daily, trade_days, targets, fee_rate)
    baseline, _, base_pnl, base_turnover, base_dd = _simulate(
        daily, trade_days, np.full(len(dates), 0.60), fee_rate,
    )
    corr = returns["equity"].rolling(lookback).corr(returns["bond"]).iloc[first:]
    curve = pd.DataFrame({
        "strategy_nav": nav, "benchmark_nav": baseline, "equity_weight": stock_weight,
        "bond_weight": 1 - stock_weight, "equity_pnl": pnl[:, 0], "bond_pnl": pnl[:, 1],
        "cost_pnl": pnl[:, 2], "rolling_corr": corr.clip(-1.0, 1.0),
        "strategy_drawdown": dd, "benchmark_drawdown": base_dd, "rebalance": trade_days,
    }, index=dates)
    metrics = pd.DataFrame({
        "strategy": _metrics(nav, dd, pnl, turnover, dates),
        "benchmark": _metrics(baseline, base_dd, base_pnl, base_turnover, dates),
    }).T
    return {
        "curve": curve, "metrics": metrics,
        "attribution": pd.Series(pnl.sum(axis=0), index=["equity", "bond", "cost"]),
        "start": dates[0], "end": dates[-1], "sessions": len(dates) - 1,
        "shared_prices": len(prices), "equity_only_dates": len(stock) - len(prices),
        "bond_only_dates": len(bond) - len(prices),
    }
