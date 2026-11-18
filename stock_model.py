"""Price-based cross-sectional watchlist with chronological, non-overlapping evaluation."""

import numpy as np
import pandas as pd

FEATURES = ("ret5", "ret21", "ret63", "trend", "vol21")
MIN_TRAIN = 160
MAX_TRAIN = 756
RIDGE = 0.10


def feature_matrix(close):
    daily = close.pct_change(fill_method=None)
    return pd.DataFrame(
        {
            "ret5": close.pct_change(5, fill_method=None),
            "ret21": close.pct_change(21, fill_method=None),
            "ret63": close.pct_change(63, fill_method=None),
            "trend": close / close.rolling(63).mean() - 1,
            "vol21": daily.rolling(21).std() * np.sqrt(252),
        }
    )


def fit_at(x, target, origin, horizon):
    # A label for day t becomes observable at the close of day t+horizon.
    end = origin - horizon + 1
    start = max(0, end - MAX_TRAIN)
    train, y = x[start:end], target[start:end]
    ok = np.isfinite(train).all(axis=1) & np.isfinite(y)
    train, y = train[ok], y[ok]
    if len(y) < MIN_TRAIN or not np.isfinite(x[origin]).all():
        return None
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-9] = 1
    z = (train - mean) / std
    base = float(y.mean())
    coef = np.linalg.solve(z.T @ z + len(y) * RIDGE * np.eye(z.shape[1]), z.T @ (y - base))
    impact = ((x[origin] - mean) / std) * coef
    return float(base + impact.sum()), base, impact, len(y)


def rank_universe(prices, horizon):
    """Use the last available common close; stale/short series never enter the ranking."""
    clean = {}
    for symbol, series in prices.items():
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0].sort_index()
        if not s.empty:
            clean[symbol] = s[~s.index.duplicated(keep="last")]
    if not clean:
        raise ValueError("no_prices")
    asof = max(s.index[-1] for s in clean.values())
    minimum = 63 + MIN_TRAIN + 7 * horizon
    eligible = {k: v for k, v in clean.items() if v.index[-1] == asof and len(v) >= minimum}
    if len(eligible) < 2:
        raise ValueError("short_history")
    panel = pd.concat(eligible, axis=1, join="inner").dropna().sort_index()
    if len(panel) < minimum:
        raise ValueError("short_history")
    last = len(panel) - 1
    origins = list(range(last - horizon, 63 + MIN_TRAIN + horizon - 2, -horizon))[:24][::-1]
    rows, checks = [], []
    for symbol, close in panel.items():
        x = feature_matrix(close).to_numpy(dtype=float)
        y = (close.shift(-horizon) / close - 1).to_numpy(dtype=float)
        for i in origins:
            pred, base, _, _ = fit_at(x, y, i, horizon)
            checks.append((panel.index[i], symbol, pred, base, y[i]))
        pred, base, impact, size = fit_at(x, y, last, horizon)
        driver = int(np.argmax(abs(impact)))
        rows.append((symbol, pred, base, FEATURES[driver], impact[driver], size))
    scores = pd.DataFrame(
        rows, columns=["symbol", "estimate", "baseline", "driver", "impact", "train_n"]
    ).set_index("symbol")
    validation = pd.DataFrame(checks, columns=["date", "symbol", "forecast", "baseline", "realized"])
    validation["error"] = validation.realized - validation.forecast
    validation["absolute_error"] = validation.error.abs()
    validation["baseline_error"] = (validation.realized - validation.baseline).abs()
    validation["hit"] = np.sign(validation.forecast) == np.sign(validation.realized)
    by_symbol = validation.groupby("symbol")
    scores["mae"] = by_symbol.absolute_error.mean()
    scores["low"] = scores.estimate + by_symbol.error.quantile(0.10)
    scores["high"] = scores.estimate + by_symbol.error.quantile(0.90)
    ics = []
    for _, date_slice in validation.groupby("date"):
        if len(date_slice) >= 3 and date_slice.forecast.nunique() > 1 and date_slice.realized.nunique() > 1:
            ics.append(date_slice.forecast.rank().corr(date_slice.realized.rank()))
    return dict(
        scores=scores.sort_values("estimate", ascending=False, kind="stable"),
        asof=asof,
        horizon=horizon,
        origins=len(origins),
        validation=validation,
        mae=validation.absolute_error.mean(),
        baseline_mae=validation.baseline_error.mean(),
        hit=validation.hit.mean(),
        rank_ic=np.mean(ics) if ics else np.nan,
        excluded=sorted(set(prices) - set(eligible)),
    )


def csv_closes(file, start, end):
    df = pd.read_csv(file, dtype={"symbol": str})
    if not {"date", "symbol", "close"} <= set(df):
        raise ValueError("csv_schema")
    df["date"] = pd.to_datetime(df.date, errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df.close, errors="coerce")
    df["symbol"] = df.symbol.str.strip().str.upper()
    if (
        df[["date", "symbol", "close"]].isna().any().any()
        or (df.close <= 0).any()
        or (df.symbol == "").any()
        or df.duplicated(["symbol", "date"]).any()
    ):
        raise ValueError("csv_schema")
    df = df[df.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    return {name: g.set_index("date").close for name, g in df.groupby("symbol", sort=False)}
