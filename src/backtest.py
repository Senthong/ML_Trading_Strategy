"""Vectorized long/flat backtester with transaction costs."""
import pandas as pd


def backtest(fwd_ret: pd.Series, position: pd.Series, cost_bps=5.0, slippage_bps=2.0) -> pd.DataFrame:
    """
    fwd_ret[t]  : return from close t to close t+1
    position[t] : 0/1 decided at close t (using info up to t) and held over t -> t+1

    net_ret[t] = position[t] * fwd_ret[t] - |position[t] - position[t-1]| * (cost + slippage)
    (starting from flat, so opening the first position also pays costs)
    """
    position = position.reindex(fwd_ret.index).fillna(0).astype(float)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    fee = turnover * (cost_bps + slippage_bps) / 1e4
    gross = position * fwd_ret
    net = gross - fee
    return pd.DataFrame(
        {"position": position, "turnover": turnover, "gross_ret": gross, "net_ret": net,
         "equity": (1 + net).cumprod()}
    )
