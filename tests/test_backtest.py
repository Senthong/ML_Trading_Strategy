import pandas as pd
import pytest
from src.backtest import backtest
from src.metrics import max_drawdown


def test_costs_only_on_position_change():
    idx = pd.date_range("2020-01-01", periods=4)
    fwd = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
    pos = pd.Series([1, 1, 0, 0], index=idx)
    bt = backtest(fwd, pos, cost_bps=10, slippage_bps=0)
    # day0: enter (turnover 1), day1: hold, day2: exit (turnover 1), day3: flat
    assert bt["turnover"].tolist() == [1, 0, 1, 0]
    assert bt["net_ret"].iloc[0] == pytest.approx(0.01 - 0.001)
    assert bt["net_ret"].iloc[1] == pytest.approx(0.01)
    assert bt["net_ret"].iloc[2] == pytest.approx(-0.001)
    assert bt["net_ret"].iloc[3] == 0


def test_flat_position_earns_nothing():
    idx = pd.date_range("2020-01-01", periods=3)
    bt = backtest(pd.Series([0.05, -0.05, 0.02], index=idx), pd.Series(0, index=idx))
    assert (bt["net_ret"] == 0).all()


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
