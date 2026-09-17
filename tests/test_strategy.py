import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy import Position, compute_sizing, resolve_exit, run_backtest


def test_sl_tp_same_bar_resolves_as_stop():
    """Si SL y TP caen en la misma barra, debe ejecutarse el stop-loss (conservador)."""
    position = Position(
        side="long",
        shares=10,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        entry_bar=0,
    )

    closed, reason, exit_price = resolve_exit(
        position, bar_high=112.0, bar_low=90.0, bar_close=105.0, bar_index=1,
    )

    assert closed
    assert reason == "stop_loss"
    assert exit_price == position.stop_loss


def test_sizing_matches_risk_budget():
    """El numero de acciones debe corresponder exactamente al riesgo presupuestado (rho * capital)."""
    capital = 100_000.0
    atr_value = 10.0
    rho = 0.01

    shares = compute_sizing(capital, atr_value, rho)
    risk_in_dollars = shares * 2 * atr_value

    assert shares == 50
    assert risk_in_dollars == rho * capital


def test_no_simultaneous_positions():
    """La maquina de estados nunca debe mantener dos posiciones abiertas al mismo tiempo."""
    rng = np.random.default_rng(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close = pd.Series(100 + np.cumsum(rng.normal(0, 2.0, size=n)), index=dates)
    high = close + rng.uniform(0.5, 2.5, size=n)
    low = close - rng.uniform(0.5, 2.5, size=n)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, size=n), index=dates)

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    trades = run_backtest(df, capital=100_000.0)

    assert len(trades) > 0

    for previous, current in zip(trades, trades[1:]):
        assert previous["exit_bar"] <= current["entry_bar"]
