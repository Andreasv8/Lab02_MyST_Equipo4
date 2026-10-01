import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    break_even_cost,
    cost_sensitivity,
    position_size,
    sharpe_band,
    win_rate_standard_error,
)
from src.backtest import BacktestConfig


def _trending_bars(n: int = 200) -> pd.DataFrame:
    """OHLCV sintetico con tendencias alternadas de 50 barras (semilla 42).

    En tramos alcistas el cierre queda cerca del High (CMF > 0) y en bajistas
    cerca del Low (CMF < 0), para que la regla de confluencia opere en ambas
    mitades de la serie.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    drift = np.where((np.arange(n) // 50) % 2 == 0, 1.0, -1.0)
    up = drift > 0
    close = pd.Series(200 + np.cumsum(drift + rng.normal(0, 1.5, size=n)), index=dates)
    high = close + np.where(up, rng.uniform(0.1, 0.5, n), rng.uniform(1, 3, n))
    low = close - np.where(up, rng.uniform(1, 3, n), rng.uniform(0.1, 0.5, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, size=n), index=dates)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def test_break_even_interpolates_first_zero_crossing():
    """Sharpe [1, 0.5, -0.5] en [0, 5, 10] bps: cruza entre 5 y 10 -> 5 + 0.5·5/1 = 7.5 bps."""
    assert break_even_cost([0, 5, 10], [1.0, 0.5, -0.5]) == pytest.approx(7.5)
    assert math.isnan(break_even_cost([0, 5, 10], [1.0, 0.8, 0.6]))
    assert math.isnan(break_even_cost([0, 5, 10], [-0.2, -0.4, -0.6]))


def test_sharpe_band_class_ranges():
    """<1 malo, 1-2 bueno, >2 excelente, >3 probable error."""
    assert [sharpe_band(s) for s in [-0.5, 0.99, 1.0, 1.7, 2.5, 3.4]] == [
        "malo", "malo", "bueno", "bueno", "excelente", "probable error"]
    assert sharpe_band(float("nan")) == "sin dato"


def test_position_size_by_hand():
    """Trade A: 25·100 / E_1 = 2500/10 000 = 25%; trade B: 20·200 / E_3 = 4000/8000 = 50%.

    Promedio (25% + 50%)/2 = 37.5%. Desde la barra 3 solo cuenta B (exit_date en barra 4) -> 50%.
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    equity_df = pd.DataFrame({"equity": [10_000.0, 10_000.0, 9_000.0, 8_000.0, 8_500.0]}, index=dates)
    trades = pd.DataFrame({
        "entry_bar": [1, 3], "shares": [25, 20], "raw_entry_price": [100.0, 200.0],
        "exit_date": [dates[2], dates[4]],
    })

    assert position_size(trades, equity_df) == pytest.approx(0.375)
    assert position_size(trades, equity_df, start=dates[3]) == pytest.approx(0.50)


def test_win_rate_standard_error_by_hand():
    """p = 0.5, n = 12: sqrt(0.25/12) = 0.1443."""
    assert win_rate_standard_error(0.5, 12) == pytest.approx(math.sqrt(0.25 / 12))


def test_cost_sensitivity_grid_and_costs_hurt():
    """11 costos (0-50 bps); la equity final a 50 bps no supera la de 0 bps; no modifica df."""
    df = _trending_bars()
    df_before = df.copy()
    dates = df.index
    periods = {"train": (dates[0], dates[99]), "test": (dates[100], dates[-1])}
    config = BacktestConfig(initial_cash=100_000.0)

    sensitivity = cost_sensitivity(df, config, periods=periods)

    assert sensitivity.index.tolist() == list(range(0, 55, 5))
    assert list(sensitivity.columns) == ["sharpe_train", "equity_final_train", "sharpe_test", "equity_final_test"]
    # Solo extremos: el floor del sizing puede romper la monotonia entre costos cercanos.
    for period in ["train", "test"]:
        assert sensitivity.loc[50, f"equity_final_{period}"] <= sensitivity.loc[0, f"equity_final_{period}"]
    pd.testing.assert_frame_equal(df, df_before)
