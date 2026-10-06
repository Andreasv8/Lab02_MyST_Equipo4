import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BacktestConfig
from src.data import load_train
from src.metrics import (
    PERIODS_PER_YEAR,
    break_even_cost,
    buy_and_hold_equity,
    cagr,
    calmar_ratio,
    cost_sensitivity,
    drawdown_series,
    exposure,
    max_drawdown,
    returns,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    trade_stats,
    turnover_stats,
    win_rate_stats,
)
from src.signals import THETA0

PPY = PERIODS_PER_YEAR
SQRT_PPY = math.sqrt(PPY)

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _trades(rows: list[dict]) -> pd.DataFrame:
    """Trades minimos con las columnas que usan las metricas."""
    defaults = {"side": "long", "shares": 1, "raw_entry_price": 100.0, "raw_exit_price": 100.0,
                "entry_atr": 1.0, "borrow_fee": 0.0, "pnl": 0.0,
                "entry_date": pd.Timestamp("2024-01-01"), "exit_date": pd.Timestamp("2024-01-02")}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_returns_drawdown_cagr_calmar_by_hand():
    """equity [100, 110, 99, 121]: DD = 99/110 - 1 = -10%.

    Con periods = 3 las 3 barras son un año: CAGR = 1.21 - 1 = 21%, Calmar = 0.21/0.10.
    """
    equity = pd.Series([100.0, 110.0, 99.0, 121.0], index=_dates(4))

    assert returns(equity).tolist() == pytest.approx([0.1, -0.1, 22 / 99])
    assert drawdown_series(equity).tolist() == pytest.approx([0.0, 0.0, -0.1, 0.0])
    assert max_drawdown(equity) == pytest.approx(-0.10)

    assert cagr(equity, periods=3) == pytest.approx(0.21)
    assert calmar_ratio(equity, periods=3) == pytest.approx(2.1)


def test_sharpe_and_sortino_by_hand():
    """r = [0.02, -0.01, 0.03, -0.02], media 0.005.

    Sharpe: desviaciones [0.015, -0.015, 0.025, -0.025], Σ² = 0.0017, std = sqrt(0.0017/3).
    Sortino: min(r, 0)² = [0, 0.0001, 0, 0.0004], downside = sqrt(0.0005/4).
    """
    r = pd.Series([0.02, -0.01, 0.03, -0.02])

    assert sharpe_ratio(r) == pytest.approx(0.005 / math.sqrt(0.0017 / 3) * SQRT_PPY)
    assert sortino_ratio(r) == pytest.approx(0.005 / math.sqrt(0.0005 / 4) * SQRT_PPY)


def test_trade_stats_profit_factor_by_hand():
    """pnl [50, -20, 30]: profit factor = 80/20 = 4, P&L promedio = 20, win rate = 2/3."""
    trades = _trades([{"pnl": 50.0}, {"pnl": -20.0}, {"pnl": 30.0}])

    stats = trade_stats(trades)
    assert stats["n_trades"] == 3
    assert stats["avg_pnl"] == pytest.approx(20.0)
    assert stats["profit_factor"] == pytest.approx(4.0)
    assert win_rate_stats(trades, 2, 3, 0.001)["win_rate"] == pytest.approx(2 / 3)


def test_break_even_win_rate_with_costs_by_hand():
    """Q = 25, ATR = 2, crudos 100 -> 106, c = 0.001, sin borrow.

    k = 0.001·25·(100 + 106) / (2·2·25) = 5.15/100 = 0.0515
    p* = 1/(1 + 3/2) = 0.4;  p*_costos = 1.0515/2.5
    """
    trades = _trades([{"shares": 25, "raw_entry_price": 100.0, "raw_exit_price": 106.0,
                       "entry_atr": 2.0, "pnl": 100.0}])

    stats = win_rate_stats(trades, sl_mult=2, tp_mult=3, cost_rate=0.001)
    assert stats["p_star"] == pytest.approx(0.4)
    assert stats["k_mean"] == pytest.approx(0.0515)
    assert stats["p_star_cost"] == pytest.approx(1.0515 / 2.5)


def test_turnover_and_cost_hurdle_by_hand():
    """Equity 1000 constante en 5 barras (N = 4, años = 4/PPY); 5 acciones 100 -> 110, borrow 0.5.

    turnover = (500 + 550)/1000/(4/PPY);  hurdle = turnover·0.001 + 0.5/1000/(4/PPY)
    """
    dates = _dates(5)
    equity = pd.Series(1000.0, index=dates)
    trades = _trades([{"shares": 5, "raw_entry_price": 100.0, "raw_exit_price": 110.0,
                       "borrow_fee": 0.5, "entry_date": dates[1], "exit_date": dates[3]}])

    stats = turnover_stats(equity, trades, cost_rate=0.001)
    expected_turnover = 1050 / 1000 / (4 / PPY)
    assert stats["turnover_annual"] == pytest.approx(expected_turnover)
    assert stats["cost_hurdle_annual"] == pytest.approx(expected_turnover * 0.001 + 0.5 / 1000 / (4 / PPY))
    assert stats["trades_per_year"] == pytest.approx(1 / (4 / PPY))


def test_exposure_by_hand():
    """shares [0, 25, 0, -25, 0] -> 2 de 5 barras con posicion = 0.4."""
    equity_df = pd.DataFrame({"shares": [0, 25, 0, -25, 0]}, index=_dates(5))

    assert exposure(equity_df) == pytest.approx(0.4)


def test_buy_and_hold_by_hand():
    """Open_0 = 100, c = 0.001: Q = floor(10000/100.1) = 99, cash = 10000 - 9909.90 = 90.10.

    equity = 90.10 + 99·[100, 110, 90] = [9990.10, 10980.10, 9000.10]
    """
    df = pd.DataFrame({"Open": [100.0, 105.0, 95.0], "Close": [100.0, 110.0, 90.0]}, index=_dates(3))

    bh = buy_and_hold_equity(df, initial_cash=10_000, cost_rate=0.001)

    assert bh["shares"].tolist() == pytest.approx([10_000 / 100.1] * 3)
    assert bh["cash"].iloc[0] == pytest.approx(0.0)
    assert bh["equity"].tolist() == pytest.approx([9990.01, 10989.01, 8991.01], abs=0.01)


def test_summarize_sub_period_rebases_and_filters_trades():
    """Sub-periodo barras 3-5 de equity [100, 120, 90, 100, 110, 99] -> [100, 110, 99].

    Rebasado a 10 000: final = 9900; DD = 99/110 - 1 = -10%; CAGR = 0.99^(PPY/2) - 1.
    Trade A (entra barra 0, sale barra 2) es de otro periodo. Trade B (entra barra 1,
    sale barra 4) cruza el corte: cuenta en este periodo y solo su pata de salida
    entra en el turnover: 2·105/media(100, 110, 99)/(2/PPY).
    """
    dates = _dates(6)
    equity_df = pd.DataFrame({"equity": [100.0, 120.0, 90.0, 100.0, 110.0, 99.0],
                              "shares": [0, 2, 0, 2, 2, 0]}, index=dates)
    trades = _trades([
        {"entry_date": dates[0], "exit_date": dates[2], "pnl": -10.0, "shares": 2},
        {"entry_date": dates[1], "exit_date": dates[4], "pnl": 8.0, "shares": 2,
         "raw_entry_price": 100.0, "raw_exit_price": 105.0},
    ])
    config = BacktestConfig(initial_cash=10_000, commission_rate=0.001, slippage_rate=0.0)

    summary = summarize(equity_df, trades, config, start=dates[3], end=dates[5])

    assert summary["equity_final"] == pytest.approx(9900.0)
    assert summary["max_drawdown"] == pytest.approx(-0.10)
    assert summary["total_return"] == pytest.approx(-0.01)
    assert summary["n_trades"] == 1
    assert summary["avg_pnl"] == pytest.approx(8.0)
    assert summary["exposure"] == pytest.approx(2 / 3)
    assert summary["turnover_annual"] == pytest.approx(2 * 105 / ((100 + 110 + 99) / 3) / (2 / PPY))


def test_break_even_interpolates_first_zero_crossing():
    """Sharpe [1, 0.5, -0.5] en [0, 5, 10] bps: cruza entre 5 y 10 -> 5 + 0.5·5/1 = 7.5 bps."""
    assert break_even_cost([0, 5, 10], [1.0, 0.5, -0.5]) == pytest.approx(7.5)
    assert math.isnan(break_even_cost([0, 5, 10], [1.0, 0.8, 0.6]))
    assert math.isnan(break_even_cost([0, 5, 10], [-0.2, -0.4, -0.6]))


def test_cost_sensitivity_grid_and_costs_hurt():
    """11 costos (0-50 bps); la equity final a 50 bps no supera la de 0 bps; no modifica df.

    Usa ~104 dias reales de train: la estrategia necesita velas de 4h para calentar.
    """
    df = load_train().iloc[:30_000]
    df_before = df.copy()
    periods = {"a": ("2022-06-01", "2022-07-31"), "b": ("2022-08-01", "2022-09-13")}

    sensitivity = cost_sensitivity(df, THETA0, periods)

    assert sensitivity.index.tolist() == list(range(0, 55, 5))
    assert list(sensitivity.columns) == ["sharpe_a", "equity_final_a", "sharpe_b", "equity_final_b"]
    for period in ["a", "b"]:
        assert sensitivity.loc[50, f"equity_final_{period}"] <= sensitivity.loc[0, f"equity_final_{period}"]
    pd.testing.assert_frame_equal(df, df_before)
