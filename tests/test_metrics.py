import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    PERIODS_PER_YEAR,
    bootstrap_mean_ci,
    break_even_cost,
    buy_and_hold_equity,
    cagr,
    calmar_ratio,
    drawdown_series,
    exit_reason_table,
    exposure,
    kruskal_by_regime,
    max_drawdown,
    performance_summary,
    pnl_breakdown,
    regime_trade_stats,
    returns,
    returns_table,
    side_table,
    sharpe_ratio,
    sortino_ratio,
    trade_returns,
)

PPY = PERIODS_PER_YEAR
SQRT_PPY = math.sqrt(PPY)

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


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


def test_break_even_interpolates_first_zero_crossing():
    """Sharpe [1, 0.5, -0.5] en [0, 5, 10] bps: cruza entre 5 y 10 -> 5 + 0.5·5/1 = 7.5 bps."""
    assert break_even_cost([0, 5, 10], [1.0, 0.5, -0.5]) == pytest.approx(7.5)
    assert math.isnan(break_even_cost([0, 5, 10], [1.0, 0.8, 0.6]))
    assert math.isnan(break_even_cost([0, 5, 10], [-0.2, -0.4, -0.6]))


def test_performance_summary_by_hand():
    """Equity 100 -> 110 -> 99 -> 121 con 2 trades (1 gana, 1 pierde).

    Retorno total = 121/100 - 1 = 21%; MDD = 99/110 - 1 = -10%;
    win rate = 50%; exposicion = 2 de 4 barras con unidades = 0.5.
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="5min")
    equity_df = pd.DataFrame({"equity": [100.0, 110.0, 99.0, 121.0], "shares": [0, 1, 1, 0]}, index=dates)
    trades = pd.DataFrame({"pnl": [10.0, -5.0]})

    summary = performance_summary(equity_df, trades)

    assert summary["total_return"] == pytest.approx(0.21)
    assert summary["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert summary["win_rate"] == 50.0
    assert summary["n_trades"] == 2
    assert summary["exposure"] == 0.5
    assert summary["sharpe"] == pytest.approx(sharpe_ratio(returns(equity_df["equity"])))
    assert summary["calmar"] == pytest.approx(calmar_ratio(equity_df["equity"]))


def test_performance_summary_without_trades_has_nan_win_rate():
    """Buy & hold no tiene trades: win rate NaN y 0 trades."""
    dates = pd.date_range("2024-01-01", periods=3, freq="5min")
    equity_df = pd.DataFrame({"equity": [100.0, 101.0, 102.0], "shares": [1, 1, 1]}, index=dates)
    summary = performance_summary(equity_df, pd.DataFrame(columns=["pnl"]))
    assert summary["n_trades"] == 0
    assert math.isnan(summary["win_rate"])


def test_returns_table_by_hand():
    """Enero cierra en 110 (+10% sobre 100), febrero en 121 (+10%), marzo en 100 (-17.36%).

    El trimestre completo: 100 -> 100 = 0%.
    """
    equity = pd.Series([100.0, 105.0, 110.0, 121.0, 100.0],
                       index=pd.to_datetime(["2023-01-01", "2023-01-15", "2023-01-31",
                                             "2023-02-15", "2023-03-31"]))
    monthly = returns_table(equity, "ME")
    assert monthly.tolist() == pytest.approx([0.10, 0.10, 100 / 121 - 1])
    assert returns_table(equity, "QE").tolist() == pytest.approx([0.0])
    assert returns_table(equity, "YE").index[0] == pd.Timestamp("2023-12-31")


# --- Analisis de trades (SPEC 14.4) y diagnostico -------------------------------

TRADES = pd.DataFrame({
    "side": ["long", "long", "short", "short"],
    "exit_reason": ["take_profit", "stop_loss", "take_profit", "stop_loss"],
    "shares": [10.0, 10.0, 5.0, 5.0],
    "raw_entry_price": [100.0, 100.0, 200.0, 200.0],
    "pnl": [30.0, -10.0, 20.0, -20.0],
    "entry_commission": [1.0, 1.0, 1.0, 1.0],
    "exit_commission": [1.0, 1.0, 1.0, 1.0],
    "entry_slippage": 0.0, "exit_slippage": 0.0, "borrow_fee": 0.0,
})


def test_trade_returns_on_entry_notional():
    """Retorno = pnl / (unidades · precio de entrada): 30/1000 = 3%, -20/1000 = -2%."""
    assert trade_returns(TRADES).tolist() == pytest.approx([0.03, -0.01, 0.02, -0.02])


def test_bootstrap_ci_constant_and_reproducible():
    """Valores constantes -> IC de un solo punto; con semilla fija el IC se repite."""
    assert bootstrap_mean_ci([0.5, 0.5, 0.5]) == (0.5, 0.5)
    values = [0.1, -0.2, 0.3, 0.05, -0.1]
    low, high = bootstrap_mean_ci(values)
    assert (low, high) == bootstrap_mean_ci(values)
    assert low <= sum(values) / len(values) <= high
    assert all(math.isnan(v) for v in bootstrap_mean_ci([0.1]))


def test_regime_trade_stats_and_kruskal():
    """Por regimen: n, win rate y media. Kruskal: grupos separados -> p bajo; iguales -> p alto."""
    regimes = pd.Series(["trend", "trend", "crisis", "crisis"])
    stats = regime_trade_stats(TRADES, regimes)
    assert stats.loc["trend", "n_trades"] == 2
    assert stats.loc["trend", "win_rate"] == 50.0
    assert stats.loc["trend", "mean_return"] == pytest.approx(0.01)

    n = 30
    separated = pd.DataFrame({"pnl": [1.0] * n + [-1.0] * n, "shares": 1.0, "raw_entry_price": 1.0})
    groups = pd.Series(["a"] * n + ["b"] * n)
    assert kruskal_by_regime(separated, groups)["p_value"] < 0.001
    same = pd.DataFrame({"pnl": [1.0, -1.0] * n, "shares": 1.0, "raw_entry_price": 1.0})
    assert kruskal_by_regime(same, groups)["p_value"] > 0.5


def test_exit_reason_and_side_tables():
    """Agrupa trades, win rate (%) y PnL total por motivo de salida y por lado."""
    by_exit = exit_reason_table(TRADES)
    assert by_exit.loc["take_profit", "total_pnl"] == 50.0
    assert by_exit.loc["stop_loss", "win_rate"] == 0.0
    by_side = side_table(TRADES)
    assert by_side.loc["long", "n_trades"] == 2
    assert by_side.loc["short", "total_pnl"] == 0.0


def test_pnl_breakdown_by_hand():
    """Neto 20; comisiones 8 -> bruto 28. Ganancia prom. 25, perdida prom. 15 -> payoff 5/3.

    Win rate de break-even = 1 / (1 + 5/3) = 37.5%; win rate real = 50%.
    """
    diag = pnl_breakdown(TRADES)
    assert diag["net_pnl"] == 20.0
    assert diag["commissions"] == 8.0
    assert diag["gross_pnl"] == 28.0
    assert diag["payoff"] == pytest.approx(25 / 15)
    assert diag["break_even_win_rate"] == pytest.approx(37.5)
    assert diag["win_rate"] == 50.0
