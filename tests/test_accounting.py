"""Prueba 3 del PDF (seccion 3.7): la contabilidad del backtest cuadra.

Escenario sintetico con 4 operaciones (2 largos y 2 cortos), ATR = 1,
sl_mult = 2, tp_mult = 3 y holding maximo de 3 barras:

| t | Open | High  | Low  | Close | señal | que pasa                                  |
| 0 | 100  | 100   | 100  | 100   | +1    |                                           |
| 1 | 100  | 100   | 100  | 100   |  0    | abre largo a 100 (SL 98, TP 103)          |
| 2 | 100  | 100   | 100  | 100   |  0    |                                           |
| 3 | 100  | 100   | 100  | 100   | -1    | cierra por holding a 100 (precio igual)   |
| 4 | 100  | 100   | 96.5 | 97    | +1    | abre corto a 100 (TP 97) y sale en el TP  |
| 5 | 97   | 97.5  | 94   | 95    | -1    | abre largo a 97 (SL 95) y sale en el SL   |
| 6 | 95   | 95.5  | 94.5 | 95    |  0    | abre corto a 95 (SL 97)                   |
| 7 | 95   | 95.2  | 94.8 | 95.1  |  0    |                                           |
| 8 | 95.1 | 96.5  | 95   | 96    |  0    | cierra por holding a 96                   |
| 9 | 96   | 96    | 96   | 96    |  0    | sin posicion                              |
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BacktestConfig, backtest

COMMISSION = 0.00125
BARS = [
    # Open, High, Low, Close, señal
    (100, 100, 100, 100, 1),
    (100, 100, 100, 100, 0),
    (100, 100, 100, 100, 0),
    (100, 100, 100, 100, -1),
    (100, 100, 96.5, 97, 1),
    (97, 97.5, 94, 95, -1),
    (95, 95.5, 94.5, 95, 0),
    (95, 95.2, 94.8, 95.1, 0),
    (95.1, 96.5, 95, 96, 0),
    (96, 96, 96, 96, 0),
]
CONFIG = BacktestConfig(initial_cash=100_000.0, rho=0.01, sl_mult=2.0, tp_mult=3.0, max_holding=3,
                        commission_rate=COMMISSION, slippage_rate=0.0, borrow_fee_annual=0.0)


def _run():
    """Corre el backtest del escenario de la tabla."""
    index = pd.date_range("2023-01-02", periods=len(BARS), freq="5min")
    bars = pd.DataFrame(BARS, columns=["Open", "High", "Low", "Close", "signal"], index=index)
    df = bars[["Open", "High", "Low", "Close"]]
    atr = pd.Series(1.0, index=index)
    return df, backtest(df, bars["signal"], atr, CONFIG)


DF, RESULT = _run()
EQUITY, TRADES = RESULT.equity, RESULT.trades


def _gross_pnl(trades: pd.DataFrame) -> pd.Series:
    """PnL antes de costos de cada trade, con precios crudos."""
    move = trades["raw_exit_price"] - trades["raw_entry_price"]
    direction = trades["side"].map({"long": 1, "short": -1})
    return direction * move * trades["shares"]


def test_scenario_has_longs_and_shorts():
    """El escenario produce las 4 operaciones esperadas y termina sin posicion."""
    assert list(zip(TRADES["side"], TRADES["exit_reason"])) == [
        ("long", "max_holding"), ("short", "take_profit"), ("long", "stop_loss"), ("short", "max_holding"),
    ]
    assert EQUITY["shares"].iloc[-1] == 0


def test_equity_is_cash_plus_units_times_close():
    """(a) En cada barra: equity = cash + unidades · Close (unidades con signo)."""
    expected = EQUITY["cash"] + EQUITY["shares"] * DF["Close"]
    assert EQUITY["equity"].to_numpy() == pytest.approx(expected.to_numpy())
    # Con posicion abierta las unidades no son 0, asi que la prueba revisa el valor a mercado.
    assert (EQUITY["shares"] != 0).sum() >= 4


def test_commissions_match_formula():
    """(b) Comision = 0.00125 · precio · unidades en cada apertura y en cada cierre."""
    entry = COMMISSION * TRADES["raw_entry_price"] * TRADES["shares"]
    exit_ = COMMISSION * TRADES["raw_exit_price"] * TRADES["shares"]
    assert TRADES["entry_commission"].to_numpy() == pytest.approx(entry.to_numpy())
    assert TRADES["exit_commission"].to_numpy() == pytest.approx(exit_.to_numpy())

    # Lo que salio de la caja por costos es exactamente la suma de las columnas.
    charged = CONFIG.initial_cash + _gross_pnl(TRADES).sum() - EQUITY["cash"].iloc[-1]
    total_commission = (TRADES["entry_commission"] + TRADES["exit_commission"]).sum()
    assert charged == pytest.approx(total_commission)


def test_round_trip_at_same_price_pays_twice_the_commission():
    """(b) Si el precio no cambia, el trade paga 2 · 0.00125 · nocional y pierde justo eso."""
    flat = TRADES.iloc[0]
    assert flat["raw_entry_price"] == flat["raw_exit_price"] == 100
    notional = flat["raw_entry_price"] * flat["shares"]
    assert flat["entry_commission"] + flat["exit_commission"] == pytest.approx(2 * COMMISSION * notional)
    assert flat["pnl"] == pytest.approx(-2 * COMMISSION * notional)


def test_final_capital_is_initial_plus_gross_pnl_minus_commissions():
    """(c) Con slippage 0: capital final = inicial + Σ PnL bruto − Σ comisiones."""
    assert (TRADES["entry_slippage"] == 0).all() and (TRADES["exit_slippage"] == 0).all()
    commissions = (TRADES["entry_commission"] + TRADES["exit_commission"]).sum()
    expected = CONFIG.initial_cash + _gross_pnl(TRADES).sum() - commissions
    assert EQUITY["equity"].iloc[-1] == pytest.approx(expected)
    assert EQUITY["cash"].iloc[-1] == pytest.approx(expected)
