import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BacktestConfig, backtest
from src.backtest import COMMISSION_RATE, SLIPPAGE_RATE

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Valores calculados a mano (ver docstring de test_golden_long_tp_then_short_tp).
EXPECTED_EQUITY = [10000.00, 10047.50, 10144.85, 10192.89, 10291.76]
EXPECTED_TRADES = [("long", "take_profit"), ("short", "take_profit")]


def test_golden_long_tp_then_short_tp():
    """Golden file: long que sale por TP y short que sale por TP, calculado a mano.

    Escenario en tests/golden/backtest_scenario.csv (ATR = 2 en todas las barras).
    config: initial_cash=10_000, rho=0.01, sl_mult=2, tp_mult=3, max_holding=10,
    cost_rate=0.001, borrow_fee_annual=0.005.

    | t | fecha | Open | High  | Low  | Close | señal |
    | 0 | 01-01 | 100  | 101   | 99   | 100   | +1    |
    | 1 | 01-02 | 100  | 103   | 99.5 | 102   | +1    |
    | 2 | 01-03 | 102  | 107   | 101  | 106.5 | -1    |
    | 3 | 01-04 | 106  | 106.5 | 103  | 104   |  0    |
    | 4 | 01-05 | 103  | 104   | 99   | 100   |  0    |

    Calculo a mano:
    - t=1: long al open 100. Q = floor(0.01·10000/(2·2)) = 25. Fill 100·1.001 =
      100.10 -> cash = 10000 - 2502.50 = 7497.50. SL 96, TP 106.
      equity = 7497.50 + 25·102 = 10047.50
    - t=2: High 107 >= TP 106 -> sale a 106·0.999 = 105.894 -> cash = 7497.50 +
      2647.35 = 10144.85 = equity.
    - t=3: short al open 106 (señal -1 de t=2). Q = floor(101.4485/4) = 25.
      Fill 106·0.999 = 105.894 -> cash = 10144.85 + 2647.35 = 12792.20.
      SL 110, TP 100. equity = 12792.20 - 25·104 = 10192.89
    - t=4: Low 99 <= TP 100 -> cubre a 100·1.001 = 100.10 -> cash = 12792.20 -
      2502.50 = 10291.76; borrow fee 1 dia = 0.005·25·106·1/360 = 0.0368 ->
      cash final = 10291.72.

    Equity esperado por barra: [10000, 10047.50, 10144.85, 10192.89, 10291.76]
    (tolerancia 0.01). Trades: 2 (long take_profit, short take_profit).
    """
    scenario = pd.read_csv(GOLDEN_DIR / "backtest_scenario.csv", index_col="Date", parse_dates=True)
    df = scenario[["Open", "High", "Low", "Close"]]
    config = BacktestConfig(initial_cash=10_000, rho=0.01, sl_mult=2, tp_mult=3, max_holding=10,
                            cost_rate=0.001, borrow_fee_annual=0.005)

    result = backtest(df, scenario["signal"], scenario["atr"], config)

    assert result.equity["equity"].tolist() == pytest.approx(EXPECTED_EQUITY, abs=0.01)
    assert list(zip(result.trades["side"], result.trades["exit_reason"])) == EXPECTED_TRADES


def test_per_bar_exit_params_fixed_at_entry():
    """sl/tp/max_holding por barra: se toman de la barra de señal (t-1) y se conservan hasta salir.

    Mismo escenario golden (ATR = 2). Valores por barra:
    tp_mult = [5, 1, 1, 1, 1], max_holding = [2, 1, 1, 1, 1], sl_mult = 2.

    - t=1: long al open 100 con los valores de t=0: SL = 100 - 2·2 = 96,
      TP = 100 + 5·2 = 110, holding 2. High 103 < 110 -> sigue abierto.
    - t=2: High 107 < 110 y Low 101 > 96; barra 2 de 2 -> sale al cierre
      106.5 por max_holding.
    Si el motor usara los valores de t=1 (TP = 102 u holding 1), el long
    saldria en t=1.
    """
    scenario = pd.read_csv(GOLDEN_DIR / "backtest_scenario.csv", index_col="Date", parse_dates=True)
    df = scenario[["Open", "High", "Low", "Close"]]
    config = BacktestConfig(initial_cash=10_000, rho=0.01, sl_mult=2, tp_mult=3, max_holding=10,
                            cost_rate=0.001, borrow_fee_annual=0.005)
    tp_mult = pd.Series([5, 1, 1, 1, 1], index=df.index, dtype=float)
    max_holding = pd.Series([2, 1, 1, 1, 1], index=df.index)

    result = backtest(df, scenario["signal"], scenario["atr"], config,
                      tp_mult=tp_mult, max_holding=max_holding)

    first = result.trades.iloc[0]
    assert (first["side"], first["entry_bar"], first["exit_bar"]) == ("long", 1, 2)
    assert first["exit_reason"] == "max_holding"
    assert first["raw_exit_price"] == pytest.approx(106.5)


def test_accounting_identities_with_lab_costs():
    """Contabilidad del backtest con los parametros del Lab 02 (seccion 3.7, prueba 3).

    1. En cada barra, equity = efectivo + unidades · Close.
    2. Sin posicion abierta al final, equity final = capital inicial + Σ pnl.
    3. Costos cobrados = (comision + slippage) · nocional operado (cada trade son 2 operaciones).
    4. El efectivo nunca es negativo (sin apalancamiento).
    """
    scenario = pd.read_csv(GOLDEN_DIR / "backtest_scenario.csv", index_col="Date", parse_dates=True)
    df = scenario[["Open", "High", "Low", "Close"]]
    config = BacktestConfig()
    assert (COMMISSION_RATE, SLIPPAGE_RATE) == (0.00125, 0.0005)
    assert (config.initial_cash, config.cost_rate, config.borrow_fee_annual) == \
        (1_000_000.0, COMMISSION_RATE + SLIPPAGE_RATE, 0.0)

    result = backtest(df, scenario["signal"], scenario["atr"], config)
    equity, trades = result.equity, result.trades

    assert len(trades) == 2
    assert (equity["equity"] == equity["cash"] + equity["shares"] * df["Close"]).all()
    assert equity["shares"].iloc[-1] == 0
    assert equity["equity"].iloc[-1] == pytest.approx(config.initial_cash + trades["pnl"].sum())

    is_long = trades["side"] == "long"
    gross = (trades["shares"] * (trades["raw_exit_price"] - trades["raw_entry_price"])).where(
        is_long, trades["shares"] * (trades["raw_entry_price"] - trades["raw_exit_price"]))
    costs_charged = (gross - trades["pnl"]).sum()
    notional = (trades["shares"] * (trades["raw_entry_price"] + trades["raw_exit_price"])).sum()
    assert costs_charged == pytest.approx(config.cost_rate * notional)
    assert (equity["cash"] >= -1e-9).all()

    