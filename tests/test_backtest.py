import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BacktestConfig, backtest

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Valores calculados a mano (ver docstring de test_golden_long_tp_then_short_tp).
EXPECTED_EQUITY = [10000.00, 10047.50, 10144.85, 10192.20, 10289.66]
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
      SL 110, TP 100. equity = 12792.20 - 25·104 = 10192.20
    - t=4: Low 99 <= TP 100 -> cubre a 100·1.001 = 100.10 -> cash = 12792.20 -
      2502.50 = 10289.70; borrow fee 1 dia = 0.005·25·106·1/360 = 0.0368 ->
      cash final = 10289.66.

    Equity esperado por barra: [10000, 10047.50, 10144.85, 10192.20, 10289.66]
    (tolerancia 0.01). Trades: 2 (long take_profit, short take_profit).
    """
    scenario = pd.read_csv(GOLDEN_DIR / "backtest_scenario.csv", index_col="Date", parse_dates=True)
    df = scenario[["Open", "High", "Low", "Close"]]
    config = BacktestConfig(initial_cash=10_000, rho=0.01, sl_mult=2, tp_mult=3, max_holding=10,
                            cost_rate=0.001, borrow_fee_annual=0.005)

    result = backtest(df, scenario["signal"], scenario["atr"], config)

    assert result.equity["equity"].tolist() == pytest.approx(EXPECTED_EQUITY, abs=0.01)
    assert list(zip(result.trades["side"], result.trades["exit_reason"])) == EXPECTED_TRADES
