"""Prueba 1 del PDF (seccion 3.7): causalidad de la señal, sin look-ahead.

La señal, el estado y los votos en t deben ser los mismos si se calculan
con los datos hasta t (df.iloc[:t+1]) o con la serie completa. Si algo
usara datos del futuro, los dos calculos no coincidirian.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_btc
from src.signals import compute_strategy

DATA = str(Path(__file__).resolve().parents[1] / "data" / "btc_project_train.csv")
# ~70 dias de train: alcanza para calentar los indicadores de 4h y es rapido.
DF = load_btc(DATA).iloc[:20_000]
FULL = compute_strategy(DF)
WARM_UP = 5_000
CHECK_COLUMNS = ["vote_ema", "vote_roc", "vote_bb", "state", "signal"]


def _first_position(mask: pd.Series) -> int:
    """Primera posicion despues del calentamiento donde mask es verdadero."""
    return int(mask.iloc[WARM_UP:].to_numpy().argmax()) + WARM_UP


def _test_points() -> dict[str, int]:
    """Valores de t: a la mitad de una vela de 4h, en su cierre, justo despues y con señal."""
    hour, minute = DF.index.hour, DF.index.minute
    mid_block = pd.Series((hour % 4 == 2) & (minute == 0), index=DF.index)
    block_close = pd.Series((hour % 4 == 3) & (minute == 55), index=DF.index)
    after_close = pd.Series((hour % 4 == 0) & (minute == 0), index=DF.index)
    signal_positions = (FULL["signal"] != 0).to_numpy().nonzero()[0]
    signal_positions = signal_positions[signal_positions >= WARM_UP]
    return {
        "mitad de vela 4h": _first_position(mid_block),
        "cierre de vela 4h": _first_position(block_close),
        "primera barra tras el cierre": _first_position(after_close),
        "barra con señal": int(signal_positions[0]),
        "otra barra con señal": int(signal_positions[-1]),
        "ultima barra": len(DF) - 1,
    }


POINTS = _test_points()


def test_points_cover_required_cases():
    """Hay al menos 5 valores de t y alguno tiene señal distinta de 0."""
    assert len(set(POINTS.values())) >= 5
    assert (FULL["signal"].iloc[list(POINTS.values())] != 0).any()


@pytest.mark.parametrize("name", list(POINTS))
def test_signal_uses_only_past_data(name):
    t = POINTS[name]
    truncated = compute_strategy(DF.iloc[:t + 1])
    assert truncated.index[-1] == DF.index[t]
    for col in CHECK_COLUMNS:
        assert truncated[col].iloc[-1] == FULL[col].iloc[t], f"{col} en t={t} ({name})"
    assert truncated["atr"].iloc[-1] == pytest.approx(FULL["atr"].iloc[t], nan_ok=True)


# --- Truncamiento del backtest completo (señal + motor) -------------------------

from src.backtest import backtest, config_from_params   # noqa: E402
from src.signals import THETA0                          # noqa: E402

CONFIG = config_from_params(THETA0)
BACKTEST_T = [8_000, 12_000, 16_000, len(DF) - 1]


def _closed_trades(df: pd.DataFrame, t: int) -> pd.DataFrame:
    """Trades de la estrategia completa sobre df que cerraron antes de la barra t."""
    features = compute_strategy(df)
    trades = backtest(df, features["signal"], features["atr"], CONFIG).trades
    return trades[trades["exit_bar"] < t].reset_index(drop=True)


FULL_TRADES = {t: _closed_trades(DF, t) for t in BACKTEST_T}


@pytest.mark.parametrize("t", BACKTEST_T)
def test_backtest_uses_only_past_data(t):
    """Los trades cerrados antes de t son iguales con df.iloc[:t+1] y con la serie completa."""
    pd.testing.assert_frame_equal(_closed_trades(DF.iloc[:t + 1], t), FULL_TRADES[t])


def test_backtest_truncation_check_is_not_trivial():
    """Hay trades cerrados antes del ultimo t, asi que la comparacion revisa algo."""
    assert len(FULL_TRADES[BACKTEST_T[-1]]) > 0
