"""Prueba de truncamiento del pipeline completo: indicadores -> señal -> backtest.

Si el pipeline es causal, el valor en la barra t calculado con la serie
completa es igual al calculado con df.iloc[:t+1] (solo datos hasta t).
Los canarios demuestran que la misma comparacion detecta una fuga a proposito.

Datos: NVDA solo hasta el fin de test (src/splits.py); validation no se usa.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BacktestConfig, backtest
from src.splits import SPLITS
from src.strategy import compute_features

TEST_END = SPLITS["test"][1]
DF = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "NVDA_daily.csv",
                 index_col="Date", parse_dates=True).loc[:TEST_END]

# Cada 40 barras desde la 60, mas la ultima barra.
T_BARS = list(range(60, len(DF), 40)) + [len(DF) - 1]
FEATURE_COLS = ["roc_10", "cmf_20", "adx_14", "atr_14", "signal"]
ATOL = 1e-9


def _check_features(df: pd.DataFrame, t: int, features_fn) -> None:
    """Fila t de features con la serie completa vs ultima fila con df.iloc[:t+1]."""
    full = features_fn(df).iloc[t][FEATURE_COLS].to_numpy(dtype=float)
    truncated = features_fn(df.iloc[:t + 1]).iloc[-1][FEATURE_COLS].to_numpy(dtype=float)
    np.testing.assert_allclose(full, truncated, rtol=0, atol=ATOL, equal_nan=True)


def _run(df: pd.DataFrame, features_fn):
    """Corre el backtest con la señal y el ATR calculados sobre df."""
    features = features_fn(df)
    return backtest(df, features["signal"], features["atr_14"], BacktestConfig())


def _check_backtest(df: pd.DataFrame, t: int, features_fn) -> None:
    """Estado en t y trades cerrados hasta t: serie completa vs df.iloc[:t+1]."""
    full = _run(df, features_fn)
    truncated = _run(df.iloc[:t + 1], features_fn)

    np.testing.assert_allclose(full.equity.iloc[t].to_numpy(dtype=float),
                               truncated.equity.iloc[-1].to_numpy(dtype=float),
                               rtol=0, atol=ATOL)

    full_trades = full.trades[full.trades["exit_bar"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_trades, truncated.trades.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=ATOL)


def _failing_bars(check, features_fn) -> list[int]:
    """Barras de T_BARS en las que la comparacion de truncamiento falla."""
    failing = []
    for t in T_BARS:
        try:
            check(DF, t, features_fn)
        except AssertionError:
            failing.append(t)
    return failing


def leaky_features(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Features reales con una señal que mira lag barras al futuro (fuga a proposito)."""
    features = compute_features(df)
    close = df["Close"]
    features["signal"] = np.where(close.shift(-lag) > close, 1, -1)
    return features


@pytest.mark.parametrize("t", T_BARS)
def test_features_are_causal(t):
    """ROC, CMF, ADX, ATR y la señal en t no cambian si se quitan las barras posteriores a t."""
    _check_features(DF, t, compute_features)


@pytest.mark.parametrize("t", T_BARS)
def test_backtest_is_causal(t):
    """cash, shares, equity en t y los trades cerrados hasta t no dependen de barras posteriores."""
    _check_backtest(DF, t, compute_features)


def test_canary_leaky_signal_fails_feature_check():
    """Una señal con close.shift(-1) > close debe fallar la comparacion de features.

    Con la serie completa, la señal en t usa Close[t+1]; truncada en t, ese
    dato no existe (NaN -> señal -1). Falla en toda t donde Close[t+1] > Close[t].
    """
    assert _failing_bars(_check_features, partial(leaky_features, lag=1))


def test_canary_leaky_signal_fails_backtest_check():
    """Una señal con close.shift(-2) > close debe fallar la comparacion del backtest.

    El motor ejecuta en t la señal de t-1. Con shift(-1) esa señal solo usa
    Close[t], que si esta en df.iloc[:t+1], asi que el estado en t no cambia:
    una fuga de 1 barra solo la detecta la comparacion de features. Con
    shift(-2) la señal de t-1 usa Close[t+1], ausente en la serie truncada,
    y el backtest en t ya difiere.
    """
    assert _failing_bars(_check_backtest, partial(leaky_features, lag=2))
