"""Pruebas de src/regimes.py: causalidad de las features, caso a mano y scaler.

Datos: NVDA solo hasta el fin de test (src/splits.py); validation no se usa.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regimes import REGIME_COLUMNS, apply_scaler, fit_scaler, regime_features
from src.splits import SPLITS, get_split

TEST_END = SPLITS["test"][1]
DF = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "NVDA_daily.csv",
                 index_col="Date", parse_dates=True).loc[:TEST_END]

# Cada 40 barras desde la 60, mas la ultima barra.
T_BARS = list(range(60, len(DF), 40)) + [len(DF) - 1]
ATOL = 1e-9
FULL_FEATURES = regime_features(DF)


@pytest.mark.parametrize("t", T_BARS)
@pytest.mark.parametrize("col", REGIME_COLUMNS)
def test_regime_feature_is_causal(col, t):
    """El valor en t con la serie completa es igual al calculado con df.iloc[:t+1]."""
    full = FULL_FEATURES.iloc[t][col]
    truncated = regime_features(DF.iloc[:t + 1]).iloc[-1][col]
    np.testing.assert_allclose(full, truncated, rtol=0, atol=ATOL, equal_nan=True)


def test_trend_r2_perfect_linear_trend():
    """Log-precio exactamente lineal en el tiempo -> trend_r2 = 1 en toda ventana completa."""
    n, window = 80, 63
    t = np.arange(n)
    df = pd.DataFrame({"Close": np.exp(0.5 + 0.01 * t)},
                      index=pd.date_range("2020-01-01", periods=n, freq="D"))

    r2 = regime_features(df, window=window)["trend_r2"]

    assert r2.iloc[:window - 1].isna().all()
    np.testing.assert_allclose(r2.iloc[window - 1:], 1.0, rtol=0, atol=1e-12)


def test_scaler_uses_train_only():
    """Train escalado queda con media 0 y std 1; test se escala con la media/std de train."""
    train = get_split(FULL_FEATURES, "train")
    test = get_split(FULL_FEATURES, "test")
    scaler = fit_scaler(train)

    train_z = apply_scaler(train, scaler)
    np.testing.assert_allclose(train_z.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(train_z.std(ddof=0), 1.0, atol=1e-12)

    expected = (test - train.mean()) / train.std(ddof=0)
    pd.testing.assert_frame_equal(apply_scaler(test, scaler), expected)
