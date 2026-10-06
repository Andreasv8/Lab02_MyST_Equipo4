"""Pruebas de la estrategia 2 (EMA + ADX en timeframe mayor): regla, remuestreo y causalidad."""

import numpy as np
import pandas as pd
import pytest

from src.data import align_to_base, load_btc, resample_ohlc
from src.strategy import compute_ema_adx_features, ema_adx_signal

DATA = "data/btc_project_train.csv"


def test_ema_adx_rule():
    idx = range(5)
    fast = pd.Series([2.0, 1.0, 2.0, 1.0, np.nan], index=idx)
    slow = pd.Series([1.0, 2.0, 1.0, 1.0, 1.0], index=idx)
    adx_values = pd.Series([30.0, 30.0, 20.0, 30.0, 30.0], index=idx)
    # alcista, bajista, ADX debil, EMAs iguales, warm-up
    assert ema_adx_signal(fast, slow, adx_values, 25.0).tolist() == [1, -1, 0, 0, 0]


def test_resample_ohlc_aggregates_one_hour():
    df = load_btc(DATA).iloc[:24]
    hourly = resample_ohlc(df, "1h")
    first = df.loc["2022-06-01 00:00":"2022-06-01 00:55"]
    assert hourly.index[0] == pd.Timestamp("2022-06-01 00:00")
    assert hourly["Open"].iloc[0] == first["Open"].iloc[0]
    assert hourly["High"].iloc[0] == first["High"].max()
    assert hourly["Low"].iloc[0] == first["Low"].min()
    assert hourly["Close"].iloc[0] == first["Close"].iloc[-1]


def test_align_to_base_has_no_look_ahead():
    df = load_btc(DATA).iloc[:36]
    hourly_close = align_to_base(resample_ohlc(df, "1h")["Close"], df.index, "1h")
    # La barra de 1h de las 00:00 se conoce al cierre de la barra de 5 min de las 00:55.
    assert hourly_close.loc[:"2022-06-01 00:50"].isna().all()
    assert hourly_close.loc["2022-06-01 00:55"] == df["Close"].loc["2022-06-01 00:55"]
    assert hourly_close.loc["2022-06-01 01:50"] == df["Close"].loc["2022-06-01 00:55"]


def test_entry_on_change_only_fires_when_trend_changes():
    feats = compute_ema_adx_features(load_btc(DATA).iloc[:20_000], ema_fast=9, ema_slow=21,
                                     timeframe="1h", entry_on_change=True)
    trend, signal = feats["trend"], feats["signal"]
    changed = trend != trend.shift(1)
    assert (signal[changed] == trend[changed]).all()
    assert (signal[~changed] == 0).all()
    assert (signal != 0).sum() < (trend != 0).sum() / 10


@pytest.mark.parametrize("timeframe", ["1h", "4h"])
@pytest.mark.parametrize("t", [3_000, 7_777, 12_345, 19_999])
def test_ema_adx_features_are_causal(timeframe, t):
    df = load_btc(DATA).iloc[:20_000]
    full = compute_ema_adx_features(df, ema_fast=9, ema_slow=21, timeframe=timeframe)
    part = compute_ema_adx_features(df.iloc[:t + 1], ema_fast=9, ema_slow=21, timeframe=timeframe)
    assert part["signal"].iloc[-1] == full["signal"].iloc[t]
    assert part["trend"].iloc[-1] == full["trend"].iloc[t]
    cols = ["ema_fast", "ema_slow", "adx", "atr"]
    np.testing.assert_allclose(part[cols].iloc[-1].to_numpy(dtype=float),
                               full[cols].iloc[t].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)
