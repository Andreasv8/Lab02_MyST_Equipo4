"""Pruebas de src/data.py: carga de archivos y paso de velas de 4h (o 1h) a 5 min sin look-ahead."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (PRICE_COLUMNS, TEST_END, TEST_START, align_to_base, load_test, load_train,
                      resample_ohlc)

TRAIN = load_train()


def test_load_train_and_test_ranges():
    """Train va de 2022-06-01 a 2023-12-31; test tiene datos de mayo-junio 2024 y no tiene NaN de precio."""
    test = load_test()
    assert TRAIN.index[0] == pd.Timestamp("2022-06-01")
    assert TRAIN.index[-1].normalize() == pd.Timestamp("2023-12-31")
    assert len(test.loc[TEST_START:TEST_END]) > 0
    assert test.index[-1] <= pd.Timestamp(TEST_END) + pd.Timedelta(days=1)
    for df in (TRAIN, test):
        assert not df[PRICE_COLUMNS].isna().any().any()
        assert df.index.is_monotonic_increasing


def test_resample_ohlc_aggregates_one_hour():
    """La vela de 1h de las 00:00 junta las barras de 00:00 a 00:55."""
    df = TRAIN.iloc[:24]
    hourly = resample_ohlc(df, "1h")
    first = df.loc["2022-06-01 00:00":"2022-06-01 00:55"]
    assert hourly.index[0] == pd.Timestamp("2022-06-01 00:00")
    assert hourly["Open"].iloc[0] == first["Open"].iloc[0]
    assert hourly["High"].iloc[0] == first["High"].max()
    assert hourly["Low"].iloc[0] == first["Low"].min()
    assert hourly["Close"].iloc[0] == first["Close"].iloc[-1]


def test_align_to_base_has_no_look_ahead():
    """La vela de 1h de las 00:00 se conoce al cierre de la barra de 5 min de las 00:55, no antes."""
    df = TRAIN.iloc[:36]
    hourly_close = align_to_base(resample_ohlc(df, "1h")["Close"], df.index, "1h")
    assert hourly_close.loc[:"2022-06-01 00:50"].isna().all()
    assert hourly_close.loc["2022-06-01 00:55"] == df["Close"].loc["2022-06-01 00:55"]
    assert hourly_close.loc["2022-06-01 01:50"] == df["Close"].loc["2022-06-01 00:55"]
