"""Pruebas de la señal (Lab 02, seccion 3.7): causalidad y regla de confirmacion 2 de 3."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_btc
from src.strategy import MIN_VOTES, compute_features, confirmation_signal, indicator_votes

DATA = "data/btc_project_train.csv"


def _votes(rows):
    return pd.DataFrame(rows, columns=["vote_roc", "vote_macd", "vote_adx"])


@pytest.mark.parametrize("votes, expected", [
    ((0, 0, 0), 0),      # nadie vota
    ((1, 0, 0), 0),      # 1 a favor de largo: NO abre
    ((0, -1, 0), 0),     # 1 a favor de corto: NO abre
    ((1, -1, 0), 0),     # 1 largo y 1 corto: NO abre
    ((1, 1, 0), 1),      # 2 de 3 largos: abre largo
    ((1, 0, 1), 1),
    ((1, 1, -1), 1),     # 2 largos contra 1 corto: abre largo
    ((1, 1, 1), 1),      # 3 de 3
    ((-1, -1, 0), -1),   # 2 de 3 cortos: abre corto
    ((-1, 1, -1), -1),
    ((-1, -1, -1), -1),
])
def test_confirmation_rule_two_of_three(votes, expected):
    assert MIN_VOTES == 2
    assert confirmation_signal(_votes([votes])).iloc[0] == expected


def test_single_indicator_never_opens_on_real_data():
    feats = compute_features(load_btc(DATA).iloc[:20_000])
    one_long = (feats["n_long"] == 1) & (feats["n_short"] < 2)
    one_short = (feats["n_short"] == 1) & (feats["n_long"] < 2)
    assert one_long.sum() > 0 and one_short.sum() > 0
    assert (feats.loc[one_long | one_short, "signal"] == 0).all()
    assert (feats.loc[feats["n_long"] >= 2, "signal"] == 1).all()
    assert (feats.loc[feats["n_short"] >= 2, "signal"] == -1).all()


def test_adx_only_votes_when_trend_is_strong():
    idx = range(4)
    votes = indicator_votes(
        roc_values=pd.Series([1.0, 1.0, -1.0, np.nan], index=idx),
        macd_hist=pd.Series([0.5, -0.5, -0.5, 0.5], index=idx),
        plus_di=pd.Series([30.0, 30.0, 10.0, 30.0], index=idx),
        minus_di=pd.Series([10.0, 10.0, 30.0, 10.0], index=idx),
        adx_values=pd.Series([40.0, 20.0, 40.0, 40.0], index=idx),   # barra 1: ADX < 25
        adx_threshold=25.0)
    assert votes["vote_adx"].tolist() == [1, 0, -1, 1]
    assert votes["vote_roc"].tolist() == [1, 1, -1, 0]             # NaN no vota
    assert confirmation_signal(votes).tolist() == [1, 0, -1, 1]


@pytest.mark.parametrize("t", [500, 5_000, 12_345, 19_999])
def test_signal_is_causal(t):
    df = load_btc(DATA).iloc[:20_000]
    full = compute_features(df)
    part = compute_features(df.iloc[:t + 1])
    assert part["signal"].iloc[-1] == full["signal"].iloc[t]
    cols = ["roc", "macd_hist", "plus_di", "minus_di", "adx", "atr_14"]
    np.testing.assert_allclose(part[cols].iloc[-1].to_numpy(dtype=float),
                               full[cols].iloc[t].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)