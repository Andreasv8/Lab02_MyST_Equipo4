"""Pruebas de src/regime_analysis.py con casos a mano."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime_analysis import (
    count_transitions,
    method_scorecard,
    regime_centroids,
    regime_runs,
    regime_shares,
    regime_silhouette,
)

LABELS = ["a", "a", "b", "b", "b", "a"]


def test_regime_runs_hand_case():
    """Rachas: a = [2, 1], b = [3]."""
    assert regime_runs(LABELS) == {"a": [2, 1], "b": [3]}


def test_regime_runs_ignores_warmup():
    """Los NaN de warm-up no forman racha."""
    assert regime_runs([np.nan, np.nan] + LABELS) == {"a": [2, 1], "b": [3]}


def test_count_transitions_hand_case():
    """a->b y b->a: 2 cambios, con o sin warm-up al inicio."""
    assert count_transitions(LABELS) == 2
    assert count_transitions([np.nan] + LABELS) == 2


def test_regime_shares_hand_case():
    """a ocupa 3 de 6 barras = 50%; un nombre ausente aparece con 0%."""
    shares = regime_shares(LABELS, names=["a", "b", "c"])
    assert shares["a"] == 50.0
    assert shares["b"] == 50.0
    assert shares["c"] == 0.0


def test_regime_centroids_hand_case():
    """Media de cada feature por regimen y por metodo."""
    features = pd.DataFrame({"volatility": [0.2, 0.4, 0.9, 1.1],
                             "trend_r2": [0.1, 0.3, 0.8, 0.6]})
    labels = pd.DataFrame({"m1": ["x", "x", "y", "y"],
                           "m2": ["x", "y", "y", np.nan]})
    centroids = regime_centroids(labels, features)

    np.testing.assert_allclose(centroids.loc[("m1", "x")], [0.3, 0.2])
    np.testing.assert_allclose(centroids.loc[("m1", "y")], [1.0, 0.7])
    np.testing.assert_allclose(centroids.loc[("m2", "x")], [0.2, 0.1])
    np.testing.assert_allclose(centroids.loc[("m2", "y")], [0.65, 0.55])  # NaN de m2 se ignora


def test_silhouette_nan_with_single_regime():
    """Con un solo regimen el silhouette no esta definido."""
    x = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    assert np.isnan(regime_silhouette(x, pd.Series(["a", "a", "a"])))


def test_method_scorecard_hand_case():
    """Un metodo, dos periodos: cambios de participacion, transiciones y consistencia de nombres."""
    names = ["crisis", "trend", "mean_reversion"]
    row = lambda trans, dur, sil, shares: {"transitions_per_month": trans, "mean_duration_all": dur,
                                           "silhouette": sil,
                                           **{f"share_{n}": v for n, v in zip(names, shares)}}
    table = pd.DataFrame.from_dict({("m", "train"): row(1.0, 20.0, 0.3, [10, 50, 40]),
                                    ("m", "test"): row(1.5, 14.0, 0.2, [16, 41, 43])},
                                   orient="index").rename_axis(["method", "period"])
    good = pd.DataFrame({"volatility": [0.7, 0.5, 0.4], "trend_r2": [0.5, 0.8, 0.2]},
                        index=pd.MultiIndex.from_product([["m"], names], names=["method", "regime"]))
    bad = good.copy()
    bad.loc[("m", "trend"), "trend_r2"] = 0.1   # trend ya no es el de mayor trend_r2

    card = method_scorecard(table, good, bad).loc["m"]

    assert card["share_shift_pp"] == pytest.approx((6 + 9 + 3) / 3)
    assert card["transitions_change"] == pytest.approx(0.5)
    assert card["mean_duration_test"] == 14.0
    assert card["names_consistent_train"]
    assert not card["names_consistent_test"]


def test_viterbi_lookahead_check_filtered_never_changes():
    """Con pocas horas: la etiqueta filtrada del HMM no cambia al truncar."""
    from src.data import load_btc
    from src.regime_analysis import viterbi_lookahead_check
    from src.regimes import fit_regime_models

    df = load_btc(str(Path(__file__).resolve().parents[1] / "data" / "btc_project_train.csv")).iloc[:30_000]
    check = viterbi_lookahead_check(df, fit_regime_models(df), step=150)
    assert len(check) >= 5
    assert check["hmm_same"].all()
