"""Pruebas de src/regime_analysis.py con casos a mano."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime_analysis import (
    count_transitions,
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
