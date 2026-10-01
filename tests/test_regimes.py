"""Pruebas de src/regimes.py: features, scaler y clasificadores de regimen.

Datos: NVDA solo hasta el fin de test (src/splits.py); validation no se usa.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regimes import (
    REGIME_COLUMNS,
    REGIME_NAMES,
    MIN_DURATION,
    apply_scaler,
    classify_kmeans,
    classify_rules,
    fit_kmeans,
    fit_regime_models,
    fit_scaler,
    hmm_expected_durations,
    hmm_forward,
    name_states,
    regime_features,
    regime_labels,
)
from src.splits import SPLITS, get_split

TEST_END = SPLITS["test"][1]
DF = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "NVDA_daily.csv",
                 index_col="Date", parse_dates=True).loc[:TEST_END]

# Cada 40 barras desde la 60, mas la ultima barra.
T_BARS = list(range(60, len(DF), 40)) + [len(DF) - 1]
ATOL = 1e-9
FULL_FEATURES = regime_features(DF)
MODELS = fit_regime_models(DF)
FULL_LABELS = regime_labels(DF, MODELS)
X_SCALED = apply_scaler(FULL_FEATURES.dropna(), MODELS.scaler).to_numpy()


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


def test_rules_hand_case():
    """Una fila por regimen con umbral de volatilidad 0.6, mas una fila de warm-up."""
    features = pd.DataFrame({
        "volatility": [0.8, 0.3, 0.3, np.nan],
        "trend_r2": [0.9, 0.7, 0.2, 0.9],   # crisis gana aunque trend_r2 > 0.5
        "autocorr_1": [0.0, 0.0, 0.0, 0.0],
    })
    labels = classify_rules(features, vol_threshold=0.6)
    assert labels.iloc[:3].tolist() == ["crisis", "trend", "mean_reversion"]
    assert pd.isna(labels.iloc[3])


def test_name_states_assigns_all_three():
    """Mayor volatilidad = crisis; de los otros dos, mayor trend_r2 = trend."""
    centers = pd.DataFrame({"volatility": [0.4, 0.9, 0.3], "trend_r2": [0.2, 0.1, 0.8],
                            "autocorr_1": [0.0, 0.0, 0.0]})
    assert name_states(centers) == {1: "crisis", 2: "trend", 0: "mean_reversion"}


def test_kmeans_names_stable_across_seeds():
    """Con dos random_state distintos los IDs pueden cambiar, pero las etiquetas por nombre no."""
    train = get_split(FULL_FEATURES, "train")
    labels = []
    for seed in (42, 7):
        model, names = fit_kmeans(train, MODELS.scaler, random_state=seed)
        labels.append(classify_kmeans(FULL_FEATURES, MODELS.scaler, model, names))
    pd.testing.assert_series_equal(labels[0], labels[1])


@pytest.mark.parametrize("t", [0, 1, 5, 100, 400, len(X_SCALED) - 1])
def test_hmm_forward_matches_predict_proba(t):
    """Ultima fila del forward sobre x[:t+1] == predict_proba(x[:t+1])[-1] (suavizado = filtrado en T)."""
    x = X_SCALED[:t + 1]
    np.testing.assert_allclose(hmm_forward(MODELS.hmm, x)[-1],
                               MODELS.hmm.predict_proba(x)[-1], rtol=0, atol=ATOL)


def test_hmm_expected_durations_at_least_5():
    """El HMM final no tiene estados de ~1 barra (ruido): toda duracion esperada >= 5."""
    durations = hmm_expected_durations(MODELS.hmm, MODELS.hmm_names)
    assert sorted(durations.index) == sorted(REGIME_NAMES)
    for j, name in MODELS.hmm_names.items():
        np.testing.assert_allclose(durations[name], 1 / (1 - MODELS.hmm.transmat_[j, j]))
    assert (durations >= MIN_DURATION).all()


@pytest.mark.parametrize("t", T_BARS)
@pytest.mark.parametrize("method", ["rules", "kmeans", "hmm"])
def test_filtered_labels_are_causal(method, t):
    """Etiqueta filtrada en t: serie completa == df.iloc[:t+1], con modelos ya ajustados en train."""
    full = FULL_LABELS.iloc[t][method]
    truncated = regime_labels(DF.iloc[:t + 1], MODELS).iloc[-1][method]
    assert (pd.isna(full) and pd.isna(truncated)) or full == truncated


def test_canary_viterbi_has_lookahead():
    """Viterbi usa el futuro: al truncar en t, su etiqueta en t cambia en al menos una t."""
    first = FULL_FEATURES.dropna().index[0]
    start = DF.index.get_loc(first)
    changed = [t for t in range(start, len(DF), 5)
               if regime_labels(DF.iloc[:t + 1], MODELS).iloc[-1]["hmm_viterbi"]
               != FULL_LABELS.iloc[t]["hmm_viterbi"]]
    assert changed
