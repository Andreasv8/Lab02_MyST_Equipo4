"""Pruebas de src/regimes.py: features, scaler, clasificadores y validacion del regimen.

Datos: las primeras 30,000 barras de BTC (unos 3.5 meses, todas dentro de train)
para que las pruebas de truncamiento corran rapido.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_btc
from src.regimes import (
    FIT_END,
    FIT_START,
    WINDOW,
    hourly,
    REGIME_COLUMNS,
    REGIME_NAMES,
    MIN_DURATION,
    apply_scaler,
    classify_kmeans,
    classify_rules,
    comparison_table,
    count_transitions,
    fit_kmeans,
    fit_regime_models,
    fit_scaler,
    hmm_expected_durations,
    hmm_forward,
    name_states,
    regime_centroids,
    regime_features,
    regime_labels,
    regime_runs,
    regime_shares,
    regime_silhouette,
    regime_validation,
    rule_regimes,
    viterbi_lookahead_check,
)

DF = load_btc(str(Path(__file__).resolve().parents[1] / "data" / "btc_project_train.csv")).iloc[:30_000]

# Una barra antes del warm-up, la primera con features y varias despues (hh:00 y no hh:00).
T_BARS = [1000, WINDOW, 2500, 5000, 9999, 12_345, 15_000, 20_001, 25_000, 27_777, len(DF) - 1]
ATOL = 1e-9
FULL_FEATURES = regime_features(DF)
HOURLY_FEATURES = hourly(FULL_FEATURES)
MODELS = fit_regime_models(DF)
FULL_LABELS = regime_labels(DF, MODELS)
X_SCALED = apply_scaler(HOURLY_FEATURES.dropna(), MODELS.scaler).to_numpy()
TRUNCATED_LABELS = {t: regime_labels(DF.iloc[:t + 1], MODELS).iloc[-1] for t in T_BARS}


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
    train = HOURLY_FEATURES.iloc[:1500]
    test = HOURLY_FEATURES.iloc[1500:]
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
    """Con dos random_state distintos los IDs pueden cambiar, pero las etiquetas por nombre no.

    K-means puede llegar a optimos locales casi iguales (inercia ~0.01% distinta)
    que difieren en unos pocos puntos frontera; se exige >= 99% de coincidencia.
    Si los nombres dependieran del ID, la coincidencia caeria muy por debajo.
    """
    train = hourly(FULL_FEATURES.loc[FIT_START:FIT_END])
    labels = []
    for seed in (42, 7):
        model, names = fit_kmeans(train, MODELS.scaler, random_state=seed)
        labels.append(classify_kmeans(HOURLY_FEATURES, MODELS.scaler, model, names))
    pd.testing.assert_series_equal(labels[0].isna(), labels[1].isna())
    valid = labels[0].notna()
    assert (labels[0][valid] == labels[1][valid]).mean() >= 0.99


@pytest.mark.parametrize("t", [0, 1, 5, 100, 400, len(X_SCALED) - 1])
def test_hmm_forward_matches_predict_proba(t):
    """Ultima fila del forward sobre x[:t+1] == predict_proba(x[:t+1])[-1] (suavizado = filtrado en T)."""
    x = X_SCALED[:t + 1]
    np.testing.assert_allclose(hmm_forward(MODELS.hmm, x)[-1],
                               MODELS.hmm.predict_proba(x)[-1], rtol=0, atol=ATOL)


def test_hmm_expected_durations_at_least_12_hours():
    """El HMM final no tiene estados de ~1 barra (ruido): toda duracion esperada >= 12 h."""
    durations = hmm_expected_durations(MODELS.hmm, MODELS.hmm_names)
    assert sorted(durations.index) == sorted(REGIME_NAMES)
    for j, name in MODELS.hmm_names.items():
        np.testing.assert_allclose(durations[name], 1 / (1 - MODELS.hmm.transmat_[j, j]))
    assert (durations >= MIN_DURATION).all()


@pytest.mark.parametrize("t", T_BARS)
@pytest.mark.parametrize("method", ["rules", "kmeans", "hmm"])
def test_regime_label_does_not_change_with_future_data(method, t):
    """Prueba obligatoria 4 del PDF (seccion 3.7): el regimen en t no usa el futuro.

    Con los modelos ya ajustados, la etiqueta en t es la misma si se calcula
    con la serie completa o solo con df.iloc[:t+1]. Se revisa para reglas,
    K-means y HMM filtrado.
    """
    full = FULL_LABELS.iloc[t][method]
    truncated = TRUNCATED_LABELS[t][method]
    assert (pd.isna(full) and pd.isna(truncated)) or full == truncated


def test_labels_only_change_on_the_hour():
    """La etiqueta se actualiza en las barras hh:00 y se mantiene el resto de la hora."""
    labels = FULL_LABELS["hmm"].dropna()
    changed = labels != labels.shift()
    assert (changed.iloc[1:][labels.index[1:].minute != 0] == False).all()   # noqa: E712
    assert changed.iloc[1:].sum() > 0


def test_canary_viterbi_has_lookahead():
    """Viterbi usa el futuro: al truncar en la hora k, su etiqueta en k cambia en al menos una k."""
    full = MODELS.hmm.predict(X_SCALED)
    changed = [k for k in range(0, len(X_SCALED), 3)
               if MODELS.hmm.predict(X_SCALED[:k + 1])[-1] != full[k]]
    assert changed


# Validacion del regimen (antes tests/test_regime_analysis.py)

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


def test_viterbi_lookahead_check_filtered_never_changes():
    """Con pocas horas: la etiqueta filtrada del HMM no cambia al truncar."""
    check = viterbi_lookahead_check(DF, MODELS, step=150)
    assert len(check) >= 5
    assert check["hmm_same"].all()


def test_regime_validation_matches_comparison_table():
    """regime_validation reusa comparison_table y marca los objetivos del PDF (> 12 h, silhouette > 0.4)."""
    periods = {"a": ("2022-06-01", "2022-07-31"), "b": ("2022-08-01", "2022-09-15")}
    table = regime_validation(DF, MODELS, periods)
    expected = comparison_table(FULL_LABELS, FULL_FEATURES, MODELS.scaler, periods)

    assert list(table.index) == list(expected.index)
    np.testing.assert_allclose(table["mean_duration_hours"], expected["mean_duration_all"])
    np.testing.assert_allclose(table["transitions_per_month"], expected["transitions_per_month"])
    np.testing.assert_allclose(table["silhouette"], expected["silhouette"])
    assert (table["duration_ok"] == (table["mean_duration_hours"] > 12)).all()
    assert (table["silhouette_ok"] == (table["silhouette"] > 0.4)).all()


def test_rule_regimes_ignore_data_after_fit_end():
    """Con ventana expansiva, cambiar los precios despues de fit_end no cambia el umbral
    ni las etiquetas hasta fit_end (el ajuste no ve el futuro)."""
    fit_end = "2022-08-31"
    labels, threshold = rule_regimes(DF, fit_end)

    corrupted = DF.copy()
    after = corrupted.index > corrupted.loc[:fit_end].index[-1]
    corrupted.loc[after, "Close"] *= np.linspace(0.5, 2.0, after.sum())
    labels_corrupted, threshold_corrupted = rule_regimes(corrupted, fit_end)

    assert threshold_corrupted == threshold
    pd.testing.assert_series_equal(labels.loc[:fit_end], labels_corrupted.loc[:fit_end])
    # La corrupcion si cambia algo despues de fit_end (la prueba no es trivial).
    assert not labels.loc[fit_end:].equals(labels_corrupted.loc[fit_end:])


def test_rule_regimes_threshold_uses_data_up_to_fit_end():
    """El umbral es el p90 de la volatilidad horaria desde el inicio de df hasta fit_end."""
    fit_end = "2022-08-31"
    _, threshold = rule_regimes(DF, fit_end)
    assert threshold == pytest.approx(HOURLY_FEATURES.loc[:fit_end, "volatility"].quantile(0.9))
