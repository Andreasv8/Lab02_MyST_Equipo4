"""Comparacion de los clasificadores de regimen de lab_02 (Act 07).

Funciones puras: reciben etiquetas (salida de regimes.regime_labels),
features y modelos ya ajustados en train, y regresan tablas. No leen
archivos ni ajustan nada. Todas ignoran los NaN de warm-up.

Los clasificadores cambian de etiqueta solo en las barras hh:00, asi que las
comparaciones de persistencia y separacion se hacen en pasos horarios.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import silhouette_score

from src.regimes import (
    REGIME_COLUMNS,
    REGIME_NAMES,
    PERIODS_PER_YEAR,
    RegimeModels,
    RegimeScaler,
    apply_scaler,
    hmm_expected_durations,
    hmm_forward,
    hourly,
    name_states,
    regime_features,
)
from src.splits import SPLITS

METHODS = ["rules", "kmeans", "hmm"]
DEFAULT_PERIODS = {"train": SPLITS["train"], "test": SPLITS["test"]}
HOURS_PER_MONTH = 24 * 30


def _clean(labels: Iterable) -> pd.Series:
    """Etiquetas como Series, sin NaN (warm-up)."""
    return pd.Series(labels).dropna()


def regime_runs(labels: Iterable) -> dict[str, list[int]]:
    """Longitudes de las rachas consecutivas de cada regimen, en orden de aparicion.

    Ejemplo: ["a", "a", "b", "b", "b", "a"] -> {"a": [2, 1], "b": [3]}.

    Parametros
    ----------
    labels : lista o pd.Series
        Etiqueta por barra.

    Regresa
    -------
    dict[str, list[int]]
        Regimen -> lista de longitudes de racha (en barras).
    """
    values = _clean(labels).to_numpy()
    runs: dict[str, list[int]] = {}
    if len(values) == 0:
        return runs
    run_id = np.concatenate([[0], np.cumsum(values[1:] != values[:-1])])
    for rid in np.unique(run_id):
        members = values[run_id == rid]
        runs.setdefault(members[0], []).append(len(members))
    return runs


def count_transitions(labels: Iterable) -> int:
    """Numero de cambios de etiqueta entre barras consecutivas."""
    values = _clean(labels).to_numpy()
    return int((values[1:] != values[:-1]).sum())


def regime_shares(labels: Iterable, names: Optional[list[str]] = None) -> pd.Series:
    """Participacion de cada regimen en % de las barras con etiqueta.

    Si se pasan names, todos aparecen en el resultado (0 si no hay barras).
    """
    shares = _clean(labels).value_counts(normalize=True) * 100
    return shares if names is None else shares.reindex(names, fill_value=0.0)


def regime_silhouette(x_scaled: pd.DataFrame, labels: pd.Series) -> float:
    """Silhouette promedio de las etiquetas sobre features escaladas. NaN si hay < 2 regimenes.

    Para cada barra: s = (b - a) / max(a, b), con a = distancia media a las
    barras de su regimen y b = distancia media al regimen mas cercano.
    Cerca de 1 = regimenes compactos y separados; cerca de 0 = se traslapan.
    """
    if labels.nunique() < 2:
        return np.nan
    return float(silhouette_score(x_scaled.to_numpy(), labels.to_numpy()))


def _period_rows(labels_df: pd.DataFrame, features: pd.DataFrame, method: str,
                 start: str, end: str) -> tuple[pd.Series, pd.DataFrame]:
    """Etiquetas de un metodo y features en el periodo, en barras hh:00 con ambas disponibles."""
    labels = hourly(labels_df).loc[start:end, method]
    feats = hourly(features).loc[start:end, REGIME_COLUMNS]
    valid = labels.notna() & feats.notna().all(axis=1)
    return labels[valid], feats[valid]


def comparison_table(labels_df: pd.DataFrame, features: pd.DataFrame, scaler: RegimeScaler,
                     periods: dict = DEFAULT_PERIODS) -> pd.DataFrame:
    """Tabla comparativa de los metodos filtrados por periodo, en pasos horarios (hh:00).

    Columnas, por (metodo, periodo):
    - silhouette: separacion de los regimenes en features ESCALADAS con el
      scaler de train (NaN si hay < 2 regimenes).
    - mean_duration_<regimen> y mean_duration_all: promedio de las rachas en
      horas (las rachas se cortan en el borde del periodo).
    - transitions_per_month: cambios de etiqueta / (horas / HOURS_PER_MONTH).
    - share_<regimen>: % de barras en cada regimen.

    Parametros
    ----------
    labels_df : pd.DataFrame
        Salida de regime_labels (columnas rules, kmeans, hmm).
    features : pd.DataFrame
        Salida de regime_features, mismo indice.
    scaler : RegimeScaler
        Ajustado en train.
    periods : dict
        Nombre -> (inicio, fin).

    Regresa
    -------
    pd.DataFrame
        Indice (method, period).
    """
    rows = {}
    for method in METHODS:
        for period, (start, end) in periods.items():
            labels, feats = _period_rows(labels_df, features, method, start, end)
            runs = regime_runs(labels)
            row = {"silhouette": regime_silhouette(apply_scaler(feats, scaler), labels)}
            for name in REGIME_NAMES:
                row[f"mean_duration_{name}"] = np.mean(runs[name]) if name in runs else np.nan
            row["mean_duration_all"] = np.mean([n for r in runs.values() for n in r])
            row["transitions_per_month"] = count_transitions(labels) / (len(labels) / HOURS_PER_MONTH)
            for name, share in regime_shares(labels, REGIME_NAMES).items():
                row[f"share_{name}"] = share
            rows[(method, period)] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["method", "period"])


def hmm_diagnostics(labels_df: pd.DataFrame, model: GaussianHMM, names: dict[int, str],
                    periods: dict = DEFAULT_PERIODS) -> tuple[pd.DataFrame, pd.Series]:
    """Diagnosticos exclusivos del HMM.

    1. Por regimen: duracion esperada del modelo, 1 / (1 - a_jj), contra la
       duracion media observada de las rachas filtradas en cada periodo
       (ambas en horas: se usan las barras hh:00).
       Si la observada es mucho menor, el forward cambia de estado mas
       seguido de lo que la matriz A sugiere.
    2. Por periodo: % de barras donde la etiqueta filtrada coincide con
       Viterbi (que usa el futuro).

    Regresa
    -------
    tuple[pd.DataFrame, pd.Series]
        (duraciones por regimen, % de acuerdo filtrado == Viterbi por periodo).
    """
    durations = pd.DataFrame({"expected": hmm_expected_durations(model, names)})
    agreement = {}
    for period, (start, end) in periods.items():
        sub = hourly(labels_df).loc[start:end, ["hmm", "hmm_viterbi"]].dropna()
        runs = regime_runs(sub["hmm"])
        durations[f"observed_{period}"] = pd.Series({n: np.mean(r) for n, r in runs.items()})
        agreement[period] = (sub["hmm"] == sub["hmm_viterbi"]).mean() * 100
    return durations.reindex(REGIME_NAMES), pd.Series(agreement, name="filtered_eq_viterbi_pct")


def regime_centroids(labels_df: pd.DataFrame, features: pd.DataFrame,
                     methods: Optional[list[str]] = None) -> pd.DataFrame:
    """Media de cada feature por regimen, en unidades ORIGINALES, por metodo.

    Sirve para justificar los nombres: crisis debe tener la mayor
    volatilidad y trend el mayor trend_r2 entre los no-crisis.

    Parametros
    ----------
    labels_df : pd.DataFrame
        Una columna de etiquetas por metodo (ya recortado al periodo deseado).
    features : pd.DataFrame
        Features en unidades originales, mismo indice.
    methods : list[str], opcional
        Columnas de labels_df a usar (por defecto todas).

    Regresa
    -------
    pd.DataFrame
        Indice (method, regime), una columna por feature.
    """
    methods = list(labels_df.columns) if methods is None else methods
    tables = {m: features.groupby(labels_df[m]).mean() for m in methods}
    return pd.concat(tables, names=["method", "regime"])


def regime_return_profile(df: pd.DataFrame, labels_df: pd.DataFrame,
                          periods: dict = DEFAULT_PERIODS) -> pd.DataFrame:
    """Rendimiento y volatilidad anualizados de BTC por regimen (descriptivo).

    A la etiqueta de t se le asocia el rendimiento simple de la barra siguiente,
    R_{t+1} = Close_{t+1} / Close_t - 1: lo que se gana DESPUES de conocer el
    regimen, y no el rendimiento que ya entro en las features de t.
    ann_return = media(R) * PERIODS_PER_YEAR; ann_volatility = std(R) * sqrt(PERIODS_PER_YEAR).
    La ultima barra no tiene t+1 y se excluye. No se usa para ajustar nada:
    solo muestra si los regimenes son economicamente distintos.

    Regresa
    -------
    pd.DataFrame
        Indice (method, period, regime); columnas n_bars, ann_return, ann_volatility.
    """
    next_return = df["Close"].shift(-1) / df["Close"] - 1
    rows = {}
    for method in METHODS:
        for period, (start, end) in periods.items():
            data = pd.DataFrame({"label": labels_df.loc[start:end, method],
                                 "ret": next_return.loc[start:end]}).dropna()
            for name in REGIME_NAMES:
                r = data.loc[data["label"] == name, "ret"]
                rows[(method, period, name)] = {
                    "n_bars": len(r),
                    "ann_return": r.mean() * PERIODS_PER_YEAR,
                    "ann_volatility": r.std() * np.sqrt(PERIODS_PER_YEAR),
                }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["method", "period", "regime"])


def viterbi_lookahead_check(df: pd.DataFrame, models: RegimeModels, step: int = 5) -> pd.DataFrame:
    """Compara la etiqueta en la hora k con la serie completa vs truncada en k, filtrado y Viterbi.

    Si un metodo es causal, su etiqueta en k no cambia al quitar las horas
    posteriores a k. El HMM filtrado (forward) no debe cambiar nunca; Viterbi
    si puede cambiar porque elige la secuencia completa con backtracking
    desde la ultima hora (look-ahead). Es el canario del notebook.

    Las features se calculan una vez (su causalidad se prueba en
    tests/test_regimes.py) y se trunca la secuencia horaria que ve el HMM.
    Filtrado truncado = ultima fila de predict_proba(x[:k+1]), que es igual
    al forward en k (tests/test_regimes.py).

    Parametros
    ----------
    df : pd.DataFrame
        Precios (hasta fin de test).
    models : RegimeModels
        Modelos ya ajustados en train (no se reajustan).
    step : int
        Se revisa una hora cada step horas, desde la primera con features.

    Regresa
    -------
    pd.DataFrame
        Indice = fecha de la hora k; columnas hmm_same y hmm_viterbi_same (bool).
    """
    x = apply_scaler(hourly(regime_features(df)).dropna(), models.scaler)
    x_np = x.to_numpy()
    full_filtered = hmm_forward(models.hmm, x_np).argmax(axis=1)
    full_viterbi = models.hmm.predict(x_np)
    rows = {}
    for k in range(0, len(x), step):
        rows[x.index[k]] = {
            "hmm_same": models.hmm.predict_proba(x_np[:k + 1])[-1].argmax() == full_filtered[k],
            "hmm_viterbi_same": models.hmm.predict(x_np[:k + 1])[-1] == full_viterbi[k],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _names_consistent(centroids: pd.DataFrame) -> bool:
    """True si name_states sobre estos centroides reproduce los nombres que ya tienen."""
    ordered = centroids.reindex(REGIME_NAMES).reset_index(drop=True)
    if ordered[["volatility", "trend_r2"]].isna().any().any():
        return False
    mapping = name_states(ordered)
    return all(mapping[i] == name for i, name in enumerate(REGIME_NAMES))


def method_scorecard(table: pd.DataFrame, centroids_train: pd.DataFrame,
                     centroids_test: pd.DataFrame) -> pd.DataFrame:
    """Criterios numericos para elegir metodo de regimen, por metodo.

    - persistencia: transiciones/mes y duracion media de rachas en test
      (menos cambios = menos costos de rotacion y regimenes utilizables).
    - estabilidad train -> test: cambio medio absoluto de participacion por
      regimen (puntos porcentuales) y cambio de transiciones/mes.
    - separacion: silhouette en train y test.
    - consistencia interna: la regla de nombres (mayor volatilidad = crisis;
      de los otros, mayor trend_r2 = trend) sigue valiendo con los centroides
      de train y de test.
    La causalidad no aparece: las tres etiquetas filtradas son causales
    (tests de truncamiento).

    Parametros
    ----------
    table : pd.DataFrame
        Salida de comparison_table (indice method, period).
    centroids_train, centroids_test : pd.DataFrame
        Salida de regime_centroids en cada periodo (indice method, regime).

    Regresa
    -------
    pd.DataFrame
        Una fila por metodo.
    """
    rows = {}
    share_cols = [f"share_{n}" for n in REGIME_NAMES]
    for method in table.index.get_level_values("method").unique():
        train, test = table.loc[(method, "train")], table.loc[(method, "test")]
        rows[method] = {
            "transitions_per_month_test": test["transitions_per_month"],
            "mean_duration_test": test["mean_duration_all"],
            "share_shift_pp": (test[share_cols] - train[share_cols]).abs().mean(),
            "transitions_change": test["transitions_per_month"] - train["transitions_per_month"],
            "silhouette_train": train["silhouette"],
            "silhouette_test": test["silhouette"],
            "names_consistent_train": _names_consistent(centroids_train.loc[method]),
            "names_consistent_test": _names_consistent(centroids_test.loc[method]),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("method")
