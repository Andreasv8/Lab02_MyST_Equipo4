"""Features y clasificadores de regimen de mercado de lab_02 (Act 07).

Tres features con ventana movil de WINDOW barras de 5 minutos (una semana),
calculadas en cada barra. Todas son causales: el valor en la barra t solo usa
datos hasta t (ver tests/test_regimes.py).

Los clasificadores trabajan en las barras hh:00 (hourly): se ajustan y
etiquetan una vez por hora, y la etiqueta se mantiene el resto de la hora.

El escalado se ajusta SOLO con train (fit_scaler) y se aplica igual a
cualquier periodo (apply_scaler).

Tres clasificadores (reglas, K-means, HMM) ajustados SOLO con train que
etiquetan cada barra por nombre: "crisis", "trend" o "mean_reversion".
Las etiquetas filtradas son causales; Viterbi se incluye solo para comparar.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from src.metrics import BARS_PER_DAY, PERIODS_PER_YEAR
from src.splits import get_split

TRADING_DAYS = 252
WINDOW = 7 * BARS_PER_DAY               # una semana de barras de 5 minutos
UPDATE_MINUTE = 0                       # el regimen se actualiza en la barra hh:00 (cada hora)
REGIME_COLUMNS = ["volatility", "trend_r2", "autocorr_1"]
REGIME_NAMES = ["crisis", "trend", "mean_reversion"]
SEED = 42
N_REGIMES = 3

MIN_DURATION = 12                       # pasos del HMM = horas
HMM_SEEDS = range(10)
TRANSMAT_PRIOR_WEIGHT = 50


def log_returns(close: pd.Series) -> pd.Series:
    """Rendimientos logaritmicos: r_t = ln(C_t / C_{t-1}). NaN en la primera barra."""
    return np.log(close / close.shift(1))


def rolling_volatility(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """Volatilidad anualizada de los rendimientos log en la ventana.

    sigma_t = std(r_{t-N+1}, ..., r_t) * sqrt(PERIODS_PER_YEAR), con ddof=1.

    Por que: el nivel de volatilidad es la variable que mas separa regimenes
    (calma vs estres); cambia el tamaño de los movimientos y el riesgo por trade.
    """
    return log_returns(close).rolling(window=window).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def rolling_trend_r2(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """R^2 de la regresion lineal del log-precio contra el tiempo en la ventana.

    En los N log-precios y = ln(C) de la ventana, ajusta y = a + b*x con
    x = 0..N-1 y regresa R^2 = 1 - SS_res/SS_tot = corr(x, y)^2, en [0, 1].

    Por que: mide la FUERZA de la tendencia sin importar su direccion.
    Cerca de 1 = el precio avanza en linea recta (tendencia limpia);
    cerca de 0 = se mueve de lado o con ruido. Se usa log-precio para que una
    tendencia de crecimiento porcentual constante sea lineal.

    Se calcula con rolling().corr() (vectorizado): corr(x, y) no cambia si x se
    desplaza, asi que x = posicion de la barra. y se centra en el primer
    log-precio para reducir error numerico; es causal (solo usa la barra 0).
    """
    y = np.log(close)
    y = y - y.iloc[0]
    x = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    return y.rolling(window=window).corr(x) ** 2


def rolling_autocorr_1(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """Autocorrelacion lag-1 de los rendimientos log en la ventana.

    Con los N rendimientos de la ventana, rho_1 = corr(r_{k-1}, r_k) sobre los
    N-1 pares consecutivos (misma definicion que pd.Series.autocorr), en [-1, 1].

    Por que: mide si los movimientos se revierten o persisten.
    Negativa = reversion a la media (a un dia de subida le sigue uno de bajada);
    positiva = persistencia/momentum de corto plazo.

    Vectorizado: corr de (r_k, r_{k-1}) en ventanas de N-1 pares.
    """
    r = log_returns(close)
    return r.rolling(window=window - 1).corr(r.shift(1))


def regime_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Calcula las 3 features de regimen sobre un DataFrame de precios.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir la columna "Close".
    window : int
        Tamaño de la ventana movil en barras (WINDOW por defecto).

    Regresa
    -------
    pd.DataFrame
        Columnas volatility, trend_r2, autocorr_1, mismo indice que df.
        NaN en el warm-up (no se dropean, para que la fila t corresponda a df.iloc[t]).
    """
    close = df["Close"]
    return pd.DataFrame({
        "volatility": rolling_volatility(close, window),
        "trend_r2": rolling_trend_r2(close, window),
        "autocorr_1": rolling_autocorr_1(close, window),
    }, index=df.index)


def hourly(features: pd.DataFrame) -> pd.DataFrame:
    """Filas de las barras hh:00: el paso de tiempo de los clasificadores.

    Por que: con barras de 5 minutos el HMM cambiaria de estado por ruido;
    a una hora por paso las duraciones esperadas son interpretables (el
    Lab 02 pide actualizar la clasificacion cada 1 a 6 horas). Se usa el
    reloj (minuto UPDATE_MINUTE) y no "cada 12 filas" para que la barra
    elegida no dependa de donde empieza el DataFrame ni de los huecos.
    """
    return features[features.index.minute == UPDATE_MINUTE]


def to_bars(labels: pd.Series, index: pd.Index) -> pd.Series:
    """Lleva etiquetas horarias a todas las barras: cada barra toma la de la ultima hh:00 <= t.

    La etiqueta de las 10:00 se usa de 10:00 a 10:55 y cambia hasta las 11:00.
    Es causal: la barra t solo ve la etiqueta de una hora ya cerrada en t.
    """
    return labels.reindex(index, method="ffill")


@dataclass(frozen=True)
class RegimeScaler:
    """Media y desviacion por columna, estimadas solo con train."""

    mean: pd.Series
    std: pd.Series


def fit_scaler(features_train: pd.DataFrame) -> RegimeScaler:
    """Estima media y desviacion estandar (ddof=0) de cada feature con datos de train.

    Por que solo train: si la media y la desviacion se calculan con todo el
    dataset, el valor escalado en train ya contiene informacion de test y
    validation (look-ahead). Los NaN de warm-up se ignoran.

    Parametros
    ----------
    features_train : pd.DataFrame
        Features del periodo de train.

    Regresa
    -------
    RegimeScaler
        Media y desviacion por columna.
    """
    return RegimeScaler(mean=features_train.mean(), std=features_train.std(ddof=0))


def apply_scaler(features: pd.DataFrame, scaler: RegimeScaler) -> pd.DataFrame:
    """Estandariza features con los parametros de train: z = (x - media_train) / std_train.

    Parametros
    ----------
    features : pd.DataFrame
        Features de cualquier periodo (mismas columnas que en fit_scaler).
    scaler : RegimeScaler
        Resultado de fit_scaler sobre train.

    Regresa
    -------
    pd.DataFrame
        Features estandarizadas, mismo indice y columnas.
    """
    return (features - scaler.mean) / scaler.std


# ---------------------------------------------------------------------------
# Utilidades comunes de los clasificadores
# ---------------------------------------------------------------------------

def name_states(centers: pd.DataFrame) -> dict[int, str]:
    """Asigna nombre a cada estado/cluster segun su centro en unidades ORIGINALES.

    Regla: el de mayor volatility = "crisis"; de los otros dos, el de mayor
    trend_r2 = "trend"; el restante = "mean_reversion". Asi el nombre no
    depende del ID que asigne el algoritmo (que cambia con la semilla).

    Parametros
    ----------
    centers : pd.DataFrame
        Una fila por estado (indice = ID), columnas volatility y trend_r2.

    Regresa
    -------
    dict[int, str]
        ID -> nombre del regimen.
    """
    crisis = centers["volatility"].idxmax()
    rest = centers.drop(index=crisis)
    trend = rest["trend_r2"].idxmax()
    mean_reversion = rest.drop(index=trend).index[0]
    return {int(crisis): "crisis", int(trend): "trend", int(mean_reversion): "mean_reversion"}


def _unscale(centers_scaled: np.ndarray, scaler: RegimeScaler) -> pd.DataFrame:
    """Regresa centros escalados a unidades originales: x = z * std_train + media_train."""
    centers = pd.DataFrame(centers_scaled, columns=scaler.mean.index)
    return centers * scaler.std + scaler.mean


def _valid_scaled(features: pd.DataFrame, scaler: RegimeScaler) -> pd.DataFrame:
    """Filas sin NaN (fuera del warm-up), escaladas con los parametros de train."""
    return apply_scaler(features.dropna(), scaler)


def _to_labels(features: pd.DataFrame, valid_index: pd.Index, ids: np.ndarray,
               names: dict[int, str]) -> pd.Series:
    """Convierte IDs en nombres sobre el indice completo; NaN donde no hay features."""
    labels = pd.Series(np.nan, index=features.index, dtype=object)
    labels.loc[valid_index] = [names[int(i)] for i in ids]
    return labels


# ---------------------------------------------------------------------------
# 1. Reglas
# ---------------------------------------------------------------------------

def fit_rule_thresholds(features_train: pd.DataFrame) -> float:
    """Umbral de crisis: percentil 90 de volatility en train (ignora NaN)."""
    return float(features_train["volatility"].quantile(0.9))


def classify_rules(features: pd.DataFrame, vol_threshold: float,
                   trend_threshold: float = 0.5) -> pd.Series:
    """Clasificador por reglas fijas, evaluadas en orden:

    1. crisis          si volatility_t > p90(volatility en train)
    2. trend           si no es crisis y trend_r2_t > 0.5
    3. mean_reversion  en cualquier otro caso

    Es punto a punto (solo usa la fila t), asi que es causal. Barras con
    alguna feature NaN (warm-up) quedan NaN.

    Parametros
    ----------
    features : pd.DataFrame
        Salida de regime_features.
    vol_threshold : float
        Umbral de volatilidad (fit_rule_thresholds sobre train).
    trend_threshold : float
        Umbral de trend_r2 para "trend".

    Regresa
    -------
    pd.Series
        Nombre del regimen por barra, mismo indice que features.
    """
    labels = pd.Series("mean_reversion", index=features.index, dtype=object)
    labels[features["trend_r2"] > trend_threshold] = "trend"
    labels[features["volatility"] > vol_threshold] = "crisis"
    labels[features[REGIME_COLUMNS].isna().any(axis=1)] = np.nan
    return labels


# ---------------------------------------------------------------------------
# 2. K-means
# ---------------------------------------------------------------------------

def _kmeans(n_clusters: int, random_state: int) -> KMeans:
    """KMeans con los parametros del lab (k-means++, 10 inicializaciones)."""
    return KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)


def fit_kmeans(features_train: pd.DataFrame, scaler: RegimeScaler,
               random_state: int = SEED) -> tuple[KMeans, dict[int, str]]:
    """Ajusta K-means (k=3) sobre las features escaladas de train.

    K-means parte el espacio de features en 3 grupos minimizando la distancia
    cuadrada de cada punto a su centroide. Se escala antes para que ninguna
    feature domine la distancia por sus unidades. Los clusters se nombran con
    name_states usando los centroides regresados a unidades originales.

    Regresa
    -------
    tuple[KMeans, dict[int, str]]
        Modelo ajustado y mapa ID -> nombre.
    """
    model = _kmeans(N_REGIMES, random_state).fit(_valid_scaled(features_train, scaler).to_numpy())
    return model, name_states(_unscale(model.cluster_centers_, scaler))


def classify_kmeans(features: pd.DataFrame, scaler: RegimeScaler, model: KMeans,
                    names: dict[int, str]) -> pd.Series:
    """Asigna cada barra al centroide mas cercano. Punto a punto, por lo tanto causal."""
    x = _valid_scaled(features, scaler)
    ids = model.predict(x.to_numpy()) if len(x) else np.array([], dtype=int)
    return _to_labels(features, x.index, ids, names)


def kmeans_elbow(features_train: pd.DataFrame, scaler: RegimeScaler,
                 k_values=range(1, 9)) -> pd.Series:
    """Curva del codo: inercia (suma de distancias cuadradas al centroide) en train por k.

    Regresa
    -------
    pd.Series
        Inercia indexada por k.
    """
    x = _valid_scaled(features_train, scaler).to_numpy()
    return pd.Series({k: _kmeans(k, SEED).fit(x).inertia_ for k in k_values}, name="inertia")


# ---------------------------------------------------------------------------
# 3. HMM
# ---------------------------------------------------------------------------

def hmm_expected_durations(model: GaussianHMM, names: dict[int, str]) -> pd.Series:
    """Duracion esperada de cada estado en barras: 1 / (1 - a_jj).

    La permanencia en el estado j es geometrica con probabilidad a_jj de
    quedarse, asi que su media es 1 / (1 - a_jj).

    Regresa
    -------
    pd.Series
        Duracion esperada indexada por nombre de regimen.
    """
    stay = np.diag(model.transmat_)
    return pd.Series({names[j]: 1 / (1 - stay[j]) for j in range(len(stay))}, name="expected_duration")


def _fit_hmm_best_seed(x_train: np.ndarray, covariance_type: str,
                       transmat_prior: Optional[np.ndarray] = None) -> GaussianHMM:
    """Ajusta el HMM con cada semilla de HMM_SEEDS y regresa el de mayor log-likelihood en train.

    En empate se queda la semilla menor.
    """
    extra = {} if transmat_prior is None else {"transmat_prior": transmat_prior}
    best_model, best_score = None, -np.inf
    for seed in HMM_SEEDS:
        model = GaussianHMM(n_components=N_REGIMES, covariance_type=covariance_type,
                            n_iter=200, random_state=seed, **extra).fit(x_train)
        score = model.score(x_train)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def _durations_ok(model: GaussianHMM) -> bool:
    """True si todas las duraciones esperadas son >= MIN_DURATION pasos (horas)."""
    return bool((1 / (1 - np.diag(model.transmat_)) >= MIN_DURATION).all())


def fit_hmm(features_train: pd.DataFrame,
            scaler: RegimeScaler) -> tuple[GaussianHMM, dict[int, str], str]:
    """Ajusta un HMM gaussiano de 3 estados sobre las features escaladas de train.

    El HMM supone un estado oculto s_t (el regimen) que sigue una cadena de
    Markov con matriz de transicion A, y que en cada estado las features se
    distribuyen normal multivariada con media y covarianza propias.

    Seleccion del modelo (solo con criterios de train, nunca con rendimientos):
    EM converge a optimos locales; con una sola semilla (42) el modelo quedaba
    con duraciones esperadas de ~1 barra en dos estados, lo cual es ruido y no
    un regimen. Por eso se usa una cascada:

    1. covariance_type="full", 10 semillas (0..9), se queda el de mayor
       log-likelihood en train (model.score). Si todas las duraciones
       esperadas son >= MIN_DURATION horas, se usa ese.
    2. Si no, covariance_type="diag" con el mismo criterio.
    3. Si no, "diag" + transmat_prior Dirichlet = 1 + 50*I: supuesto a priori
       de que los regimenes persisten (mas peso en la diagonal de A).

    features_train debe venir ya en pasos horarios (hourly).

    Regresa
    -------
    tuple[GaussianHMM, dict[int, str], str]
        Modelo, mapa ID -> nombre (name_states sobre las medias en unidades
        originales) y la opcion elegida: "full", "diag" o "diag+prior".
    """
    x_train = _valid_scaled(features_train, scaler).to_numpy()

    model, choice = _fit_hmm_best_seed(x_train, "full"), "full"
    if not _durations_ok(model):
        model, choice = _fit_hmm_best_seed(x_train, "diag"), "diag"
    if not _durations_ok(model):
        prior = np.ones((N_REGIMES, N_REGIMES)) + TRANSMAT_PRIOR_WEIGHT * np.eye(N_REGIMES)
        model, choice = _fit_hmm_best_seed(x_train, "diag", prior), "diag+prior"

    return model, name_states(_unscale(model.means_, scaler)), choice


def hmm_seed_scan(features_train: pd.DataFrame, scaler: RegimeScaler, seeds,
                  covariance_type: str = "full") -> pd.DataFrame:
    """Ajusta el HMM con cada semilla y resume log-likelihood y persistencia en train.

    Sirve para documentar la seleccion de fit_hmm: EM llega a optimos locales
    distintos segun la semilla; algunos tienen estados de ~1 barra (ruido).

    Parametros
    ----------
    features_train : pd.DataFrame
        Features de train.
    scaler : RegimeScaler
        Ajustado en train.
    seeds : iterable de int
        Semillas a probar.
    covariance_type : str
        "full" o "diag".

    Regresa
    -------
    pd.DataFrame
        Indice = semilla; columnas log_likelihood, a_11, a_22, a_33 (diagonal de
        A en el orden de IDs del modelo) y min_duration = min_j 1/(1 - a_jj).
    """
    x_train = _valid_scaled(features_train, scaler).to_numpy()
    rows = {}
    for seed in seeds:
        model = GaussianHMM(n_components=N_REGIMES, covariance_type=covariance_type,
                            n_iter=200, random_state=seed).fit(x_train)
        stay = np.diag(model.transmat_)
        rows[seed] = {"log_likelihood": model.score(x_train),
                      **{f"a_{j + 1}{j + 1}": stay[j] for j in range(N_REGIMES)},
                      "min_duration": (1 / (1 - stay)).min()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("seed")


def hmm_forward(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """Algoritmo forward: probabilidades FILTRADAS P(s_t | x_1..x_t).

    alpha_1 ∝ pi ⊙ p(x_1 | s)
    alpha_t ∝ (alpha_{t-1} · A) ⊙ p(x_t | s),   normalizado para sumar 1 en cada t

    Se calcula en log-espacio (logsumexp) para evitar underflow. La fila t
    solo usa x_1..x_t, por lo que la etiqueta argmax es causal.

    Parametros
    ----------
    model : GaussianHMM
        HMM ya ajustado (startprob_, transmat_, means_, covars_).
    x : np.ndarray
        Observaciones escaladas, forma (T, n_features).

    Regresa
    -------
    np.ndarray
        Forma (T, n_estados); cada fila suma 1.
    """
    n_states = model.n_components
    log_emission = np.column_stack([
        multivariate_normal(model.means_[k], model.covars_[k]).logpdf(x).reshape(-1)
        for k in range(n_states)
    ]) if len(x) else np.empty((0, n_states))

    with np.errstate(divide="ignore"):
        log_start, log_trans = np.log(model.startprob_), np.log(model.transmat_)

    log_alpha = np.empty((len(x), n_states))
    for t in range(len(x)):
        if t == 0:
            prior = log_start
        else:
            prior = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
        unnormalized = prior + log_emission[t]
        log_alpha[t] = unnormalized - logsumexp(unnormalized)
    return np.exp(log_alpha)


def classify_hmm_filtered(features: pd.DataFrame, scaler: RegimeScaler, model: GaussianHMM,
                          names: dict[int, str]) -> pd.Series:
    """Etiqueta filtrada: argmax_s P(s_t | x_1..x_t) con hmm_forward. Causal.

    El forward arranca en la primera barra con features, que es la misma con
    la serie completa y con la truncada.
    """
    x = _valid_scaled(features, scaler)
    ids = hmm_forward(model, x.to_numpy()).argmax(axis=1)
    return _to_labels(features, x.index, ids, names)


def classify_hmm_viterbi(features: pd.DataFrame, scaler: RegimeScaler, model: GaussianHMM,
                         names: dict[int, str]) -> pd.Series:
    """Etiqueta Viterbi: secuencia mas probable s_1..s_T dado TODO x_1..x_T (model.predict).

    Tiene look-ahead: Viterbi maximiza P(s_1..s_T | x_1..x_T) y reconstruye la
    secuencia con backtracking desde la ultima barra T hacia atras, asi que el
    estado elegido en t depende de x_{t+1..T}. Agregar barras nuevas puede
    cambiar etiquetas pasadas. Solo sirve para comparar, nunca para operar.
    """
    x = _valid_scaled(features, scaler)
    ids = model.predict(x.to_numpy()) if len(x) else np.array([], dtype=int)
    return _to_labels(features, x.index, ids, names)


# ---------------------------------------------------------------------------
# 4. Orquestacion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeModels:
    """Todo lo ajustado en train para etiquetar regimenes."""

    scaler: RegimeScaler
    vol_threshold: float
    kmeans: KMeans
    kmeans_names: dict
    hmm: GaussianHMM
    hmm_names: dict
    hmm_choice: str


def fit_regime_models(df: pd.DataFrame) -> RegimeModels:
    """Ajusta scaler, umbral de reglas, K-means y HMM SOLO con las barras de train.

    Las features se calculan sobre df (son causales), se recortan a train con
    src/splits.py y se toman solo las barras hh:00 (hourly).

    Parametros
    ----------
    df : pd.DataFrame
        Precios con columna "Close"; debe cubrir el periodo de train.

    Regresa
    -------
    RegimeModels
        Modelos y parametros ajustados.
    """
    features_train = hourly(get_split(regime_features(df), "train"))
    scaler = fit_scaler(features_train)
    kmeans, kmeans_names = fit_kmeans(features_train, scaler)
    hmm, hmm_names, hmm_choice = fit_hmm(features_train, scaler)
    return RegimeModels(scaler, fit_rule_thresholds(features_train),
                        kmeans, kmeans_names, hmm, hmm_names, hmm_choice)


def regime_labels(df: pd.DataFrame, models: RegimeModels) -> pd.DataFrame:
    """Etiquetas de regimen por barra con los modelos ya ajustados en train.

    Regresa
    -------
    pd.DataFrame
        Indice de df; columnas rules, kmeans, hmm (filtradas, causales) y
        hmm_viterbi (con look-ahead, solo para comparar). Se clasifica en las
        barras hh:00 y la etiqueta se mantiene hasta la siguiente hora.
        NaN en warm-up y antes de la primera hh:00 con features.
    """
    features = hourly(regime_features(df))
    labels = pd.DataFrame({
        "rules": classify_rules(features, models.vol_threshold),
        "kmeans": classify_kmeans(features, models.scaler, models.kmeans, models.kmeans_names),
        "hmm": classify_hmm_filtered(features, models.scaler, models.hmm, models.hmm_names),
        "hmm_viterbi": classify_hmm_viterbi(features, models.scaler, models.hmm, models.hmm_names),
    }, index=features.index)
    return labels.apply(to_bars, index=df.index)
