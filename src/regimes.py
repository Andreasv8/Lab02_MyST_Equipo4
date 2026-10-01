"""Features de regimen de mercado de lab_02 (Act 07).

Tres features con ventana movil de 63 dias (~un trimestre de sesiones),
recalculadas a diario. Todas son causales: el valor en la barra t solo usa
datos hasta t (ver tests/test_regimes.py).

El escalado se ajusta SOLO con train (fit_scaler) y se aplica igual a
cualquier periodo (apply_scaler).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
REGIME_COLUMNS = ["volatility", "trend_r2", "autocorr_1"]


def log_returns(close: pd.Series) -> pd.Series:
    """Rendimientos logaritmicos: r_t = ln(C_t / C_{t-1}). NaN en la primera barra."""
    return np.log(close / close.shift(1))


def rolling_volatility(close: pd.Series, window: int = 63) -> pd.Series:
    """Volatilidad anualizada de los rendimientos log en la ventana.

    sigma_t = std(r_{t-N+1}, ..., r_t) * sqrt(252), con ddof=1.

    Por que: el nivel de volatilidad es la variable que mas separa regimenes
    (calma vs estres); cambia el tamaño de los movimientos y el riesgo por trade.
    """
    return log_returns(close).rolling(window=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def _r2_vs_time(y: np.ndarray) -> float:
    """R^2 de la regresion y = a + b*x con x = 0..n-1; R^2 = corr(x, y)^2. NaN si var(y) = 0."""
    x = np.arange(len(y), dtype=float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    ss_y = np.dot(y_c, y_c)
    if ss_y == 0:
        return np.nan
    return np.dot(x_c, y_c) ** 2 / (np.dot(x_c, x_c) * ss_y)


def rolling_trend_r2(close: pd.Series, window: int = 63) -> pd.Series:
    """R^2 de la regresion lineal del log-precio contra el tiempo en la ventana.

    En los N log-precios y = ln(C) de la ventana, ajusta y = a + b*x con
    x = 0..N-1 y regresa R^2 = 1 - SS_res/SS_tot = corr(x, y)^2, en [0, 1].

    Por que: mide la FUERZA de la tendencia sin importar su direccion.
    Cerca de 1 = el precio avanza en linea recta (tendencia limpia);
    cerca de 0 = se mueve de lado o con ruido. Se usa log-precio para que una
    tendencia de crecimiento porcentual constante sea lineal.
    """
    return np.log(close).rolling(window=window).apply(_r2_vs_time, raw=True)


def _lag1_corr(r: np.ndarray) -> float:
    """Correlacion de Pearson entre r[:-1] y r[1:]. NaN si alguno tiene varianza 0."""
    a, b = r[:-1], r[1:]
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    if denom == 0:
        return np.nan
    return np.dot(a_c, b_c) / denom


def rolling_autocorr_1(close: pd.Series, window: int = 63) -> pd.Series:
    """Autocorrelacion lag-1 de los rendimientos log en la ventana.

    Con los N rendimientos de la ventana, rho_1 = corr(r_{k-1}, r_k) sobre los
    N-1 pares consecutivos (misma definicion que pd.Series.autocorr), en [-1, 1].

    Por que: mide si los movimientos se revierten o persisten.
    Negativa = reversion a la media (a un dia de subida le sigue uno de bajada);
    positiva = persistencia/momentum de corto plazo.
    """
    return log_returns(close).rolling(window=window).apply(_lag1_corr, raw=True)


def regime_features(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Calcula las 3 features de regimen sobre un DataFrame de precios.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir la columna "Close".
    window : int
        Tamaño de la ventana movil en barras (63 por defecto).

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
