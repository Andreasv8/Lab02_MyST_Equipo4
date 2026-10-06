"""Indicadores tecnicos, votos y señal de la estrategia de lab_02 (docs/SPEC.md).

Los indicadores se calculan a mano con pandas/numpy.

Todas las funciones son causales: el valor en la barra t solo usa datos
hasta t (ver tests/test_causality.py).
"""

from typing import Optional

import pandas as pd

from src.data import align_to_base, resample_ohlc


def sma(series: pd.Series, window: int) -> pd.Series:
    """Media movil simple sobre una serie."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Media movil exponencial sobre una serie (alpha = 2/(span+1))."""
    return series.ewm(span=span, adjust=False).mean()


def bollinger_percent_b(close: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
    """%B de Bollinger: posicion relativa del precio dentro de las bandas.

    %B = (Close - banda_inferior) / (banda_superior - banda_inferior).
    0 = toca la banda inferior, 1 = toca la banda superior.
    """
    mid = sma(close, window)
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (close - lower) / (upper - lower)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: maximo entre High-Low y la distancia de High/Low al cierre previo."""
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR (Average True Range) con suavizado de Wilder."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """ROC (Rate of Change): variacion porcentual respecto a N periodos atras."""
    return ((close - close.shift(window)) / close.shift(window)) * 100


def directional_indicators(high: pd.Series, low: pd.Series, close: pd.Series,
                           window: int = 14) -> tuple[pd.Series, pd.Series]:
    """+DI y -DI de Wilder: la DIRECCION del movimiento, en [0, 100].

    +DI > -DI = dominan las subidas; -DI > +DI = dominan las bajadas.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1 / window
    smoothed_tr = true_range(high, low, close).ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / smoothed_tr
    return plus_di, minus_di


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ADX con suavizado de Wilder: FUERZA de la tendencia (0-100), no su direccion."""
    plus_di, minus_di = directional_indicators(high, low, close, window)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


# Lab 02, seccion 3.1: al menos 2 de los 3 indicadores deben coincidir en direccion.
MIN_VOTES = 2


def confirmation_signal(votes: pd.DataFrame, min_votes: int = MIN_VOTES) -> pd.Series:
    """Regla de confirmacion 2 de 3 (Lab 02, seccion 3.2).

    L_t = indicadores con voto +1;  S_t = indicadores con voto -1
    señal_t = +1 si L_t >= 2;  -1 si S_t >= 2;  0 en otro caso.
    Con un solo indicador a favor NO se abre posicion.
    """
    n_long = (votes == 1).sum(axis=1)
    n_short = (votes == -1).sum(axis=1)
    signal = pd.Series(0, index=votes.index, dtype=int)
    signal.loc[n_long >= min_votes] = 1
    signal.loc[n_short >= min_votes] = -1
    return signal


# ---------------------------------------------------------------------------
# Estrategia final (docs/SPEC.md): tres votos en velas de 4h, filtro ADX y
# entrada solo cuando cambia el estado.
# ---------------------------------------------------------------------------

# Los indicadores se calculan en velas de 4 horas (ver docs/SPEC.md, seccion 1).
TIMEFRAME = "4h"

# Parametros iniciales θ0, antes de optimizar (docs/SPEC.md, seccion 8).
THETA0 = {
    "ema_fast": 12,
    "ema_slow": 48,
    "roc_window": 12,
    "bb_window": 20,
    "bb_std": 2,
    "bb_threshold": 0.7,
    "adx_window": 14,
    "adx_threshold": 20,
    "atr_window": 14,
    "sl_mult": 2.0,
    "rr": 3.0,
    "max_holding": 2016,    # barras de 5 min = 7 dias
    "rho": 0.01,
}


def _sign(values: pd.Series) -> pd.Series:
    """+1 si el valor es positivo, -1 si es negativo y 0 si es cero o NaN."""
    return (values > 0).astype(int) - (values < 0).astype(int)


def strategy_votes(ema_fast: pd.Series, ema_slow: pd.Series, roc_values: pd.Series,
                   percent_b: pd.Series, bb_threshold: float) -> pd.DataFrame:
    """Calcula el voto de cada indicador: +1 (largo), -1 (corto) o 0 (no vota).

    vote_ema = signo(EMA_rapida - EMA_lenta)              tendencia
    vote_roc = signo(ROC)                                 momento
    vote_bb  = +1 si %B > u; -1 si %B < 1 - u; 0 si no    volatilidad

    Recibe las series de cada indicador (mismo indice) y el umbral u de %B.
    Regresa un DataFrame con las columnas vote_ema, vote_roc y vote_bb.
    Los NaN del calentamiento no votan.
    """
    vote_bb = (percent_b > bb_threshold).astype(int) - (percent_b < 1 - bb_threshold).astype(int)
    return pd.DataFrame({
        "vote_ema": _sign(ema_fast - ema_slow),
        "vote_roc": _sign(roc_values),
        "vote_bb": vote_bb,
    }, index=ema_fast.index)


def confirmed_state(votes: pd.DataFrame, adx_values: pd.Series, adx_threshold: float,
                    min_votes: int = MIN_VOTES) -> pd.Series:
    """Aplica la regla 2 de 3 con el filtro de ADX.

    estado = +1 si hay al menos min_votes votos +1 y ADX > umbral;
    estado = -1 si hay al menos min_votes votos -1 y ADX > umbral;
    estado = 0 en otro caso. El ADX no vota: solo dice si hay tendencia fuerte.

    Recibe los votos, el ADX y su umbral. Regresa el estado (+1, -1 o 0).
    """
    state = confirmation_signal(votes, min_votes)
    return state.where(adx_values > adx_threshold, 0).astype(int)


def entry_signal(state: pd.Series) -> pd.Series:
    """Deja la señal solo en la barra donde cambia el estado.

    señal_t = estado_t si estado_t != estado_(t-1); 0 si no cambio.
    Asi no se vuelve a entrar en cada barra mientras dura la misma tendencia.

    Recibe el estado por barra. Regresa la señal de entrada (+1, -1 o 0).
    """
    previous = state.shift(1, fill_value=0)
    return state.where(state != previous, 0).astype(int)


# Columnas de voto que puede usar la opcion de un solo voto (analisis de robustez).
SINGLE_VOTES = {"ema": "vote_ema", "roc": "vote_roc", "bb": "vote_bb"}


def strategy_features_4h(df: pd.DataFrame, params: dict = THETA0,
                         single_vote: Optional[str] = None) -> pd.DataFrame:
    """Indicadores, votos y estado de la estrategia en velas de 4h (antes de pasar a 5 min).

    Con single_vote ("ema", "roc" o "bb") el estado usa solo ese voto: basta
    1 voto a favor, con el mismo filtro de ADX. Sirve para comparar un
    indicador solo contra la regla 2 de 3 (docs/SPEC.md, seccion 14.3).

    Recibe las velas de 5 min, los parametros θ y, opcional, el voto unico.
    Regresa un DataFrame indexado por el inicio de cada vela de 4h.
    """
    # 1. Indicadores en velas de 4h.
    bars = resample_ohlc(df, TIMEFRAME)
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    features = pd.DataFrame({
        "ema_fast": ema(close, params["ema_fast"]),
        "ema_slow": ema(close, params["ema_slow"]),
        "roc": roc(close, params["roc_window"]),
        "percent_b": bollinger_percent_b(close, params["bb_window"], params["bb_std"]),
        "adx": adx(high, low, close, params["adx_window"]),
        "atr": atr(high, low, close, params["atr_window"]),
    })
    # Sin voto de EMA hasta que la EMA lenta tenga historia suficiente.
    features.loc[features.index[:params["ema_slow"] - 1], ["ema_fast", "ema_slow"]] = float("nan")

    # 2. Votos y estado, tambien en 4h (regla 2 de 3, o un solo voto).
    votes = strategy_votes(features["ema_fast"], features["ema_slow"], features["roc"],
                           features["percent_b"], params["bb_threshold"])
    features = features.join(votes)
    if single_vote is None:
        features["state"] = confirmed_state(votes, features["adx"], params["adx_threshold"])
    else:
        only = votes[[SINGLE_VOTES[single_vote]]]
        features["state"] = confirmed_state(only, features["adx"], params["adx_threshold"], min_votes=1)
    return features


def compute_strategy(df: pd.DataFrame, params: dict = THETA0,
                     single_vote: Optional[str] = None) -> pd.DataFrame:
    """Calcula indicadores, votos, estado y señal de la estrategia final.

    Los indicadores se calculan en velas de 4h y se pasan a 5 min con
    align_to_base: una vela de 4h solo se usa cuando ya cerro (sin look-ahead).
    La señal se toma al cierre de la barra t de 5 min; el motor la ejecuta
    al open de t+1. single_vote ("ema", "roc" o "bb") usa un solo voto en vez
    de la regla 2 de 3 (solo para el analisis de robustez).

    Recibe las velas de 5 min (Open, High, Low, Close) y los parametros θ.
    Regresa un DataFrame con el mismo indice que df y las columnas ema_fast,
    ema_slow, roc, percent_b, adx, atr, vote_ema, vote_roc, vote_bb, state
    y signal.
    """
    features = strategy_features_4h(df, params, single_vote)

    # 3. Pasar a 5 min sin look-ahead; antes de la primera vela cerrada no hay voto.
    aligned = align_to_base(features, df.index, TIMEFRAME)
    for col in [*SINGLE_VOTES.values(), "state"]:
        aligned[col] = aligned[col].fillna(0).astype(int)

    # 4. Señal de entrada: solo en la barra de 5 min donde cambia el estado.
    aligned["signal"] = entry_signal(aligned["state"])
    return aligned


def indicator_correlation(df: pd.DataFrame, params: dict = THETA0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlacion entre los tres votos y entre los indicadores, en velas de 4h.

    La distancia entre EMAs se normaliza: (EMA_rapida - EMA_lenta) / EMA_lenta,
    para que no dependa del nivel de precio. Se ignoran las velas de calentamiento.
    Regresa (correlacion de votos, correlacion de indicadores).
    """
    features = strategy_features_4h(df, params).dropna()
    votes = features[list(SINGLE_VOTES.values())]
    indicators = pd.DataFrame({
        "ema_gap": (features["ema_fast"] - features["ema_slow"]) / features["ema_slow"],
        "roc": features["roc"],
        "percent_b": features["percent_b"],
        "adx": features["adx"],
    })
    return votes.corr(), indicators.corr()
