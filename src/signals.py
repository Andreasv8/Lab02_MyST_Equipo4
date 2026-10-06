"""Indicadores tecnicos, votos y señal de la estrategia de lab_02 (docs/SPEC.md).

Los indicadores se calculan a mano con pandas/numpy.

Todas las funciones son causales: el valor en la barra t solo usa datos
hasta t (ver tests/test_truncation.py).
"""

import numpy as np
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


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) con suavizado de Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Linea MACD: EMA rapida menos EMA lenta."""
    return ema(close, fast) - ema(close, slow)

def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Histograma del MACD: linea MACD menos su linea de señal (EMA de la linea MACD).

    > 0 = el impulso alcista se acelera; < 0 = se acelera el bajista.
    NaN hasta tener `slow` barras, para no votar con medias sin calentar.
    """
    line = macd_line(close, fast, slow)
    hist = line - ema(line, signal)
    return hist.where(close.rolling(window=slow).count() >= slow)

def stochastic_percent_k(high: pd.Series, low: pd.Series, close: pd.Series,
                          window: int = 14, smooth: int = 3) -> pd.Series:
    """%K estocastico, suavizado con una media movil simple.

    %K crudo = (Close - minimo_N) / (maximo_N - minimo_N) * 100.
    """
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    raw_k = (close - lowest_low) / (highest_high - lowest_low) * 100
    return raw_k.rolling(window=smooth).mean()


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


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV (On-Balance Volume): volumen acumulado con signo segun la direccion del precio."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Precio tipico: (High + Low + Close) / 3."""
    return (high + low + close) / 3


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """CCI (Commodity Channel Index).

    CCI = (precio_tipico - SMA(precio_tipico)) / (0.015 * desviacion_media_absoluta).
    """
    tp = typical_price(high, low, close)
    sma_tp = sma(tp, window)
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """MFI (Money Flow Index): RSI ponderado por volumen.

    Usa precio_tipico * Volume como money flow; separa flujo positivo y
    negativo segun la direccion del precio tipico y combina igual que el RSI.
    """
    tp = typical_price(high, low, close)
    raw_money_flow = tp * volume

    tp_direction = tp.diff()
    positive_flow = raw_money_flow.where(tp_direction > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_direction < 0, 0.0)

    positive_sum = positive_flow.rolling(window=window).sum()
    negative_sum = negative_flow.rolling(window=window).sum()

    money_ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + money_ratio))


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """ROC (Rate of Change): variacion porcentual respecto a N periodos atras."""
    return ((close - close.shift(window)) / close.shift(window)) * 100


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R: distancia del cierre al maximo de N barras, en [-100, 0].

    %R = (maximo_N - Close) / (maximo_N - minimo_N) * -100.
    0 = cierra en el maximo del rango, -100 = cierra en el minimo.
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return (highest_high - close) / (highest_high - lowest_low) * -100


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

def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                       window: int = 20) -> pd.Series:
    """CMF (Chaikin Money Flow): presion compradora/vendedora ponderada por volumen, en [-1, 1].

    Multiplicador = ((Close - Low) - (High - Close)) / (High - Low): +1 si cierra
    en el maximo de la barra, -1 si cierra en el minimo.
    CMF = suma_N(multiplicador * Volume) / suma_N(Volume).
    """
    multiplier = ((close - low) - (high - close)) / (high - low)
    multiplier = multiplier.fillna(0.0)  # barra con High == Low: sin presion
    money_flow_volume = multiplier * volume
    return money_flow_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()


def donchian_position(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Posicion del cierre dentro del canal de Donchian de N barras, en [0, 1].

    (Close - minimo_N) / (maximo_N - minimo_N). 1 = cierra en maximos de N
    barras (breakout alcista), 0 = cierra en minimos.
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return (close - lowest_low) / (highest_high - lowest_low)


def obv_change_ratio(close: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    """Cambio del OBV en N barras, normalizado por el volumen de esas N barras, en [-1, 1].

    (OBV_t - OBV_{t-N}) / suma_N(Volume). Es la pendiente del OBV (estacionaria),
    no su nivel: +1 si todo el volumen fue en dias de subida, -1 si fue en dias de bajada.
    """
    obv_series = obv(close, volume)
    return obv_series.diff(window) / volume.rolling(window=window).sum()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volumen relativo: Volume / SMA_N(Volume). >1 = volumen por encima de lo normal."""
    return volume / sma(volume, window)


def bollinger_bandwidth(close: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
    """Ancho de las bandas de Bollinger relativo a la media: (superior - inferior) / media.

    Mide volatilidad en terminos relativos al precio (estacionario).
    """
    mid = sma(close, window)
    std = close.rolling(window=window).std()
    return (2 * num_std * std) / mid


def price_distance(close: pd.Series, moving_average: pd.Series) -> pd.Series:
    """Distancia relativa del precio a una media movil: Close / MA - 1."""
    return close / moving_average - 1


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores del lab sobre un DataFrame OHLCV.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".

    Regresa
    -------
    pd.DataFrame
        Una columna por indicador, sin NaNs (dropeados al final).
    """
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    indicators = pd.DataFrame({
        "SMA_20": sma(close, 20),
        "SMA_50": sma(close, 50),
        "EMA_20": ema(close, 20),
        "BB_percent_b_20_2": bollinger_percent_b(close, 20, 2),
        "RSI_14": rsi(close, 14),
        "MACD_line_12_26_9": macd_line(close, 12, 26),
        "Stoch_percent_K_14_3": stochastic_percent_k(high, low, close, 14, 3),
        "ATR_14": atr(high, low, close, 14),
        "OBV": obv(close, volume),
        "CCI_20": cci(high, low, close, 20),
        "MFI_14": mfi(high, low, close, volume, 14),
        "ROC_10": roc(close, 10),
    })

    return indicators.dropna()


def build_stationary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Candidatos para la seleccion de features, todos en forma estacionaria.

    Los indicadores de nivel de precio se transforman para que la correlacion
    no salga alta solo porque todos siguen al precio: SMA/EMA como distancia
    Close/MA - 1, MACD como fraccion del precio y OBV como cambio normalizado.
    El ATR queda fuera: solo se usa para SL/TP y sizing.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".

    Regresa
    -------
    pd.DataFrame
        Una columna por candidato (17), sin NaNs (dropeados al final).
    """
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    indicators = pd.DataFrame({
        # Tendencia / overlays
        "SMA20_dist": price_distance(close, sma(close, 20)),
        "SMA50_dist": price_distance(close, sma(close, 50)),
        "EMA20_dist": price_distance(close, ema(close, 20)),
        "MACD_pct": macd_line(close, 12, 26) / close,
        "ADX_14": adx(high, low, close, 14),
        "Donchian_pos_20": donchian_position(high, low, close, 20),
        # Momentum / osciladores
        "RSI_14": rsi(close, 14),
        "Stoch_K_14_3": stochastic_percent_k(high, low, close, 14, 3),
        "Williams_R_14": williams_r(high, low, close, 14),
        "CCI_20": cci(high, low, close, 20),
        "ROC_10": roc(close, 10),
        "BB_percent_b_20_2": bollinger_percent_b(close, 20, 2),
        # Volumen
        "MFI_14": mfi(high, low, close, volume, 14),
        "CMF_20": chaikin_money_flow(high, low, close, volume, 20),
        "OBV_change_10": obv_change_ratio(close, volume, 10),
        "Volume_ratio_20": volume_ratio(volume, 20),
        # Volatilidad
        "BB_bandwidth_20_2": bollinger_bandwidth(close, 20, 2),
    })

    return indicators.dropna()


# Umbral de Wilder para considerar que el mercado esta en tendencia.
ADX_THRESHOLD = 25
# Lab 02, seccion 3.1: al menos 2 de los 3 indicadores deben coincidir en direccion.
MIN_VOTES = 2
VOTE_COLUMNS = ["vote_roc", "vote_macd", "vote_adx"]


def indicator_votes(roc_values: pd.Series, macd_hist: pd.Series, plus_di: pd.Series,
                    minus_di: pd.Series, adx_values: pd.Series,
                    adx_threshold: float = ADX_THRESHOLD) -> pd.DataFrame:
    """Voto de cada indicador en cada barra: +1 (largo), -1 (corto) o 0 (no vota).

    vote_roc  = signo(ROC)                          momento
    vote_macd = signo(MACD - señal)                 tendencia
    vote_adx  = signo(+DI - -DI) si ADX > umbral    tendencia; 0 si ADX <= umbral

    Los NaN del warm-up y los valores exactamente 0 no votan.
    """
    def _sign(values: pd.Series) -> pd.Series:
        return (values > 0).astype(int) - (values < 0).astype(int)

    return pd.DataFrame({
        "vote_roc": _sign(roc_values),
        "vote_macd": _sign(macd_hist),
        "vote_adx": _sign(plus_di - minus_di).where(adx_values > adx_threshold, 0),
    }, index=roc_values.index).astype(int)


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


def compute_features(df: pd.DataFrame, roc_window: int = 10, macd_fast: int = 12,
                     macd_slow: int = 26, macd_signal: int = 9, adx_window: int = 14,
                     adx_threshold: float = ADX_THRESHOLD) -> pd.DataFrame:
    """Calcula los tres indicadores, sus votos y la señal de confirmacion 2 de 3.

    Indicadores: ROC (momento), MACD (tendencia) y ADX con +DI/-DI (tendencia).
    Ninguno usa volumen: en los datos de BTC falta en ~46% de las barras.
    """
    close, high, low = df["Close"], df["High"], df["Low"]

    roc_values = roc(close, roc_window)
    macd_hist = macd_histogram(close, macd_fast, macd_slow, macd_signal)
    plus_di, minus_di = directional_indicators(high, low, close, adx_window)
    adx_values = adx(high, low, close, adx_window)
    votes = indicator_votes(roc_values, macd_hist, plus_di, minus_di, adx_values, adx_threshold)

    return pd.DataFrame({
        "roc": roc_values,
        "macd_hist": macd_hist,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx": adx_values,
        "atr_14": atr(high, low, close, 14),
        **{col: votes[col] for col in VOTE_COLUMNS},
        "n_long": (votes == 1).sum(axis=1),
        "n_short": (votes == -1).sum(axis=1),
        "signal": confirmation_signal(votes),
    })


def ema_adx_signal(ema_fast: pd.Series, ema_slow: pd.Series, adx_values: pd.Series,
                   adx_threshold: float = ADX_THRESHOLD) -> pd.Series:
    """Señal de la estrategia EMA + ADX (estrategia 2).

    señal_t = +1 si EMA_rapida > EMA_lenta y ADX > umbral   (tendencia alcista)
    señal_t = -1 si EMA_rapida < EMA_lenta y ADX > umbral   (tendencia bajista)
    señal_t =  0 en otro caso (ADX debil, EMAs iguales o NaN de warm-up)
    """
    trend = (ema_fast > ema_slow).astype(int) - (ema_fast < ema_slow).astype(int)
    return trend.where(adx_values > adx_threshold, 0).astype(int)


def compute_ema_adx_features(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50,
                             adx_window: int = 14, adx_threshold: float = ADX_THRESHOLD,
                             atr_window: int = 14, timeframe: str = "1h",
                             entry_on_change: bool = True) -> pd.DataFrame:
    """Indicadores y señal EMA + ADX calculados en un timeframe mayor y llevados a 5 min.

    Las barras de 5 minutos son muy ruidosas (efficiency ratio ~ random walk),
    asi que EMAs, ADX y ATR se calculan sobre barras agregadas de `timeframe`
    y se alinean sin look-ahead al indice de 5 minutos (ver align_to_base).
    La ejecucion (entrada, SL/TP, holding) sigue ocurriendo en 5 minutos.

    Parametros
    ----------
    df : pd.DataFrame
        OHLC de 5 minutos con indice de fechas.
    ema_fast, ema_slow : int
        Spans de las EMAs, en barras de `timeframe`.
    adx_window, atr_window : int
        Ventanas de ADX y ATR, en barras de `timeframe`.
    adx_threshold : float
        Umbral de fuerza de tendencia.
    timeframe : str
        Regla de pandas para agregar ("1h", "4h", ...).
    entry_on_change : bool
        True: signal solo vale +-1 en la barra de 5 minutos donde la
        tendencia cambia (evento), y 0 mientras se mantiene. Evita reentrar
        en la barra siguiente a cada SL/TP dentro de la misma tendencia,
        que en train era el 85% de los trades y pagaba 0.35% cada vez.
        False: signal = tendencia (estado) en cada barra.

    Regresa
    -------
    pd.DataFrame
        ema_fast, ema_slow, adx, atr, trend (estado) y signal, con el mismo
        indice que df.
    """
    htf = resample_ohlc(df, timeframe)
    high, low, close = htf["High"], htf["Low"], htf["Close"]
    features = pd.DataFrame({
        "ema_fast": ema(close, ema_fast),
        "ema_slow": ema(close, ema_slow),
        "adx": adx(high, low, close, adx_window),
        "atr": atr(high, low, close, atr_window),
    })
    # Sin señal hasta que la EMA lenta tenga historia suficiente (warm-up).
    features.loc[features.index[:ema_slow - 1], ["ema_fast", "ema_slow"]] = float("nan")
    features["signal"] = ema_adx_signal(features["ema_fast"], features["ema_slow"],
                                        features["adx"], adx_threshold)
    aligned = align_to_base(features, df.index, timeframe)
    aligned["trend"] = aligned.pop("signal").fillna(0).astype(int)
    trend = aligned["trend"]
    aligned["signal"] = trend.where(trend != trend.shift(1), 0) if entry_on_change else trend
    return aligned


def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
    """Señal de entrada por barra con la regla de confluencia (docs/SPEC.md, seccion 3).

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".

    Regresa
    -------
    pd.Series
        Señal por barra: 1 (long), -1 (short) o 0 (flat).
    """
    return compute_features(df)["signal"]



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


def compute_strategy(df: pd.DataFrame, params: dict = THETA0) -> pd.DataFrame:
    """Calcula indicadores, votos, estado y señal de la estrategia final.

    Los indicadores se calculan en velas de 4h y se pasan a 5 min con
    align_to_base: una vela de 4h solo se usa cuando ya cerro (sin look-ahead).
    La señal se toma al cierre de la barra t de 5 min; el motor la ejecuta
    al open de t+1.

    Recibe las velas de 5 min (Open, High, Low, Close) y los parametros θ.
    Regresa un DataFrame con el mismo indice que df y las columnas ema_fast,
    ema_slow, roc, percent_b, adx, atr, vote_ema, vote_roc, vote_bb, state
    y signal.
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

    # 2. Votos y estado, tambien en 4h.
    votes = strategy_votes(features["ema_fast"], features["ema_slow"], features["roc"],
                           features["percent_b"], params["bb_threshold"])
    features = features.join(votes)
    features["state"] = confirmed_state(votes, features["adx"], params["adx_threshold"])

    # 3. Pasar a 5 min sin look-ahead; antes de la primera vela cerrada no hay voto.
    aligned = align_to_base(features, df.index, TIMEFRAME)
    for col in [*votes.columns, "state"]:
        aligned[col] = aligned[col].fillna(0).astype(int)

    # 4. Señal de entrada: solo en la barra de 5 min donde cambia el estado.
    aligned["signal"] = entry_signal(aligned["state"])
    return aligned
