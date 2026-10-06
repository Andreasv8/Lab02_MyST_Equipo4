"""Indicadores tecnicos de lab_02, calculados a mano con pandas/numpy.

Todas las funciones son causales: el valor en la barra t solo usa datos
hasta t (ver tests/test_truncation.py).
"""

import numpy as np
import pandas as pd


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
