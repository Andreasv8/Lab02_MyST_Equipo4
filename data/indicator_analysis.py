"""Calcula indicadores tecnicos sobre NVDA_daily.csv y analiza su correlacion.

Todos los indicadores se calculan manualmente con pandas/numpy (sin la
libreria ta). Genera un heatmap de correlacion (indicator_corr.png) e
imprime la matriz numerica en consola.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
INPUT_CSV = DATA_DIR / "NVDA_daily.csv"
OUTPUT_PNG = DATA_DIR / "indicator_corr.png"


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


def stochastic_percent_k(high: pd.Series, low: pd.Series, close: pd.Series,
                          window: int = 14, smooth: int = 3) -> pd.Series:
    """%K estocastico, suavizado con una media movil simple.

    %K crudo = (Close - minimo_N) / (maximo_N - minimo_N) * 100.
    """
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    raw_k = (close - lowest_low) / (highest_high - lowest_low) * 100
    return raw_k.rolling(window=smooth).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR (Average True Range) con suavizado de Wilder."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


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


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Genera y guarda un heatmap de una matriz de correlacion."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlacion entre indicadores tecnicos -- NVDA")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(INPUT_CSV, index_col="Date", parse_dates=True)

    indicators = build_indicators(df)
    corr = indicators.corr()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(corr.round(3))

    plot_correlation_heatmap(corr, OUTPUT_PNG)
    print(f"\nHeatmap guardado en {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
