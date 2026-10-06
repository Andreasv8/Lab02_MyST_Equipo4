"""Carga, validacion y auditoria de los datos de BTCUSDT de 5 minutos (Lab 02)."""

from pathlib import Path

import pandas as pd

BAR_MINUTES = 5
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

# Archivos de datos, resueltos desde la raiz del repo.
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = str(ROOT / "data" / "btc_project_train.csv")
TEST_PATH = str(ROOT / "data" / "btc_project_test.csv")

# El archivo de test trae un dia suelto (2023-12-31) y despues un hueco de
# 122 dias. Solo se evalua mayo-junio 2024.
TEST_START = "2024-05-02"
TEST_END = "2024-06-03"


def load_btc(path: str) -> pd.DataFrame:
    """Lee un CSV de BTCUSDT de 5 minutos y lo deja listo para el motor.

    Limpieza (declarada en el reporte):
    - Se eliminan las barras sin precio (huecos del proveedor).
    - Se eliminan las barras fuera de la rejilla de 5 minutos.
    - El volumen faltante se pone en 0; no se inventa volumen.

    Regresa un DataFrame con Open, High, Low, Close, Volume e indice Datetime.
    """
    raw = pd.read_csv(path, usecols=["Datetime", *PRICE_COLUMNS, "Volume"], parse_dates=["Datetime"])
    df = raw.dropna(subset=PRICE_COLUMNS).set_index("Datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    on_grid = (df.index.minute % BAR_MINUTES == 0) & (df.index.second == 0)
    df = df[on_grid].copy()
    df["Volume"] = df["Volume"].fillna(0.0)
    return df


def load_train() -> pd.DataFrame:
    """Carga el archivo de train completo (2022-06-01 a 2023-12-31), ya limpio."""
    return load_btc(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    """Carga el archivo de test completo, ya limpio.

    Incluye el dia suelto 2023-12-31; para evaluar se recorta con
    TEST_START y TEST_END. Se usa una sola vez, al final.
    """
    return load_btc(TEST_PATH)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agrega barras de 5 minutos a un timeframe mayor (p. ej. "1h", "4h").

    Cada barra agregada cubre [T, T + rule) y queda etiquetada con su inicio T.
    Las barras agregadas sin datos (huecos) se eliminan.

    Regresa un DataFrame con Open, High, Low, Close, Volume.
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df[list(agg)].resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=PRICE_COLUMNS)


def align_to_base(values: pd.DataFrame | pd.Series, base_index: pd.DatetimeIndex,
                  rule: str) -> pd.DataFrame | pd.Series:
    """Lleva valores de un timeframe mayor al indice de 5 minutos SIN look-ahead.

    La barra agregada que empieza en T se conoce al cierre de la barra de
    5 minutos que empieza en T + rule - 5 min. Se mueve su etiqueta a ese
    instante y cada barra base toma el ultimo valor ya disponible (ffill).
    Si faltan barras al final del bloque, el valor llega en la siguiente
    barra base existente, nunca antes.

    Parametros
    ----------
    values : pd.DataFrame o pd.Series
        Valores calculados sobre resample_ohlc(df, rule).
    base_index : pd.DatetimeIndex
        Indice de las barras de 5 minutos.
    rule : str
        El mismo timeframe usado en resample_ohlc.

    Regresa
    -------
    Valores reindexados a base_index (NaN antes de la primera barra completa).
    """
    available_at = values.index + pd.Timedelta(rule) - pd.Timedelta(minutes=BAR_MINUTES)
    shifted = values.copy()
    shifted.index = available_at
    return shifted.reindex(base_index, method="ffill")
