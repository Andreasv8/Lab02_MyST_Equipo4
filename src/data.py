"""Carga, validacion y auditoria de los datos de BTCUSDT de 5 minutos (Lab 02)."""

import pandas as pd

BAR_MINUTES = 5
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


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