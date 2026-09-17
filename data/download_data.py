"""Descarga precios diarios de NVDA y los guarda como CSV local.

Los notebooks NO deben llamar a yfinance directamente: este script es el
unico punto de descarga y deja el CSV listo en este mismo directorio (data/).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent

TICKER = "NVDA"
START = "2024-01-01"
END = "2026-09-15"
INTERVAL = "1d"


def download_daily_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga barras diarias OHLCV para un ticker en un rango de fechas.

    Parametros
    ----------
    ticker : str
        Simbolo a descargar (ej. "NVDA").
    start : str
        Fecha inicial en formato "YYYY-MM-DD".
    end : str
        Fecha final en formato "YYYY-MM-DD".

    Regresa
    -------
    pd.DataFrame
        OHLCV con indice de fecha, una fila por dia de trading.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"yfinance no regreso datos para {ticker}")

    # yf.download puede regresar columnas MultiIndex (Precio, Ticker) cuando
    # se pasa un solo simbolo dentro de una lista; aqui aplanamos a columnas simples.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def main() -> None:
    df = download_daily_bars(TICKER, START, END)
    out_path = DATA_DIR / f"{TICKER}_daily.csv"
    df.to_csv(out_path)
    print(f"{TICKER}: {len(df)} barras guardadas en {out_path}")


if __name__ == "__main__":
    main()
