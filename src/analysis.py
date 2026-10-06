"""Analisis de lab_02 sobre el motor de backtest: sensibilidad a costos y desglose de trades.

Funciones puras: reciben datos y configuracion, corren backtest() o
resumen sus salidas, y no leen archivos.
"""

import math
from dataclasses import replace
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, backtest, compute_win_rate
from src.metrics import summarize
from src.signals import compute_features
from src.splits import SPLITS

DEFAULT_PERIODS = {"train": SPLITS["train"], "test": SPLITS["test"]}


def cost_sensitivity(df: pd.DataFrame, config: BacktestConfig,
                     round_trip_bps: Iterable[float] = range(0, 55, 5),
                     periods: Optional[dict] = None) -> pd.DataFrame:
    """Sharpe y equity final por costo de ida y vuelta, por periodo.

    Para cada costo de ida y vuelta (en bps) corre backtest() con
    commission_rate = bps / 2 / 10_000 (costo por lado) y slippage 0 sobre toda la serie de df,
    y mide cada periodo por separado con summarize() (equity rebasada al
    inicio del periodo). El borrow fee no cambia entre escenarios.

    Parametros
    ----------
    df : pd.DataFrame
        OHLCV con indice de fechas (ya recortado a los periodos permitidos).
    config : BacktestConfig
        Configuracion base; solo se sustituye el costo por lado.
    round_trip_bps : Iterable[float]
        Costos de ida y vuelta en puntos base (default 0, 5, ..., 50).
    periods : dict, opcional
        {nombre: (inicio, fin)}. Default: train y test de src/splits.py.

    Regresa
    -------
    pd.DataFrame
        Indice bps; columnas sharpe_<periodo> y equity_final_<periodo>.
    """
    periods = DEFAULT_PERIODS if periods is None else periods
    features = compute_features(df)

    rows = []
    for bps in round_trip_bps:
        run_config = replace(config, commission_rate=bps / 2 / 10_000, slippage_rate=0.0)
        result = backtest(df, features["signal"], features["atr_14"], run_config)
        row = {"bps": bps}
        for name, (start, end) in periods.items():
            summary = summarize(result.equity, result.trades, run_config, start, end)
            row[f"sharpe_{name}"] = summary["sharpe"]
            row[f"equity_final_{name}"] = summary["equity_final"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("bps")


def break_even_cost(bps: pd.Series, sharpe: pd.Series) -> float:
    """Costo de ida y vuelta en el que el Sharpe cruza 0 por primera vez.

    Interpola linealmente entre los dos puntos de la malla donde el Sharpe
    pasa de > 0 a <= 0:  bps* = b_0 + S_0 · (b_1 - b_0) / (S_0 - S_1).

    Parametros
    ----------
    bps : pd.Series
        Costos en bps, en orden creciente.
    sharpe : pd.Series
        Sharpe en cada costo (mismo orden).

    Regresa
    -------
    float
        Costo de break-even en bps; NaN si el Sharpe no cruza 0 en la malla
        (siempre > 0 o ya <= 0 desde el primer punto).
    """
    b = np.asarray(bps, dtype=float)
    s = np.asarray(sharpe, dtype=float)
    if len(s) == 0 or not s[0] > 0:
        return np.nan
    for i in range(1, len(s)):
        if s[i] <= 0:
            return b[i - 1] + s[i - 1] * (b[i] - b[i - 1]) / (s[i - 1] - s[i])
    return np.nan


def sharpe_band(sharpe: float) -> str:
    """Interpretacion del Sharpe anualizado con los rangos de clase.

    < 1 malo; 1-2 bueno; 2-3 excelente; > 3 probable error (revisar sesgos).

    Parametros
    ----------
    sharpe : float
        Sharpe anualizado.

    Regresa
    -------
    str
        Etiqueta del rango ("sin dato" si es NaN).
    """
    if not np.isfinite(sharpe):
        return "sin dato"
    if sharpe > 3:
        return "probable error"
    if sharpe > 2:
        return "excelente"
    if sharpe >= 1:
        return "bueno"
    return "malo"


def position_size(trades: pd.DataFrame, equity_df: pd.DataFrame, start: Optional[str] = None,
                  end: Optional[str] = None) -> float:
    """Nocional promedio por trade como fraccion del capital al entrar.

    tamaño_i = shares_i · raw_entry_price_i / E_{entry_bar_i}
    tamaño = media(tamaño_i)

    E_{entry_bar} es la equity al cierre de la barra de entrada. Los trades se
    asignan al periodo por exit_date, igual que en summarize().

    Parametros
    ----------
    trades : pd.DataFrame
        Trades de backtest() (usa shares, raw_entry_price, entry_bar, exit_date).
    equity_df : pd.DataFrame
        Salida equity de backtest() (indice posicional = entry_bar).
    start, end : str, opcional
        Limites del periodo (inclusive).

    Regresa
    -------
    float
        Tamaño promedio en [0, 1] (NaN si no hay trades en el periodo).
    """
    exit_date = trades["exit_date"]
    mask = pd.Series(True, index=trades.index)
    if start is not None:
        mask &= exit_date >= pd.Timestamp(start)
    if end is not None:
        mask &= exit_date <= pd.Timestamp(end)
    period = trades[mask]
    if period.empty:
        return np.nan

    equity_at_entry = equity_df["equity"].to_numpy()[period["entry_bar"].to_numpy()]
    return float((period["shares"] * period["raw_entry_price"] / equity_at_entry).mean())


def win_rate_standard_error(p: float, n: int) -> float:
    """Error estandar de un win rate estimado con n trades: sqrt(p·(1 - p) / n)."""
    return math.sqrt(p * (1 - p) / n)


def trade_breakdown(trades: pd.DataFrame, start: Optional[str] = None,
                    end: Optional[str] = None) -> pd.DataFrame:
    """Desglose de trades por lado (long / short) para un periodo.

    Los trades se asignan al periodo por exit_date (cuando se realiza el P&L),
    igual que en summarize().

    Parametros
    ----------
    trades : pd.DataFrame
        Trades de backtest().
    start, end : str, opcional
        Limites del periodo (inclusive).

    Regresa
    -------
    pd.DataFrame
        Indice side; columnas n_trades, total_pnl, avg_pnl, win_rate (fraccion).
    """
    exit_date = trades["exit_date"]
    mask = pd.Series(True, index=trades.index)
    if start is not None:
        mask &= exit_date >= pd.Timestamp(start)
    if end is not None:
        mask &= exit_date <= pd.Timestamp(end)
    period = trades[mask]

    rows = {}
    for side in ["long", "short"]:
        side_trades = period[period["side"] == side]
        rows[side] = {
            "n_trades": len(side_trades),
            "total_pnl": side_trades["pnl"].sum(),
            "avg_pnl": side_trades["pnl"].mean() if len(side_trades) else np.nan,
            "win_rate": compute_win_rate(side_trades.to_dict("records")) / 100 if len(side_trades) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("side")
