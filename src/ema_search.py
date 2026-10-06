"""Busqueda de hiperparametros de la estrategia 2: cruce de EMAs con filtro ADX.

θ, PARAM_SPACE, el backtest de θ y Optuna viven en src/optimization.py; aqui
queda la etapa 1 y las tablas de la busqueda. Todo SOLO con train:
1. screen_classic: los pares de EMAs mas usados en la literatura, sin optimizar.
2. optimize_theta (src/optimization.py): random search + TPE sobre PARAM_SPACE.
   De ahi sale THETA0, el centro de la meseta del top 10% (ver su comentario).
Test solo se evalua al final; validation no se carga.
"""

import itertools
import math
from typing import Optional

import optuna
import pandas as pd

from src.metrics import summarize
from src.optimization import BARS_PER_HOUR, params_to_theta, run_theta, theta_config

# Pares (rapida, lenta) clasicos: 9/21, 12/26 (MACD), 20/50 y 50/200 (golden cross).
CLASSIC_PAIRS = [(9, 21), (12, 26), (20, 50), (50, 200)]
CLASSIC_TIMEFRAMES = ["1h", "4h"]
CLASSIC_ADX = [20.0, 25.0]

# Version de libro: entrada por estado (cada barra en tendencia), SL 2·ATR,
# TP 3·ATR (rr 1.5) y holding maximo de 7 dias.
THETA_CLASSIC = {
    "timeframe": "1h",
    "ema_fast": 20,
    "ema_slow": 50,
    "adx_threshold": 25.0,
    "entry_on_change": False,
    "sl_mult": 2.0,
    "rr": 1.5,
    "max_holding": 7 * 24 * BARS_PER_HOUR,
}


def evaluate(df: pd.DataFrame, theta: dict) -> pd.Series:
    """Metricas (summarize) del backtest de θ sobre df, con θ al frente."""
    result = run_theta(df, theta)
    return pd.concat([pd.Series(theta), summarize(result.equity, result.trades, theta_config(theta))])


def screen_classic(df_train: pd.DataFrame, entry_on_change: bool = False,
                   rr: float = THETA_CLASSIC["rr"]) -> pd.DataFrame:
    """Etapa 1: evalua en train los pares de EMAs clasicos sin optimizar nada.

    Combina CLASSIC_PAIRS x CLASSIC_TIMEFRAMES x CLASSIC_ADX con la salida
    de THETA_CLASSIC (salvo entry_on_change y rr). Regresa una fila por
    combinacion, ordenada por Calmar.
    """
    rows = []
    for (fast, slow), timeframe, threshold in itertools.product(CLASSIC_PAIRS, CLASSIC_TIMEFRAMES,
                                                                 CLASSIC_ADX):
        theta = {**THETA_CLASSIC, "timeframe": timeframe, "ema_fast": fast, "ema_slow": slow,
                 "adx_threshold": threshold, "entry_on_change": entry_on_change, "rr": rr}
        rows.append(evaluate(df_train, theta))
    return pd.DataFrame(rows).sort_values("calmar", ascending=False).reset_index(drop=True)


def trials_table(study: optuna.Study, top: Optional[int] = None) -> pd.DataFrame:
    """Trials validos (J finito) como θ + J, ordenados de mejor a peor."""
    rows = [{**params_to_theta(t.params), "J": t.value} for t in study.trials
            if t.value is not None and math.isfinite(t.value)]
    table = pd.DataFrame(rows).sort_values("J", ascending=False).reset_index(drop=True)
    return table.head(top) if top else table
