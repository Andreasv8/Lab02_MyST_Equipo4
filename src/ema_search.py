"""Busqueda de hiperparametros de la estrategia 2: cruce de EMAs con filtro ADX.

θ = (timeframe, ema_fast, ema_slow, adx_threshold, entry_on_change, sl_mult, rr,
max_holding).
EMAs, ADX(14) y ATR(14) se calculan en barras de `timeframe` (1h o 4h) y la
ejecucion ocurre en barras de 5 minutos (ver compute_ema_adx_features).
max_holding se expresa en barras de 5 minutos, como lo usa el motor.

Dos etapas, SOLO con train:
1. screen_classic: los pares de EMAs mas usados en la literatura, sin optimizar.
2. optimize_ema: Optuna (random search + TPE) sobre PARAM_SPACE.
Test solo se evalua al final; validation no se carga.
"""

import itertools
import math
from typing import Optional

import optuna
import pandas as pd

from src.backtest import BacktestConfig, BacktestResult, backtest
from src.metrics import summarize
from src.optimization import objective
from src.strategy import compute_ema_adx_features

SEED = 42
N_TRIALS = 300
N_STARTUP_TRIALS = 100
MIN_TRADES = 30
BARS_PER_HOUR = 12

# Pares (rapida, lenta) clasicos: 9/21, 12/26 (MACD), 20/50 y 50/200 (golden cross).
CLASSIC_PAIRS = [(9, 21), (12, 26), (20, 50), (50, 200)]
CLASSIC_TIMEFRAMES = ["1h", "4h"]
CLASSIC_ADX = [20.0, 25.0]

# Salida clasica: SL 2·ATR, TP 3·ATR (rr 1.5), holding maximo de 7 dias.
# Entrada por estado (cada barra en tendencia), como en la version de libro.
THETA0 = {
    "timeframe": "1h",
    "ema_fast": 20,
    "ema_slow": 50,
    "adx_threshold": 25.0,
    "entry_on_change": False,
    "sl_mult": 2.0,
    "rr": 1.5,
    "max_holding": 7 * 24 * BARS_PER_HOUR,
}

# θ elegida en train (no el mejor trial): centro de la meseta del top 10% de
# Optuna, con ADX = 25 (Wilder) porque ADX ~27 era un pico aislado, y sin
# take-profit (rr alto) porque Calmar crece monotono con rr. En train:
# +28.6%, Sharpe 2.0, max DD -11.9%, Calmar 2.6, 46 trades; 96% de las 108 θ
# vecinas tienen retorno > 0.
THETA_ROBUST = {
    "timeframe": "4h",
    "ema_fast": 12,
    "ema_slow": 18,
    "adx_threshold": 25.0,
    "entry_on_change": True,
    "sl_mult": 3.0,
    "rr": 20.0,
    "max_holding": 5 * 24 * BARS_PER_HOUR,
}

# Espacio de busqueda: nombre -> (tipo, opciones o (minimo, maximo)).
# ema_slow se busca como multiplo de ema_fast para garantizar rapida < lenta.
PARAM_SPACE = {
    "timeframe": ("cat", ["1h", "4h"]),
    "ema_fast": ("int", 5, 60),
    "slow_ratio": ("float", 1.5, 5.0),
    "adx_threshold": ("float", 15.0, 35.0),
    "entry_on_change": ("cat", [True, False]),
    "sl_mult": ("float", 1.0, 4.0),
    # rr alto ~ sin take-profit: la salida queda en SL, cruce opuesto u holding.
    "rr": ("float", 1.0, 10.0),
    "max_holding": ("int", 4 * BARS_PER_HOUR, 14 * 24 * BARS_PER_HOUR),
}


def theta_config(theta: dict) -> BacktestConfig:
    """BacktestConfig con la salida de θ (tp = sl · rr); sizing y costos por defecto."""
    return BacktestConfig(sl_mult=theta["sl_mult"], tp_mult=theta["sl_mult"] * theta["rr"],
                          max_holding=int(theta["max_holding"]))


def run_ema(df: pd.DataFrame, theta: dict) -> BacktestResult:
    """Backtest de la estrategia EMA + ADX con parametros θ sobre df (5 minutos)."""
    features = compute_ema_adx_features(df, ema_fast=int(theta["ema_fast"]),
                                        ema_slow=int(theta["ema_slow"]),
                                        adx_threshold=theta["adx_threshold"],
                                        timeframe=theta["timeframe"],
                                        entry_on_change=bool(theta["entry_on_change"]))
    return backtest(df, features["signal"], features["atr"], theta_config(theta))


def evaluate(df: pd.DataFrame, theta: dict) -> pd.Series:
    """Metricas (summarize) del backtest de θ sobre df, con θ al frente."""
    result = run_ema(df, theta)
    return pd.concat([pd.Series(theta), summarize(result.equity, result.trades, theta_config(theta))])


def screen_classic(df_train: pd.DataFrame, entry_on_change: bool = False,
                   rr: float = THETA0["rr"]) -> pd.DataFrame:
    """Etapa 1: evalua en train los pares de EMAs clasicos sin optimizar nada.

    Combina CLASSIC_PAIRS x CLASSIC_TIMEFRAMES x CLASSIC_ADX con la salida
    de THETA0 (salvo entry_on_change y rr). Regresa una fila por
    combinacion, ordenada por Calmar.
    """
    rows = []
    for (fast, slow), timeframe, threshold in itertools.product(CLASSIC_PAIRS, CLASSIC_TIMEFRAMES,
                                                                 CLASSIC_ADX):
        theta = {**THETA0, "timeframe": timeframe, "ema_fast": fast, "ema_slow": slow,
                 "adx_threshold": threshold, "entry_on_change": entry_on_change, "rr": rr}
        rows.append(evaluate(df_train, theta))
    return pd.DataFrame(rows).sort_values("calmar", ascending=False).reset_index(drop=True)


def suggest_theta(trial: optuna.Trial) -> dict:
    """Propone un θ dentro de PARAM_SPACE; ema_slow = round(ema_fast · slow_ratio)."""
    raw = {}
    for name, spec in PARAM_SPACE.items():
        if spec[0] == "cat":
            raw[name] = trial.suggest_categorical(name, spec[1])
        elif spec[0] == "int":
            raw[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            raw[name] = trial.suggest_float(name, spec[1], spec[2])
    return params_to_theta(raw)


def params_to_theta(params: dict) -> dict:
    """Convierte los parametros de un trial (con slow_ratio) en θ (con ema_slow)."""
    theta = {k: v for k, v in params.items() if k != "slow_ratio"}
    theta["ema_slow"] = max(int(params["ema_fast"]) + 1,
                            round(params["ema_fast"] * params["slow_ratio"]))
    return theta


def optimize_ema(df_train: pd.DataFrame, n_trials: int = N_TRIALS,
                 n_startup_trials: int = N_STARTUP_TRIALS, min_trades: int = MIN_TRADES,
                 seed: int = SEED, enqueue: Optional[list] = None) -> optuna.Study:
    """Etapa 2: maximiza J = Calmar en train con Optuna (random search y luego TPE).

    Parametros
    ----------
    df_train : pd.DataFrame
        OHLC de 5 minutos de train unicamente.
    n_trials, n_startup_trials, seed : int
        Presupuesto, trials aleatorios iniciales y semilla.
    min_trades : int
        Minimo de trades cerrados para que J sea valido (si no, J = -inf).
    enqueue : list[dict], opcional
        Parametros (formato de PARAM_SPACE) a evaluar primero, p. ej. los
        mejores clasicos de screen_classic.

    Regresa
    -------
    optuna.Study
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="ema_adx")
    for params in enqueue or []:
        study.enqueue_trial(params)
    study.optimize(lambda trial: objective(run_ema(df_train, suggest_theta(trial)), min_trades),
                   n_trials=n_trials)
    return study


def trials_table(study: optuna.Study, top: Optional[int] = None) -> pd.DataFrame:
    """Trials validos (J finito) como θ + J, ordenados de mejor a peor."""
    rows = [{**params_to_theta(t.params), "J": t.value} for t in study.trials
            if t.value is not None and math.isfinite(t.value)]
    table = pd.DataFrame(rows).sort_values("J", ascending=False).reset_index(drop=True)
    return table.head(top) if top else table
