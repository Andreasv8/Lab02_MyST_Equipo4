"""Optimizacion de parametros de la estrategia de lab_02 (Act 07, punto 7).

Estrategia: cruce de EMAs con filtro ADX, calculada en barras de 1h/4h y
ejecutada en barras de 5 minutos (compute_ema_adx_features).

θ = (timeframe, ema_fast, ema_slow, adx_threshold, entry_on_change, sl_mult,
rr, max_holding), con tp_mult = sl_mult · rr. ADX(14), ATR(14) del timeframe,
ρ = 1% y costos quedan FIJOS. max_holding esta en barras de 5 minutos.

Se optimiza SOLO con train y de dos formas:
- θ* unico: maximiza J(backtest(train, θ)).
- θ*_j por regimen (HMM filtrado): maximiza J del backtest en train donde
  solo se permiten entradas cuando la etiqueta de la barra de señal es j.
Despues la estrategia combinada usa en cada barra el θ del regimen filtrado.
Test solo se evalua; validation no se carga.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
import pandas as pd

from src.backtest import BacktestConfig, BacktestResult, backtest
from src.metrics import buy_and_hold_equity, calmar_ratio, summarize
from src.regimes import REGIME_NAMES, RegimeModels, regime_labels
from src.splits import SPLITS, get_split
from src.signals import compute_ema_adx_features

SEED = 42
N_TRIALS = 500
N_STARTUP_TRIALS = 150
MIN_TRADES = 20          # θ* unico
MIN_TRADES_REGIME = 10   # θ*_j por regimen
BARS_PER_HOUR = 12

# θ0 = θ robusta elegida SOLO con train (src/ema_search.py): centro de la meseta
# del top 10% de 300 trials de Optuna, con ADX = 25 (Wilder) porque ADX ~27 era
# un pico aislado, y sin take-profit efectivo (rr = 20) porque Calmar crece
# monotono con rr. En train: +28.6%, Sharpe 2.0, max DD -11.9%, Calmar 2.6,
# 46 trades; 96% de las 108 θ vecinas tienen retorno > 0.
THETA0 = {
    "timeframe": "4h",
    "ema_fast": 12,
    "ema_slow": 18,
    "adx_threshold": 25.0,
    "entry_on_change": True,
    "sl_mult": 3.0,
    "rr": 20.0,
    "max_holding": 5 * 24 * BARS_PER_HOUR,   # 5 dias
}

# Espacio de busqueda: nombre -> ("cat", opciones) o (tipo, minimo, maximo).
# ema_slow no se busca directo: ema_slow = round(ema_fast · slow_ratio) > ema_fast.
# sl_mult esta en ATR del timeframe (el ATR de 4h es ~0.5% del precio, contra un
# costo de ida y vuelta de 0.35%). rr llega a 20 (~sin TP): con tope 10 el mejor
# trial quedaba en el borde (9.6).
PARAM_SPACE = {
    "timeframe": ("cat", ["1h", "4h"]),
    "ema_fast": ("int", 5, 60),
    "slow_ratio": ("float", 1.5, 5.0),
    "adx_threshold": ("float", 15.0, 35.0),
    "entry_on_change": ("cat", [True, False]),
    "sl_mult": ("float", 1.0, 4.0),
    "rr": ("float", 1.0, 20.0),
    "max_holding": ("int", 4 * BARS_PER_HOUR, 14 * 24 * BARS_PER_HOUR),   # 4 horas a 14 dias
}

# Metricas de summarize que suponen un sl/tp escalar (no aplican si cambian por regimen).
SCALAR_EXIT_METRICS = ["p_star", "p_star_cost", "k_mean"]


def params_to_theta(params: dict) -> dict:
    """Convierte parametros de Optuna (con slow_ratio) en θ (con ema_slow).

    Si params ya trae ema_slow (θ completo), lo regresa sin cambios.
    """
    if "slow_ratio" not in params:
        return dict(params)
    theta = {k: v for k, v in params.items() if k != "slow_ratio"}
    theta["ema_slow"] = max(int(params["ema_fast"]) + 1,
                            round(params["ema_fast"] * params["slow_ratio"]))
    return theta


def theta_signal(df: pd.DataFrame, theta: dict) -> tuple[pd.Series, pd.Series]:
    """Señal EMA + ADX y ATR del timeframe de θ, alineados a las barras de 5 minutos.

    Regresa
    -------
    tuple[pd.Series, pd.Series]
        (señal 1/-1/0 por barra, ATR(14) del timeframe por barra).
    """
    features = compute_ema_adx_features(df, ema_fast=int(theta["ema_fast"]),
                                        ema_slow=int(theta["ema_slow"]),
                                        adx_threshold=theta["adx_threshold"],
                                        timeframe=theta["timeframe"],
                                        entry_on_change=bool(theta["entry_on_change"]))
    return features["signal"], features["atr"]


def theta_config(theta: dict) -> BacktestConfig:
    """BacktestConfig con la salida de θ (tp = sl · rr); sizing y costos por defecto."""
    return BacktestConfig(sl_mult=theta["sl_mult"], tp_mult=theta["sl_mult"] * theta["rr"],
                          max_holding=int(theta["max_holding"]))


def run_theta(df: pd.DataFrame, theta: dict, allowed: Optional[pd.Series] = None) -> BacktestResult:
    """Backtest de la estrategia con parametros θ.

    Parametros
    ----------
    df : pd.DataFrame
        OHLC de 5 minutos con indice de fechas.
    theta : dict
        Parametros (ver THETA0).
    allowed : pd.Series, opcional
        Booleana por barra. Donde es False la señal se pone en 0. Como el
        motor ejecuta en t la señal de t-1, solo hay ENTRADAS cuando la barra
        de señal esta permitida (p. ej. etiqueta filtrada == j). En barras no
        permitidas tampoco hay salida por señal opuesta; SL, TP y holding
        maximo siguen activos.

    Regresa
    -------
    BacktestResult
    """
    signal, atr = theta_signal(df, theta)
    if allowed is not None:
        signal = signal.where(allowed.reindex(df.index, fill_value=False).astype(bool), 0)
    return backtest(df, signal, atr, theta_config(theta))


def objective(result: BacktestResult, min_trades: int) -> float:
    """J = Calmar ratio (CAGR / |max DD|) de la curva de equity.

    Restriccion: con menos de min_trades trades cerrados, o Calmar no finito
    (sin drawdown), J = -inf y la configuracion se descarta. Evita premiar θ
    con 2 o 3 trades afortunados.
    """
    if len(result.trades) < min_trades:
        return -math.inf
    j = calmar_ratio(result.equity["equity"])
    return float(j) if np.isfinite(j) else -math.inf


def suggest_theta(trial: optuna.Trial) -> dict:
    """Propone un θ dentro de PARAM_SPACE; ema_slow sale de slow_ratio (params_to_theta)."""
    params = {}
    for name, spec in PARAM_SPACE.items():
        if spec[0] == "cat":
            params[name] = trial.suggest_categorical(name, spec[1])
        elif spec[0] == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            params[name] = trial.suggest_float(name, spec[1], spec[2])
    return params_to_theta(params)


def best_theta(study: optuna.Study) -> dict:
    """θ del mejor trial del estudio (con ema_slow en vez de slow_ratio)."""
    return params_to_theta(study.best_params)


def optimize_theta(df_train: pd.DataFrame, min_trades: int, allowed: Optional[pd.Series] = None,
                   n_trials: int = N_TRIALS, n_startup_trials: int = N_STARTUP_TRIALS,
                   seed: int = SEED, study_name: Optional[str] = None,
                   enqueue: Optional[list] = None) -> optuna.Study:
    """Maximiza J sobre train con Optuna: random search y despues TPE (bayesiano).

    TPESampler(n_startup_trials=150, seed=42):
    - Trials 1..150: θ aleatorio uniforme en PARAM_SPACE (random search).
      Explora todo el espacio sin suponer nada y da la base para el modelo.
    - Trials 151..n: TPE (Tree-structured Parzen Estimator). Parte los trials
      vistos en "buenos" (mejor cuantil de J) y "malos", estima con kernels
      las densidades l(θ) = p(θ | bueno) y g(θ) = p(θ | malo) y propone el θ
      que maximiza l(θ) / g(θ): busqueda bayesiana que se concentra en las
      zonas prometedoras sin abandonar del todo la exploracion.
    Los trials con J = -inf (pocos trades) cuentan como los peores.

    Parametros
    ----------
    df_train : pd.DataFrame
        OHLC de 5 minutos de train unicamente.
    min_trades : int
        Minimo de trades para que J sea valido.
    allowed : pd.Series, opcional
        Barras de señal permitidas (ver run_theta).
    n_trials, n_startup_trials, seed : int
        Presupuesto de busqueda y semilla.
    enqueue : list[dict], opcional
        Parametros (formato de PARAM_SPACE, con slow_ratio) a evaluar primero.

    Regresa
    -------
    optuna.Study
        Estudio completo (best_params, best_value, trials). Usar best_theta
        para obtener θ con ema_slow.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    for params in enqueue or []:
        study.enqueue_trial(params)
    study.optimize(lambda trial: objective(run_theta(df_train, suggest_theta(trial), allowed),
                                           min_trades),
                   n_trials=n_trials)
    return study


@dataclass
class OptimizationResult:
    """Resultado de optimize_all.

    Atributos
    ---------
    theta_star : dict
        θ* unico.
    j_star : float
        J(θ*) en train.
    regime_thetas : dict
        Regimen -> θ*_j, o None si en ese regimen no se opera.
    regime_j : dict
        Regimen -> mejor J_j en train (puede ser <= 0 o -inf).
    studies : dict
        "global" y cada regimen -> optuna.Study (para convergencia e importancia).
    """

    theta_star: dict
    j_star: float
    regime_thetas: dict
    regime_j: dict
    studies: dict


def optimize_all(df: pd.DataFrame, models: RegimeModels, n_trials: int = N_TRIALS,
                 n_startup_trials: int = N_STARTUP_TRIALS) -> OptimizationResult:
    """Corre los 4 estudios (θ* unico y θ*_j por regimen) SOLO con train.

    Regla a priori por regimen: si el mejor J_j <= 0 o ningun θ alcanzo
    MIN_TRADES_REGIME (J_j = -inf), en ese regimen NO se opera (θ_j = None).

    Parametros
    ----------
    df : pd.DataFrame
        OHLCV; solo se usa su tramo de train.
    models : RegimeModels
        Modelos de regimen ya ajustados en train (se usa la etiqueta "hmm").

    Regresa
    -------
    OptimizationResult
    """
    df_train = get_split(df, "train")
    labels = regime_labels(df_train, models)["hmm"]

    studies = {"global": optimize_theta(df_train, MIN_TRADES, n_trials=n_trials,
                                        n_startup_trials=n_startup_trials, study_name="global")}
    regime_thetas, regime_j = {}, {}
    for name in REGIME_NAMES:
        study = optimize_theta(df_train, MIN_TRADES_REGIME, allowed=labels == name,
                               n_trials=n_trials, n_startup_trials=n_startup_trials,
                               study_name=name)
        studies[name] = study
        regime_j[name] = study.best_value
        regime_thetas[name] = best_theta(study) if study.best_value > 0 else None

    return OptimizationResult(theta_star=best_theta(studies["global"]),
                              j_star=studies["global"].best_value,
                              regime_thetas=regime_thetas, regime_j=regime_j, studies=studies)


def regime_strategy_inputs(df: pd.DataFrame, models: RegimeModels,
                           regime_thetas: dict) -> pd.DataFrame:
    """Señal y parametros de salida por barra de la estrategia combinada por regimen.

    En cada barra t, con la etiqueta filtrada ŝ_t (HMM forward, conocida al
    cierre de t): la señal es la de θ*_{ŝ_t} y sl/tp/holding son los de ese θ.
    El motor ejecuta en t+1 y fija sl/tp/holding al entrar. Si ŝ_t no tiene θ
    (no se opera) o es warm-up, la señal es 0; ahi sl/tp/holding quedan con
    θ0 como relleno, pero nunca se usan porque no hay entrada.

    Regresa
    -------
    pd.DataFrame
        Indice de df; columnas regime, signal, atr, sl_mult, tp_mult, max_holding.
    """
    labels = regime_labels(df, models)["hmm"]
    _, atr = theta_signal(df, THETA0)
    out = pd.DataFrame({
        "regime": labels,
        "signal": 0,
        "atr": atr,
        "sl_mult": THETA0["sl_mult"],
        "tp_mult": THETA0["sl_mult"] * THETA0["rr"],
        "max_holding": THETA0["max_holding"],
    }, index=df.index)

    for name, theta in regime_thetas.items():
        if theta is None:
            continue
        mask = labels == name
        signal, _ = theta_signal(df, theta)
        out.loc[mask, "signal"] = signal[mask]
        out.loc[mask, "sl_mult"] = theta["sl_mult"]
        out.loc[mask, "tp_mult"] = theta["sl_mult"] * theta["rr"]
        out.loc[mask, "max_holding"] = int(theta["max_holding"])
    return out


def run_regime_strategy(df: pd.DataFrame, models: RegimeModels, regime_thetas: dict) -> BacktestResult:
    """Backtest de la estrategia combinada (θ del regimen filtrado en cada barra)."""
    inputs = regime_strategy_inputs(df, models, regime_thetas)
    return backtest(df, inputs["signal"], inputs["atr"], BacktestConfig(),
                    sl_mult=inputs["sl_mult"], tp_mult=inputs["tp_mult"],
                    max_holding=inputs["max_holding"])


def strategy_runs(df: pd.DataFrame, models: RegimeModels, opt: OptimizationResult) -> dict:
    """Corre buy & hold, θ0, θ* y θ*_regimen sobre df completo (hasta fin de test).

    Regresa
    -------
    dict
        Nombre -> (equity DataFrame, trades DataFrame, BacktestConfig para summarize).
    """
    base_config = BacktestConfig()
    theta0_run = run_theta(df, THETA0)
    star_run = run_theta(df, opt.theta_star)
    regime_run = run_regime_strategy(df, models, opt.regime_thetas)
    no_trades = theta0_run.trades.iloc[0:0]
    bh_equity = buy_and_hold_equity(df, base_config.initial_cash, base_config.cost_rate)
    return {
        "buy & hold": (bh_equity, no_trades, base_config),
        "θ0": (theta0_run.equity, theta0_run.trades, theta_config(THETA0)),
        "θ*": (star_run.equity, star_run.trades, theta_config(opt.theta_star)),
        "θ*_régimen": (regime_run.equity, regime_run.trades, base_config),
    }


def comparison_table(df: pd.DataFrame, models: RegimeModels, opt: OptimizationResult,
                     periods: tuple = ("train", "test")) -> pd.DataFrame:
    """Metricas de summarize en train y test: buy & hold, θ0, θ* y θ*_regimen.

    Cada backtest corre sobre df completo (hasta fin de test) y se recorta por
    periodo con summarize, como en Act 06; por causalidad, el tramo de train
    es identico al backtest que se optimizo. En la estrategia por regimen,
    p_star, p_star_cost y k_mean suponen un sl/tp escalar y quedan NaN.

    Regresa
    -------
    pd.DataFrame
        Columnas "<estrategia> <periodo>", filas = metricas de summarize.
    """
    columns = {}
    for name, (equity, trades, config) in strategy_runs(df, models, opt).items():
        for period in periods:
            stats = summarize(equity, trades, config, *SPLITS[period])
            if name == "θ*_régimen":
                stats[SCALAR_EXIT_METRICS] = np.nan
            columns[f"{name} {period}"] = stats
    return pd.DataFrame(columns)


def theta_table(df: pd.DataFrame, models: RegimeModels, opt: OptimizationResult) -> pd.DataFrame:
    """θ0, θ* y θ*_j con su J en train, numero de trades en train y si se opera.

    Para θ*_j se reporta el mejor θ del estudio aunque la regla a priori lo
    apague (operates = False). tp_mult = sl_mult · rr se agrega como columna.

    Regresa
    -------
    pd.DataFrame
        Una fila por θ; columnas = parametros, tp_mult, J_train, trades_train, operates.
    """
    df_train = get_split(df, "train")
    labels = regime_labels(df_train, models)["hmm"]
    rows = {}

    def _row(theta: dict, allowed: Optional[pd.Series], min_trades: int, operates: bool) -> dict:
        result = run_theta(df_train, theta, allowed)
        return {**theta, "tp_mult": theta["sl_mult"] * theta["rr"],
                "J_train": objective(result, min_trades), "trades_train": len(result.trades),
                "operates": operates}

    rows["θ0"] = _row(THETA0, None, MIN_TRADES, True)
    rows["θ*"] = _row(opt.theta_star, None, MIN_TRADES, True)
    for name in REGIME_NAMES:
        best = best_theta(opt.studies[name]) if name in opt.studies else opt.regime_thetas[name]
        rows[f"θ*_{name}"] = _row(best, labels == name, MIN_TRADES_REGIME,
                                  opt.regime_thetas[name] is not None)
    return pd.DataFrame.from_dict(rows, orient="index")


def convergence_curve(study: optuna.Study) -> pd.Series:
    """Mejor J acumulado por numero de trial (-inf hasta el primer θ valido)."""
    values = np.array([t.value for t in study.trials], dtype=float)
    return pd.Series(np.maximum.accumulate(values), index=np.arange(1, len(values) + 1),
                     name=study.study_name)


def param_importances(study: optuna.Study) -> pd.Series:
    """Importancia relativa de cada parametro de θ en J (fANOVA de Optuna, suma 1)."""
    return pd.Series(optuna.importance.get_param_importances(study),
                     name=study.study_name)
