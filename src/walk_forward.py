"""Optimizacion robusta walk-forward de lab_02 (Act 07 v2, ACT07_ROBUST.md).

La señal queda fija en θ0 (ROC 10, CMF 20, ADX 14 > 25) y solo se optimiza la
salida (sl_mult, rr, max_holding). J = Calmar de la equity fuera de muestra:
los 3 tramos de evaluacion del walk-forward (ventana creciente dentro de
train) encadenados.

Anti look-ahead: en cada bloque el scaler, el umbral p90 y el HMM se ajustan
UNA vez con datos <= fin del tramo de ajuste y se reutilizan en todos los
trials. Las etiquetas del tramo de evaluacion son las filtradas (causales),
calculadas en las barras hh:00 y mantenidas el resto de la hora.
Test y validation no se usan aqui.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.backtest import BacktestConfig, BacktestResult, backtest
from src.optimization import PARAM_SPACE, SEED, THETA0, objective, theta_config, theta_signal
from src.regimes import (
    REGIME_NAMES,
    RegimeScaler,
    _unscale,
    classify_hmm_filtered,
    fit_hmm,
    fit_rule_thresholds,
    fit_scaler,
    hourly,
    regime_features,
    to_bars,
)
from src.splits import SPLITS

FIT_START = SPLITS["train"][0]
N_TRIALS = 200
N_STARTUP_TRIALS = 100
PLATEAU_FRACTION = 0.10
MIN_TRADES_GLOBAL = 20   # θ* robusto
MIN_TRADES_REGIME = 10   # θ*_j robusto
EXIT_PARAMS = ["sl_mult", "rr", "max_holding"]
STUDY_NAMES = ["global"] + REGIME_NAMES


@dataclass(frozen=True)
class Block:
    """Un bloque del walk-forward: ajuste FIT_START..fit_end, evaluacion eval_start..eval_end."""

    fit_end: str
    eval_start: str
    eval_end: str


# Ventana creciente dentro de train de BTC (2022-06-01 .. 2023-05-14): tres
# tramos de evaluacion de 2 meses; el ultimo termina en el fin de train.
# Las fechas son dias completos (df.loc[:"2023-01-14"] incluye todo ese dia).
BLOCKS = [
    Block("2022-11-14", "2022-11-15", "2023-01-14"),
    Block("2023-01-14", "2023-01-15", "2023-03-14"),
    Block("2023-03-14", "2023-03-15", "2023-05-14"),
]


# ---------------------------------------------------------------------------
# Modelos y datos por bloque (una sola vez por bloque, no por trial)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockModels:
    """Lo ajustado con los datos de ajuste de un bloque."""

    scaler: RegimeScaler
    vol_threshold: float
    hmm: GaussianHMM
    hmm_names: dict
    hmm_choice: str


def fit_block_models(df: pd.DataFrame, fit_end: str) -> BlockModels:
    """Ajusta scaler, umbral p90 y HMM (cascada de 10 semillas) con datos FIT_START..fit_end.

    Las features se calculan sobre df.loc[:fit_end], asi que nada posterior a
    fit_end entra al ajuste aunque df sea mas largo. Se ajusta en las barras
    hh:00 (hourly), igual que fit_regime_models.

    Parametros
    ----------
    df : pd.DataFrame
        Precios con columna "Close".
    fit_end : str
        Ultimo dia del tramo de ajuste (inclusive).

    Regresa
    -------
    BlockModels
    """
    features = hourly(regime_features(df.loc[:fit_end]).loc[FIT_START:fit_end])
    scaler = fit_scaler(features)
    hmm, names, choice = fit_hmm(features, scaler)
    return BlockModels(scaler, fit_rule_thresholds(features), hmm, names, choice)


def block_centroids(models: BlockModels) -> pd.DataFrame:
    """Medias del HMM en unidades originales, una fila por regimen (nombre)."""
    centers = _unscale(models.hmm.means_, models.scaler)
    centers.index = [models.hmm_names[i] for i in centers.index]
    return centers.loc[REGIME_NAMES]


@dataclass(frozen=True)
class BlockData:
    """Insumos del tramo de evaluacion, precalculados una vez por bloque."""

    df_eval: pd.DataFrame
    signal: pd.Series
    atr: pd.Series
    labels: pd.Series


def prepare_block(df: pd.DataFrame, block: Block, models: BlockModels) -> BlockData:
    """Señal θ0, ATR y etiqueta HMM filtrada del tramo de evaluacion.

    Todo se calcula sobre df.loc[:eval_end] (nada posterior al tramo) y se
    recorta a eval_start..eval_end. El backtest corre solo sobre el tramo:
    el motor entra en t con la señal de t-1, asi que el tramo arranca plano y
    solo hay entradas con barra de señal dentro del tramo (decision 4).
    """
    df_trunc = df.loc[:block.eval_end]
    signal, atr = theta_signal(df_trunc, THETA0)
    hourly_labels = classify_hmm_filtered(hourly(regime_features(df_trunc)), models.scaler,
                                          models.hmm, models.hmm_names)
    labels = to_bars(hourly_labels, df_trunc.index)
    window = slice(block.eval_start, block.eval_end)
    return BlockData(df_trunc.loc[window], signal.loc[window], atr.loc[window], labels.loc[window])


# ---------------------------------------------------------------------------
# Backtest walk-forward
# ---------------------------------------------------------------------------

def run_block(data: BlockData, theta: dict, regime: Optional[str] = None) -> BacktestResult:
    """Backtest del tramo de evaluacion con la salida de θ.

    Con regime, la señal se pone en 0 donde la etiqueta filtrada != regime
    (solo hay entradas con barra de señal en ese regimen).
    """
    signal = data.signal if regime is None else data.signal.where(data.labels == regime, 0)
    return backtest(data.df_eval, signal, data.atr, theta_config(theta))


def chain_equity(equities: list, initial_cash: float) -> pd.Series:
    """Encadena las equities de varios tramos por rendimientos (no por niveles).

    Cada tramo arranca con initial_cash; su rendimiento por barra es
    E_t / E_{t-1} - 1 (la primera barra contra initial_cash). La equity
    encadenada es initial_cash · Π(1 + r).

    Parametros
    ----------
    equities : list de pd.Series
        Equity por barra de cada tramo, en orden cronologico.
    initial_cash : float
        Capital inicial de cada tramo.

    Regresa
    -------
    pd.Series
        Equity encadenada sobre las fechas de todos los tramos.
    """
    returns = pd.concat([eq / eq.shift(1).fillna(initial_cash) - 1 for eq in equities])
    return initial_cash * (1 + returns).cumprod()


def walk_forward_run(blocks_data: list, theta: dict, regime: Optional[str] = None) -> BacktestResult:
    """Corre θ en los tramos de evaluacion y regresa la equity encadenada y los trades juntos."""
    runs = [run_block(data, theta, regime) for data in blocks_data]
    equity = chain_equity([r.equity["equity"] for r in runs], theta_config(theta).initial_cash)
    trades = pd.concat([r.trades for r in runs], ignore_index=True)
    return BacktestResult(equity=equity.to_frame("equity"), trades=trades)



def run_block_regimes(data: BlockData, regime_thetas: dict) -> BacktestResult:
    """Backtest del tramo con la estrategia combinada por regimen.

    En cada barra de señal con etiqueta filtrada j y θ_j (no None) la señal es
    la de θ0 y sl/tp/holding son los de θ_j; el motor los fija al entrar. En
    regimenes sin θ (o apagados) la señal es 0 y la salida queda con θ0 de
    relleno (nunca se usa porque no hay entrada), como en v1.
    """
    active = [name for name, theta in regime_thetas.items() if theta is not None]
    signal = data.signal.where(data.labels.isin(active), 0)
    index = data.signal.index
    sl = pd.Series(float(THETA0["sl_mult"]), index=index)
    tp = pd.Series(float(THETA0["sl_mult"] * THETA0["rr"]), index=index)
    holding = pd.Series(float(THETA0["max_holding"]), index=index)
    for name in active:
        theta, mask = regime_thetas[name], data.labels == name
        sl[mask] = theta["sl_mult"]
        tp[mask] = theta["sl_mult"] * theta["rr"]
        holding[mask] = int(theta["max_holding"])
    return backtest(data.df_eval, signal, data.atr, BacktestConfig(),
                    sl_mult=sl, tp_mult=tp, max_holding=holding)


def walk_forward_regimes_run(blocks_data: list, regime_thetas: dict) -> BacktestResult:
    """Estrategia combinada en los tramos de evaluacion: equity encadenada y trades juntos."""
    runs = [run_block_regimes(data, regime_thetas) for data in blocks_data]
    equity = chain_equity([r.equity["equity"] for r in runs], BacktestConfig().initial_cash)
    trades = pd.concat([r.trades for r in runs], ignore_index=True)
    return BacktestResult(equity=equity.to_frame("equity"), trades=trades)

# ---------------------------------------------------------------------------
# Busqueda y meseta
# ---------------------------------------------------------------------------

def suggest_exit(trial: optuna.Trial) -> dict:
    """θ0 con sl_mult, rr y max_holding propuestos en los rangos de PARAM_SPACE."""
    theta = dict(THETA0)
    for name in EXIT_PARAMS:
        kind, low, high = PARAM_SPACE[name]
        suggest = trial.suggest_int if kind == "int" else trial.suggest_float
        theta[name] = suggest(name, low, high)
    return theta


def optimize_exit(blocks_data: list, min_trades: int, regime: Optional[str] = None,
                  n_trials: int = N_TRIALS, n_startup_trials: int = N_STARTUP_TRIALS,
                  seed: int = SEED, study_name: Optional[str] = None) -> optuna.Study:
    """Maximiza J walk-forward con TPESampler(n_startup_trials=100, seed=42): 100 random + 100 TPE.

    Parametros
    ----------
    blocks_data : list de BlockData
        Tramos de evaluacion precalculados.
    min_trades : int
        Minimo de trades (en los tramos juntos) para que J sea valido.
    regime : str, opcional
        Si se da, solo hay entradas con etiqueta filtrada == regime.

    Regresa
    -------
    optuna.Study
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    study.optimize(lambda trial: objective(walk_forward_run(blocks_data, suggest_exit(trial), regime),
                                           min_trades),
                   n_trials=n_trials)
    return study


def round_half_up(x: float) -> int:
    """Entero mas cercano; x.5 hacia arriba (decision 9 de ACT07_ROBUST.md)."""
    return int(math.floor(x + 0.5))


def top_trials(study: optuna.Study, fraction: float = PLATEAU_FRACTION) -> list:
    """Top fraction de los trials por J (ceil(fraction · n)); solo J finito.

    Si hay menos trials finitos que el tamaño del top, se usan los disponibles.
    Empates en J se ordenan por numero de trial.
    """
    finite = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    n_top = math.ceil(fraction * len(study.trials))
    return sorted(finite, key=lambda t: (-t.value, t.number))[:n_top]


def plateau_theta(study: optuna.Study, fraction: float = PLATEAU_FRACTION) -> Optional[dict]:
    """θ de la meseta: mediana por parametro del top fraction; max_holding half-up.

    Regresa None si no hay trials con J finito.
    """
    top = top_trials(study, fraction)
    if not top:
        return None
    theta = dict(THETA0)
    for name in EXIT_PARAMS:
        median = float(np.median([t.params[name] for t in top]))
        theta[name] = round_half_up(median) if PARAM_SPACE[name][0] == "int" else median
    return theta


def operates(study: optuna.Study) -> bool:
    """Regla a priori: se opera solo si el mejor J > 0 (J = -inf si nadie alcanzo el minimo de trades)."""
    return bool(study.best_value > 0)


# ---------------------------------------------------------------------------
# Orquestacion y reporte
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Resultado de run_walk_forward.

    Atributos
    ---------
    models : list de BlockModels
        Modelos de regimen de cada bloque.
    blocks_data : list de BlockData
        Tramos de evaluacion.
    studies : dict
        "global" y cada regimen -> optuna.Study.
    thetas : dict
        Estudio -> θ de la meseta (None si no hubo J finito).
    operates : dict
        Estudio -> True si se opera (para "global" siempre True).
    """

    models: list
    blocks_data: list
    studies: dict
    thetas: dict
    operates: dict


def study_min_trades(name: str) -> int:
    """Minimo de trades del estudio: 20 para global, 10 por regimen."""
    return MIN_TRADES_GLOBAL if name == "global" else MIN_TRADES_REGIME


def run_walk_forward(df: pd.DataFrame, n_trials: int = N_TRIALS,
                     n_startup_trials: int = N_STARTUP_TRIALS) -> WalkForwardResult:
    """Walk-forward completo: modelos por bloque, 4 estudios y θ de la meseta.

    Parametros
    ----------
    df : pd.DataFrame
        OHLCV; solo se usa hasta el fin del ultimo tramo de evaluacion (fin de train).

    Regresa
    -------
    WalkForwardResult
    """
    models = [fit_block_models(df, block.fit_end) for block in BLOCKS]
    blocks_data = [prepare_block(df, block, m) for block, m in zip(BLOCKS, models)]

    studies, thetas, operate = {}, {}, {}
    for name in STUDY_NAMES:
        regime = None if name == "global" else name
        study = optimize_exit(blocks_data, study_min_trades(name), regime, n_trials,
                              n_startup_trials, study_name=name)
        studies[name] = study
        thetas[name] = plateau_theta(study)
        operate[name] = True if regime is None else operates(study)
    return WalkForwardResult(models, blocks_data, studies, thetas, operate)


def operating_thetas(result: WalkForwardResult) -> dict:
    """Regimen -> θ de la meseta, o None si la regla a priori lo apaga."""
    return {name: result.thetas[name] if result.operates[name] else None for name in REGIME_NAMES}


def blocks_table(result: WalkForwardResult) -> pd.DataFrame:
    """Por bloque: cascada HMM, centroides (unidades originales), p90 y barras por regimen en el tramo."""
    rows = {}
    for i, (block, models, data) in enumerate(zip(BLOCKS, result.models, result.blocks_data), 1):
        row = {"fit": f"{FIT_START}..{block.fit_end}",
               "eval": f"{block.eval_start}..{block.eval_end}",
               "hmm_choice": models.hmm_choice, "vol_p90": models.vol_threshold}
        for regime, center in block_centroids(models).iterrows():
            for col, value in center.items():
                row[f"{regime} {col}"] = value
            row[f"{regime} eval_bars"] = int((data.labels == regime).sum())
        rows[f"bloque {i}"] = row
    return pd.DataFrame(rows)


def summary_table(result: WalkForwardResult) -> pd.DataFrame:
    """Por estudio: mejor trial, trials finitos, θ de la meseta con su J y trades, rango del top."""
    rows = {}
    for name, study in result.studies.items():
        regime = None if name == "global" else name
        top = top_trials(study)
        theta = result.thetas[name]
        row = {"min_trades": study_min_trades(name), "best_J": study.best_value,
               "n_finite": len(top_trials(study, 1.0)), "n_top": len(top),
               "operates": result.operates[name]}
        for p in EXIT_PARAMS:
            row[f"best {p}"] = study.best_params[p]
        if theta is not None:
            run = walk_forward_run(result.blocks_data, theta, regime)
            row["plateau_J"] = objective(run, study_min_trades(name))
            row["plateau_trades"] = len(run.trades)
            for p in EXIT_PARAMS:
                values = [t.params[p] for t in top]
                row[f"plateau {p}"] = theta[p]
                row[f"top {p} min"] = min(values)
                row[f"top {p} max"] = max(values)
        rows[name] = row
    return pd.DataFrame(rows)
