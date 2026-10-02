"""Evaluacion unica en test de Act 07 v2 y criterios pre-registrados (ACT07_ROBUST.md).

Compara buy & hold, θ0, v1 (θ*, θ*_regimen) y v2 (θ* robusto, θ*_regimen
robusto) con summarize en train y test. Los modelos de regimen de test son
los ajustados con todo train (decision 7). Validation no se usa.

Criterios de lectura (seccion 8 y aclaraciones 10-12):
1. v2 cae menos que v1: J_ref -> J_test, en absoluta Y en relativa.
2. Ningun parametro final a <= 10% del ancho del rango de un limite.
"""

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig
from src.metrics import buy_and_hold_equity, calmar_ratio, summarize
from src.optimization import (
    PARAM_SPACE,
    SCALAR_EXIT_METRICS,
    THETA0,
    OptimizationResult,
    run_regime_strategy,
    run_theta,
    theta_config,
)
from src.regimes import RegimeModels
from src.splits import SPLITS
from src.walk_forward import (
    EXIT_PARAMS,
    WalkForwardResult,
    operating_thetas,
    walk_forward_regimes_run,
    walk_forward_run,
)

EDGE_FRACTION = 0.10
EDGE_TOL = 1e-12   # 3 - 2.8 = 0.2000000000000002 en punto flotante
REGIME_STRATEGIES = ["v1 θ*_régimen", "v2 θ*_régimen robusto"]
# Pares (v1, v2) del criterio 1, por fila de la tabla.
DROP_PAIRS = {"θ*": ("v1 θ*", "v2 θ* robusto"),
              "θ*_régimen": ("v1 θ*_régimen", "v2 θ*_régimen robusto")}


def robust_strategy_runs(df: pd.DataFrame, models: RegimeModels, opt_v1: OptimizationResult,
                         wf: WalkForwardResult) -> dict:
    """Corre las 6 estrategias sobre df completo (train + test, una sola corrida cada una).

    Regresa
    -------
    dict
        Nombre -> (equity DataFrame, trades DataFrame, BacktestConfig para summarize).
    """
    base = BacktestConfig()
    theta_v2 = wf.thetas["global"]
    runs = {
        "θ0": (run_theta(df, THETA0), theta_config(THETA0)),
        "v1 θ*": (run_theta(df, opt_v1.theta_star), theta_config(opt_v1.theta_star)),
        "v1 θ*_régimen": (run_regime_strategy(df, models, opt_v1.regime_thetas), base),
        "v2 θ* robusto": (run_theta(df, theta_v2), theta_config(theta_v2)),
        "v2 θ*_régimen robusto": (run_regime_strategy(df, models, operating_thetas(wf)), base),
    }
    no_trades = runs["θ0"][0].trades.iloc[0:0]
    out = {"buy & hold": (buy_and_hold_equity(df, base.initial_cash, base.cost_rate), no_trades, base)}
    out.update({name: (run.equity, run.trades, config) for name, (run, config) in runs.items()})
    return out


def robust_comparison_table(runs: dict, periods: tuple = ("train", "test")) -> pd.DataFrame:
    """summarize de cada estrategia y periodo; columnas "<estrategia> <periodo>".

    En las estrategias por regimen p_star, p_star_cost y k_mean suponen un
    sl/tp escalar y quedan NaN (como en v1).
    """
    columns = {}
    for name, (equity, trades, config) in runs.items():
        for period in periods:
            stats = summarize(equity, trades, config, *SPLITS[period])
            if name in REGIME_STRATEGIES:
                stats[SCALAR_EXIT_METRICS] = np.nan
            columns[f"{name} {period}"] = stats
    return pd.DataFrame(columns)


def walk_forward_j(wf: WalkForwardResult) -> dict:
    """J walk-forward (Calmar OOS encadenado) de v2 θ* robusto y de la combinada (aclaracion 10)."""
    return {
        "θ*": calmar_ratio(walk_forward_run(wf.blocks_data, wf.thetas["global"]).equity["equity"]),
        "θ*_régimen": calmar_ratio(
            walk_forward_regimes_run(wf.blocks_data, operating_thetas(wf)).equity["equity"]),
    }


def drop(j_ref: float, j_test: float) -> tuple[float, float]:
    """Caida de J: absoluta = J_ref - J_test; relativa = (J_ref - J_test) / J_ref."""
    absolute = j_ref - j_test
    return absolute, absolute / j_ref


def drop_criterion(v1_ref: float, v1_test: float, v2_ref: float, v2_test: float) -> dict:
    """Criterio 1 para un par: se cumple si v2 cae menos que v1 en absoluta Y en relativa."""
    v1_abs, v1_rel = drop(v1_ref, v1_test)
    v2_abs, v2_rel = drop(v2_ref, v2_test)
    return {"v1 J_ref": v1_ref, "v1 J_test": v1_test, "v1 caída abs": v1_abs, "v1 caída rel": v1_rel,
            "v2 J_ref": v2_ref, "v2 J_test": v2_test, "v2 caída abs": v2_abs, "v2 caída rel": v2_rel,
            "abs cumple": v2_abs < v1_abs, "rel cumple": v2_rel < v1_rel,
            "cumple": bool(v2_abs < v1_abs and v2_rel < v1_rel)}


def drop_table(table: pd.DataFrame, wf_j: dict) -> pd.DataFrame:
    """Criterio 1 por fila (θ*, θ*_régimen).

    v1: J_ref = Calmar en train, J_test = Calmar en test (de robust_comparison_table).
    v2: J_ref = J walk-forward (walk_forward_j), J_test = Calmar en test.
    """
    calmar = table.loc["calmar"]
    rows = {row: drop_criterion(calmar[f"{v1} train"], calmar[f"{v1} test"], wf_j[row], calmar[f"{v2} test"])
            for row, (v1, v2) in DROP_PAIRS.items()}
    return pd.DataFrame(rows).T


def edge_params(thetas: dict, fraction: float = EDGE_FRACTION) -> pd.DataFrame:
    """Criterio 2: distancia de cada parametro final al limite mas cercano de su rango.

    en_borde = distancia <= fraction · (max - min). Los estudios con θ None
    (regimen apagado) se omiten.

    Parametros
    ----------
    thetas : dict
        Estudio -> θ (o None).

    Regresa
    -------
    pd.DataFrame
        Una fila por (estudio, parametro): valor, limite cercano, distancia / ancho, en_borde.
    """
    rows = []
    for name, theta in thetas.items():
        if theta is None:
            continue
        for param in EXIT_PARAMS:
            _, low, high = PARAM_SPACE[param]
            value = theta[param]
            nearest = low if value - low <= high - value else high
            distance = abs(value - nearest) / (high - low)
            rows.append({"estudio": name, "parámetro": param, "valor": value, "límite cercano": nearest,
                         "distancia / ancho": distance, "en_borde": bool(distance <= fraction + EDGE_TOL)})
    return pd.DataFrame(rows).set_index(["estudio", "parámetro"])
