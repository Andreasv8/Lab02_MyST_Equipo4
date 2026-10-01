"""Pruebas de src/optimization.py y de la parametrizacion de compute_features.

Datos: NVDA solo hasta el fin de test (src/splits.py); validation no se usa.
No se corren los 500 trials aqui: los θ por regimen del test de
truncamiento estan fijos a mano.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization import (
    THETA0,
    objective,
    optimize_theta,
    regime_strategy_inputs,
    run_regime_strategy,
    run_theta,
)
from src.regimes import fit_regime_models, regime_labels
from src.splits import SPLITS, get_split
from src.strategy import compute_features

TEST_END = SPLITS["test"][1]
DF = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "NVDA_daily.csv",
                 index_col="Date", parse_dates=True).loc[:TEST_END]
DF_TRAIN = get_split(DF, "train")
MODELS = fit_regime_models(DF)

# Cada 40 barras desde la 60, mas la ultima barra.
T_BARS = list(range(60, len(DF), 40)) + [len(DF) - 1]
ATOL = 1e-9

# θ por regimen fijos a mano (no optimizados) para probar la estrategia combinada.
HAND_REGIME_THETAS = {
    "crisis": {"roc_window": 5, "cmf_window": 30, "adx_threshold": 20.0,
               "sl_mult": 1.5, "rr": 2.0, "max_holding": 6},
    "trend": THETA0,
    "mean_reversion": None,
}


def test_compute_features_defaults_unchanged():
    """Con los defaults explicitos el resultado es identico al de siempre."""
    pd.testing.assert_frame_equal(compute_features(DF),
                                  compute_features(DF, roc_window=10, cmf_window=20,
                                                   adx_window=14, adx_threshold=25))


def test_compute_features_uses_params():
    """Cambiar la ventana de ROC cambia la columna roc_10 (el nombre se conserva)."""
    default = compute_features(DF)["roc_10"]
    short = compute_features(DF, roc_window=5)["roc_10"]
    assert not np.allclose(default.dropna(), short.loc[default.dropna().index])


def test_objective_discards_few_trades():
    """Con menos de min_trades trades cerrados, J = -inf."""
    result = run_theta(DF_TRAIN, THETA0)
    assert objective(result, min_trades=len(result.trades) + 1) == -math.inf
    assert np.isfinite(objective(result, min_trades=len(result.trades)))


def test_run_theta_allowed_only_enters_in_regime():
    """Con allowed = (etiqueta == trend), toda entrada tiene etiqueta trend en la barra de señal."""
    labels = regime_labels(DF_TRAIN, MODELS)["hmm"]
    result = run_theta(DF_TRAIN, THETA0, allowed=labels == "trend")
    assert len(result.trades) > 0
    assert (labels.iloc[result.trades["entry_bar"] - 1] == "trend").all()


def test_regime_inputs_follow_regime():
    """Sin θ (mean_reversion) o en warm-up la señal es 0; en crisis los parametros son los de crisis."""
    inputs = regime_strategy_inputs(DF, MODELS, HAND_REGIME_THETAS)
    off = inputs["regime"].isna() | (inputs["regime"] == "mean_reversion")
    assert (inputs.loc[off, "signal"] == 0).all()
    crisis = inputs["regime"] == "crisis"
    assert (inputs.loc[crisis, "sl_mult"] == 1.5).all()
    assert (inputs.loc[crisis, "tp_mult"] == 3.0).all()
    assert (inputs.loc[crisis, "max_holding"] == 6).all()


FULL_INPUTS = regime_strategy_inputs(DF, MODELS, HAND_REGIME_THETAS)
FULL_RUN = run_regime_strategy(DF, MODELS, HAND_REGIME_THETAS)


@pytest.mark.parametrize("t", T_BARS)
def test_regime_strategy_is_causal(t):
    """Estrategia combinada: señal, parametros y equity en t, y trades cerrados hasta t,
    iguales con df.iloc[:t+1] (modelos y θ ya fijos)."""
    truncated_df = DF.iloc[:t + 1]
    cols = ["signal", "atr", "sl_mult", "tp_mult", "max_holding"]
    np.testing.assert_allclose(
        FULL_INPUTS.iloc[t][cols].to_numpy(dtype=float),
        regime_strategy_inputs(truncated_df, MODELS, HAND_REGIME_THETAS).iloc[-1][cols].to_numpy(dtype=float),
        rtol=0, atol=ATOL, equal_nan=True)

    truncated = run_regime_strategy(truncated_df, MODELS, HAND_REGIME_THETAS)
    np.testing.assert_allclose(FULL_RUN.equity.iloc[t].to_numpy(dtype=float),
                               truncated.equity.iloc[-1].to_numpy(dtype=float), rtol=0, atol=ATOL)
    full_trades = FULL_RUN.trades[FULL_RUN.trades["exit_bar"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_trades, truncated.trades.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=ATOL)


def test_optimize_theta_reproducible():
    """Misma semilla -> mismos parametros y mismo J (8 trials, 4 aleatorios)."""
    runs = [optimize_theta(DF_TRAIN, min_trades=5, n_trials=8, n_startup_trials=4) for _ in range(2)]
    assert runs[0].best_params == runs[1].best_params
    assert runs[0].best_value == runs[1].best_value



def test_theta_table_hand_thetas():
    """theta_table con θ fijos (sin Optuna): J de θ0 coincide con objective y tp = sl · rr."""
    from src.optimization import MIN_TRADES, OptimizationResult, theta_table

    regime_thetas = {"crisis": HAND_REGIME_THETAS["crisis"], "trend": THETA0, "mean_reversion": THETA0}
    opt = OptimizationResult(theta_star=THETA0, j_star=np.nan, regime_thetas=regime_thetas,
                             regime_j={}, studies={})
    table = theta_table(DF, MODELS, opt)

    assert list(table.index) == ["θ0", "θ*", "θ*_crisis", "θ*_trend", "θ*_mean_reversion"]
    assert table.loc["θ0", "J_train"] == pytest.approx(objective(run_theta(DF_TRAIN, THETA0), MIN_TRADES))
    assert table.loc["θ*_crisis", "tp_mult"] == pytest.approx(1.5 * 2.0)
    assert table["operates"].all()
