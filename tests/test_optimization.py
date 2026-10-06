"""Pruebas de src/optimization.py con la estrategia EMA + ADX.

Datos: BTC de 5 minutos solo hasta el fin de test (src/splits.py); validation
no se usa. No se corren los 500 trials aqui: los θ por regimen de la prueba de
truncamiento estan fijos a mano.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_btc
from src.optimization import (
    PARAM_SPACE,
    THETA0,
    objective,
    optimize_theta,
    params_to_theta,
    regime_strategy_inputs,
    run_regime_strategy,
    run_theta,
    theta_signal,
)
from src.regimes import fit_regime_models, regime_labels
from src.splits import SPLITS, get_split
from src.strategy import compute_ema_adx_features

TEST_END = SPLITS["test"][1]
DF = load_btc(str(Path(__file__).resolve().parents[1] / "data" / "btc_project_train.csv")).loc[:TEST_END]
DF_TRAIN = get_split(DF, "train")
MODELS = fit_regime_models(DF)

# 8 barras repartidas en train y test, mas la ultima barra.
T_BARS = list(np.linspace(20_000, len(DF) - 2, 8).astype(int)) + [len(DF) - 1]
ATOL = 1e-9

# θ por regimen fijos a mano (no optimizados) para probar la estrategia combinada.
HAND_REGIME_THETAS = {
    "crisis": {**THETA0, "timeframe": "1h", "ema_fast": 9, "ema_slow": 21,
               "sl_mult": 1.5, "rr": 2.0, "max_holding": 288},
    "trend": THETA0,
    "mean_reversion": None,
}


def test_theta_signal_matches_strategy():
    """theta_signal de θ0 es la señal y el ATR de compute_ema_adx_features con esos parametros."""
    signal, atr = theta_signal(DF_TRAIN, THETA0)
    features = compute_ema_adx_features(DF_TRAIN, ema_fast=12, ema_slow=18, adx_threshold=25.0,
                                        timeframe="4h", entry_on_change=True)
    pd.testing.assert_series_equal(signal, features["signal"])
    pd.testing.assert_series_equal(atr, features["atr"])


def test_params_to_theta_hand_cases():
    """slow_ratio -> ema_slow = round(rapida · ratio), siempre > rapida; un θ completo no cambia."""
    assert params_to_theta({"ema_fast": 12, "slow_ratio": 1.5})["ema_slow"] == 18
    assert params_to_theta({"ema_fast": 5, "slow_ratio": 1.5})["ema_slow"] == 8
    assert "slow_ratio" not in params_to_theta({"ema_fast": 12, "slow_ratio": 2.0})
    assert params_to_theta(THETA0) == THETA0


def test_theta0_inside_param_space():
    """Cada parametro de θ0 cae dentro de su rango de busqueda."""
    for name, spec in PARAM_SPACE.items():
        if name == "slow_ratio":
            assert spec[1] <= THETA0["ema_slow"] / THETA0["ema_fast"] <= spec[2]
        elif spec[0] == "cat":
            assert THETA0[name] in spec[1]
        else:
            assert spec[1] <= THETA0[name] <= spec[2]


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
    assert crisis.any()
    assert (inputs.loc[crisis, "sl_mult"] == 1.5).all()
    assert (inputs.loc[crisis, "tp_mult"] == 3.0).all()
    assert (inputs.loc[crisis, "max_holding"] == 288).all()


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
