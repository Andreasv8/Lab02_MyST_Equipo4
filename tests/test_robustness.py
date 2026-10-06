"""Pruebas de la robustez pre-registrada (docs/SPEC.md, seccion 14) en src/optimize.py."""

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import COMMISSION_RATE
from src.data import load_train
from src.optimize import (INT_PARAMS, SEARCH_BOUNDS, SEARCH_PARAMS, build_oos_inputs, entry_regimes,
                          make_windows, oos_cost_curve, optimize_window, run_oos, shift_param,
                          suggest_params, transitions_table)
from src.signals import indicator_correlation, THETA0


# --- Sensibilidad ±20% (14.1) ------------------------------------------------

def test_shift_param_integer_moves_at_least_one_unit():
    """Enteros: se redondean, se mueven al menos 1 unidad y se recortan a los limites.

    ema_fast 5 · 0.8 = 4 -> se recorta a 5 (el limite) = "en el limite".
    roc_window 6 · 1.2 = 7.2 -> 7.  bb_window 12 · 0.8 = 9.6 -> 10.
    ema_fast 7 · 1.05 = 7.35 -> 7 (igual al base) -> se mueve 1 unidad a 8.
    """
    assert shift_param("ema_fast", 5, 0.8) == (5, True)
    assert shift_param("roc_window", 6, 1.2) == (7, False)
    assert shift_param("bb_window", 12, 0.8) == (10, False)
    assert shift_param("ema_fast", 7, 1.05) == (8, False)


def test_shift_param_float_and_clipping():
    """Floats: base · factor recortado a los limites; si queda igual al base, "en el limite"."""
    assert shift_param("sl_mult", 2.0, 1.2) == (pytest.approx(2.4), False)
    assert shift_param("bb_threshold", 0.80, 1.2) == (0.90, False)
    assert shift_param("bb_threshold", 0.90, 1.2) == (0.90, True)


def test_search_bounds_match_suggest_params():
    """Los limites de SEARCH_BOUNDS son los mismos que pide suggest_params."""
    trial = optuna.trial.FixedTrial({name: SEARCH_BOUNDS[name][0] for name in SEARCH_PARAMS})
    low = suggest_params(trial)
    for name in SEARCH_PARAMS:
        assert low[name] == SEARCH_BOUNDS[name][0]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    suggest_params(trial)
    for name in SEARCH_PARAMS:
        dist = trial.distributions[name]
        assert (dist.low, dist.high) == SEARCH_BOUNDS[name]
    assert set(INT_PARAMS) <= set(SEARCH_PARAMS)


# --- Curva de costos (14.2) y regimen de entrada ---------------------------------

DF = load_train()
WINDOWS = make_windows(DF.index)
RESULTS = [optimize_window(DF, WINDOWS[i], i, n_trials=4) for i in range(3)]
INPUTS = build_oos_inputs(DF, RESULTS)


def test_cost_curve_at_real_cost_matches_oos_curve():
    """En 12.5 pb la curva de costos da exactamente el retorno de la curva OOS original."""
    curve = oos_cost_curve(DF, INPUTS, bps_grid=[0.0, 12.5, 50.0])
    equity = run_oos(DF, INPUTS).equity["equity"]
    assert COMMISSION_RATE == 12.5 / 10_000
    assert curve.loc[12.5, "total_return"] == pytest.approx(equity.iloc[-1] / equity.iloc[0] - 1)
    assert curve.loc[0.0, "total_return"] >= curve.loc[12.5, "total_return"] >= curve.loc[50.0, "total_return"]


def test_entry_regimes_take_the_label_of_entry_bar():
    """El regimen de entrada es la etiqueta de la barra de entrada en las entradas OOS."""
    trades = run_oos(DF, INPUTS).trades
    regimes = entry_regimes(trades, INPUTS)
    for i, trade in trades.iterrows():
        assert regimes[i] == INPUTS["regime"].iloc[trade["entry_bar"]]


def test_transitions_table_by_hand():
    """Etiquetas horarias trend -> crisis -> trend en 30 dias (1 mes); un cierre regime_exit en trend."""
    index = pd.date_range("2023-01-01", periods=720, freq="h")
    regime = pd.Series(["trend"] * 240 + ["crisis"] * 240 + ["trend"] * 240, index=index)
    inputs = pd.DataFrame({"regime": regime})
    trades = pd.DataFrame({"entry_bar": [10, 300], "exit_reason": ["regime_exit", "stop_loss"]})
    empty = {name: None for name in ["global", "crisis", "trend", "mean_reversion"]}
    results = [{"number": 0, "train_start": index[0], "test_start": index[0],
                "best": {**empty, "global": {"params": {}, "calmar": 1.0, "n_trades": 5}}}]

    table = transitions_table(inputs, trades, results)

    assert table.loc["crisis", "transitions_in_per_month"] == pytest.approx(1.0)
    assert table.loc["trend", "transitions_in_per_month"] == pytest.approx(1.0)
    assert table.loc["total", "transitions_in_per_month"] == pytest.approx(2.0)
    assert table.loc["trend", "regime_exits"] == 1
    assert table.loc["crisis", "r5_windows"] == 1


# --- Correlacion (14.5) ---------------------------------------------------------

def test_indicator_correlation_is_symmetric_with_unit_diagonal():
    corr_votes, corr_indicators = indicator_correlation(DF.iloc[:60_000], THETA0)
    for corr in (corr_votes, corr_indicators):
        np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)
        np.testing.assert_allclose(np.diag(corr), 1.0)
    assert list(corr_indicators.columns) == ["ema_gap", "roc", "percent_b", "adx"]
