"""Pruebas de src/optimize.py: ventanas, look-ahead, regla R5 y armado de la curva OOS."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_train
from src.optimize import (REGIME_NAMES, STUDIES, crisis_entries, empty_inputs, fill_inputs,
                          make_windows, optimize_window, select_params)

DF = load_train()
WINDOWS = make_windows(DF.index)


# --- Ventanas ---------------------------------------------------------------

def test_windows_lengths_and_dates():
    """Train de 1 mes, test de 7 dias que empieza donde termina el train, avance de 7 dias."""
    first = WINDOWS[0]
    assert first["train_start"] == pd.Timestamp("2022-07-01")
    for w in WINDOWS:
        assert w["train_start"] == w["train_end"] - pd.DateOffset(months=1)   # el mes antes del test
        assert pd.Timedelta(days=28) <= w["train_end"] - w["train_start"] <= pd.Timedelta(days=31)
        assert w["test_start"] == w["train_end"]
        assert w["test_end"] - w["test_start"] == pd.Timedelta(days=7)
    steps = {b["test_start"] - a["test_start"] for a, b in zip(WINDOWS, WINDOWS[1:])}
    assert steps == {pd.Timedelta(days=7)}


def test_test_weeks_do_not_overlap_and_fit_in_file():
    """Las semanas de test van una tras otra sin traslape y todas caben en el archivo."""
    for a, b in zip(WINDOWS, WINDOWS[1:]):
        assert a["test_end"] == b["test_start"]
    assert WINDOWS[0]["train_start"] >= DF.index[0]
    assert WINDOWS[-1]["test_end"] <= DF.index[-1] + pd.Timedelta(minutes=5)
    # La siguiente semana ya no cabria.
    assert WINDOWS[-1]["test_end"] + pd.Timedelta(days=7) > DF.index[-1] + pd.Timedelta(minutes=5)


# --- Sin look-ahead -----------------------------------------------------------

def test_window_params_ignore_data_after_train_end():
    """Cambiar los precios despues de train_end no cambia los parametros elegidos en esa ventana."""
    window = WINDOWS[0]
    original = optimize_window(DF, window, number=0, n_trials=4)

    corrupted = DF.copy()
    after = corrupted.index >= window["train_end"]
    for col in ["Open", "High", "Low", "Close"]:
        corrupted.loc[after, col] *= 1.5
    changed = optimize_window(corrupted, window, number=0, n_trials=4)

    assert original["best"] == changed["best"]
    assert any(original["best"][name] is not None for name in STUDIES)


# --- Regla R5 -----------------------------------------------------------------

def _best(tag: str) -> dict:
    """Mejor trial sintetico con un parametro que identifica su estudio."""
    return {"params": {"tag": tag}, "calmar": 1.0, "n_trades": 5}


def test_r5_regime_without_valid_trial_uses_global():
    """Un regimen sin trial valido usa los parametros globales de la ventana."""
    best = {"global": _best("g"), "crisis": None, "trend": _best("t"), "mean_reversion": None}
    params, used_r5 = select_params(best)
    assert params["crisis"] == {"tag": "g"}
    assert params["trend"] == {"tag": "t"}
    assert params["mean_reversion"] == {"tag": "g"}
    assert used_r5 == {"crisis": True, "trend": False, "mean_reversion": True}


def test_r5_without_valid_global_the_week_is_not_operated():
    """Si el global tampoco tiene trial valido, ningun estudio tiene parametros."""
    best = {"global": None, "crisis": _best("c"), "trend": None, "mean_reversion": None}
    params, _ = select_params(best)
    assert all(params[name] is None for name in STUDIES)


# --- Armado OOS -----------------------------------------------------------------

def _theta(sl_mult: float, rr: float, max_holding: int, rho: float) -> dict:
    return {"sl_mult": sl_mult, "rr": rr, "max_holding": max_holding, "rho": rho}


def test_oos_inputs_use_params_of_each_bar_regime():
    """Cada barra toma sl, tp = rr·sl, holding y rho de su regimen; la señal solo pasa en su regimen."""
    index = pd.date_range("2023-01-02", periods=6, freq="5min")
    regimes = pd.Series(["trend", "trend", "crisis", "mean_reversion", np.nan, "crisis"], index=index)
    features = pd.DataFrame({"signal": 1, "atr": 2.0}, index=index)
    params = {"trend": _theta(2.0, 3.0, 300, 0.01), "crisis": _theta(1.0, 2.0, 400, 0.005),
              "mean_reversion": _theta(3.0, 1.5, 500, 0.02)}

    inputs = empty_inputs(regimes)
    for name in REGIME_NAMES:
        bars = index[(regimes == name).to_numpy()]
        inputs = fill_inputs(inputs, bars, features, params[name])

    for name, theta in params.items():
        rows = inputs[inputs["regime"] == name]
        assert (rows["sl_mult"] == theta["sl_mult"]).all()
        assert (rows["tp_mult"] == theta["rr"] * theta["sl_mult"]).all()
        assert (rows["max_holding"] == theta["max_holding"]).all()
        assert (rows["rho"] == theta["rho"]).all()
        assert (rows["signal"] == 1).all()
    # Barra sin regimen (calentamiento): no se opera.
    assert inputs["signal"].iloc[4] == 0 and np.isnan(inputs["atr"].iloc[4])


def test_force_exit_only_when_regime_changes_to_crisis():
    """force_exit es True solo en la barra donde el regimen pasa a crisis (tambien despues de NaN)."""
    labels = pd.Series(["trend", "crisis", "crisis", "mean_reversion", "crisis", np.nan, "crisis"])
    assert crisis_entries(labels).tolist() == [False, True, False, False, True, False, True]
