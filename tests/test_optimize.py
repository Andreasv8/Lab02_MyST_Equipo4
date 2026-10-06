"""Pruebas de src/optimize.py: ventanas, look-ahead, regla R5 y armado de la curva OOS."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_train
from src.backtest import BacktestResult
from src.optimize import (REGIME_NAMES, STUDIES, crisis_entries, degradation_table, empty_inputs,
                          fill_inputs, make_windows, optimize_window, select_params, theta_final)

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


# --- θ_final (SPEC seccion 13) ------------------------------------------------

def _search_theta(**changes) -> dict:
    """θ del espacio de busqueda con valores base y algunos cambios."""
    base = {"ema_fast": 10, "slow_ratio": 3.0, "roc_window": 12, "bb_window": 20, "bb_threshold": 0.7,
            "adx_threshold": 20.0, "sl_mult": 2.0, "rr": 3.0, "max_holding": 1000, "rho": 0.01}
    return {**base, **changes}


def _result(number: int, best: dict) -> dict:
    return {"number": number, "best": best}


def test_theta_final_median_rounding_and_ema_slow():
    """Mediana por parametro; enteros con .5 hacia arriba; ema_slow recalculada; sin trial valido no cuenta.

    global: ema_fast [10, 13] -> 11.5 -> 12; slow_ratio [3, 4] -> 3.5; ema_slow = round(12·3.5) = 42.
    rr [3, 5] -> 4. La tercera ventana no tiene trial global valido y no entra en la mediana.
    """
    g1 = {"params": _search_theta(ema_fast=10, slow_ratio=3.0, rr=3.0)}
    g2 = {"params": _search_theta(ema_fast=13, slow_ratio=4.0, rr=5.0)}
    empty = {name: None for name in STUDIES}
    results = [_result(0, {**empty, "global": g1}), _result(1, {**empty, "global": g2}),
               _result(2, empty)]

    final = theta_final(results)

    theta = final["global"]
    assert theta["ema_fast"] == 12
    assert theta["slow_ratio"] == pytest.approx(3.5)
    assert theta["ema_slow"] == 42
    assert theta["rr"] == pytest.approx(4.0)
    assert (theta["bb_std"], theta["adx_window"], theta["atr_window"]) == (2, 14, 14)
    assert all(final[name] is None for name in REGIME_NAMES)


def test_theta_final_ema_slow_at_least_fast_plus_one():
    """Si round(ema_fast · slow_ratio) <= ema_fast, se usa ema_fast + 1."""
    empty = {name: None for name in STUDIES}
    best = {"params": _search_theta(ema_fast=5, slow_ratio=1.0)}
    final = theta_final([_result(0, {**empty, "global": best})])
    assert final["global"]["ema_slow"] == 6


# --- Degradacion train -> OOS ---------------------------------------------------

def test_degradation_weekly_return_comparison_by_hand():
    """Una ventana de 4 semanas de train con +8% (2% por semana) y una semana OOS de +1%.

    Proporcion que sobrevive en retorno semanal = 1% / 2% = 0.5.
    """
    train_start, test_start = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-29")
    best = {"params": {}, "calmar": 4.0, "n_trades": 6, "total_return": 0.08}
    result = {"number": 0, "train_start": train_start, "train_end": test_start,
              "test_start": test_start, "test_end": test_start + pd.Timedelta(days=7),
              "best": {"global": best}}
    index = pd.date_range(test_start, periods=3, freq="3D")
    equity = pd.DataFrame({"equity": [100.0, 100.5, 101.0], "shares": 0.0}, index=index)
    oos = {"curva": BacktestResult(equity=equity, trades=pd.DataFrame(columns=["pnl"]))}

    table, summary = degradation_table([result], oos)

    assert table.loc[0, "weekly_return_train"] == pytest.approx(0.02)
    assert table.loc[0, "weekly_return_oos_curva"] == pytest.approx(0.01)
    weekly = summary[summary["comparison"] == "weekly_return"].iloc[0]
    assert weekly["share_survives"] == pytest.approx(0.5)
    assert set(summary["comparison"]) == {"calmar", "weekly_return"}
