"""Pruebas de la evaluacion final (docs/SPEC.md, seccion 13) con datos SINTETICOS.

No se usa btc_project_test.csv: la evaluacion real se corre una sola vez en main.py.
Los datos imitan la forma del archivo real: train hasta 2023-12-31 00:00 y un
test con un dia suelto (2023-12-31), un hueco y despues unas semanas de mayo 2024.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import market_impact_table, returns
from src.optimize import REGIME_NAMES, final_test_inputs, paste_train_test
from src.signals import THETA0


def _random_walk(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Velas OHLCV de 5 min con una caminata aleatoria."""
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.003, len(index))))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = rng.uniform(0, 0.002, len(index))
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * (1 + spread),
        "Low": np.minimum(open_, close) * (1 - spread),
        "Close": close,
        "Volume": rng.uniform(1e6, 5e6, len(index)),
    }, index=index)


TRAIN = _random_walk(pd.date_range("2023-08-01", "2023-12-31 00:00", freq="5min"), seed=42)
TEST = pd.concat([
    _random_walk(pd.date_range("2023-12-31 00:00", "2023-12-31 23:55", freq="5min"), seed=7),
    _random_walk(pd.date_range("2024-05-02", "2024-05-20 23:55", freq="5min"), seed=8),
])
START, END = "2024-05-02", "2024-05-20"

# θ distinto por regimen para saber de donde sale cada parametro.
FINAL = {
    "global": {**THETA0, "sl_mult": 2.0, "rr": 2.0, "max_holding": 500, "rho": 0.010},
    "crisis": {**THETA0, "sl_mult": 1.0, "rr": 3.0, "max_holding": 600, "rho": 0.005},
    "trend": {**THETA0, "sl_mult": 3.0, "rr": 4.0, "max_holding": 700, "rho": 0.015},
    "mean_reversion": {**THETA0, "sl_mult": 4.0, "rr": 5.0, "max_holding": 800, "rho": 0.020},
}
REGIME_INPUTS = final_test_inputs(TRAIN, TEST, FINAL, True, START, END)
GLOBAL_INPUTS = final_test_inputs(TRAIN, TEST, FINAL, False, START, END)


def test_paste_keeps_train_bar_on_shared_timestamp():
    """La barra 2023-12-31 00:00 esta en los dos archivos: queda una sola, la de train."""
    pasted = paste_train_test(TRAIN, TEST)
    shared = pd.Timestamp("2023-12-31 00:00")
    assert pasted.index.is_unique and pasted.index.is_monotonic_increasing
    assert pasted.loc[shared, "Close"] == TRAIN.loc[shared, "Close"]


def test_no_bars_or_entries_before_test_start():
    """Las entradas solo cubren la ventana de test: nada antes de test_start ni despues de test_end."""
    for inputs in (REGIME_INPUTS, GLOBAL_INPUTS):
        assert inputs.index[0] >= pd.Timestamp(START)
        assert inputs.index[-1] < pd.Timestamp(END) + pd.Timedelta(days=1)
    assert (REGIME_INPUTS["signal"] != 0).any()   # la prueba no es trivial: si hay señales


def test_bar_params_come_from_regime_theta():
    """Cada barra lleva sl, tp = rr·sl, holding y rho del θ_final de su regimen."""
    seen = set()
    for regime in REGIME_NAMES:
        rows = REGIME_INPUTS[REGIME_INPUTS["regime"] == regime]
        if rows.empty:
            continue
        seen.add(regime)
        theta = FINAL[regime]
        assert (rows["sl_mult"] == theta["sl_mult"]).all()
        assert (rows["tp_mult"] == theta["rr"] * theta["sl_mult"]).all()
        assert (rows["max_holding"] == theta["max_holding"]).all()
        assert (rows["rho"] == theta["rho"]).all()
    assert len(seen) >= 2


def test_global_version_uses_global_theta_and_no_force_exit():
    """Solo global: θ_final global en todas las barras y nunca salida forzada."""
    theta = FINAL["global"]
    assert (GLOBAL_INPUTS["sl_mult"] == theta["sl_mult"]).all()
    assert (GLOBAL_INPUTS["rho"] == theta["rho"]).all()
    assert not GLOBAL_INPUTS["force_exit"].any()


def test_missing_regime_theta_falls_back_to_global():
    """R5: si un regimen no tiene θ_final, sus barras usan el global."""
    inputs = final_test_inputs(TRAIN, TEST, {**FINAL, "trend": None}, True, START, END)
    rows = inputs[inputs["regime"] == "trend"]
    assert not rows.empty
    assert (rows["sl_mult"] == FINAL["global"]["sl_mult"]).all()


def test_market_impact_by_hand():
    """Volumen de 1,000,000 USD por barra y nocional de 10,000: participacion 1% en 5 min.

    Una vela de 4h junta 48 barras: 48,000,000 -> participacion 10,000 / 48M.
    Impacto = σ_5min · sqrt(0.01) en pb; break-even de 5.6 pb.
    """
    index = pd.date_range("2023-01-01", periods=48 * 10, freq="5min")
    close = pd.Series(np.where(np.arange(len(index)) % 2 == 0, 100.0, 101.0), index=index)
    df = pd.DataFrame({"Close": close, "Volume": 1_000_000.0})
    trades = pd.DataFrame({"shares": [100.0, 50.0], "raw_entry_price": [100.0, 200.0]})   # 10,000 cada uno

    row = market_impact_table(df, {"curva": trades}, break_even_bps=5.6).loc["curva"]

    sigma = returns(close).std()
    assert row["avg_notional_usd"] == 10_000
    assert row["participation_5min_pct"] == pytest.approx(1.0)
    assert row["participation_4h_pct"] == pytest.approx(100 * 10_000 / 48_000_000)
    assert row["impact_bps"] == pytest.approx(sigma * math.sqrt(0.01) * 10_000)
    assert row["impact_vs_break_even_pct"] == pytest.approx(100 * row["impact_bps"] / 5.6)
    assert row["bars_without_volume_pct"] == 0.0
