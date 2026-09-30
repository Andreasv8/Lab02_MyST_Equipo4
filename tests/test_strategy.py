import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.strategy as strategy
from src.strategy import TOTAL_COST_RATE, Position, compute_features, compute_sizing, resolve_exit, run_backtest

# Orden cronologico de los eventos dentro de una barra (SPEC.md, seccion 7).
PHASE_ORDER = {"open": 0, "intrabar": 1, "close": 2}


def test_sl_tp_same_bar_resolves_as_stop():
    """Si SL y TP caen en la misma barra, debe ejecutarse el stop-loss (conservador)."""
    position = Position(
        side="long",
        shares=10,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        entry_bar=0,
    )

    closed, reason, exit_price = resolve_exit(
        position, bar_high=112.0, bar_low=90.0, bar_close=105.0, bar_index=1,
    )

    assert closed
    assert reason == "stop_loss"
    assert exit_price == position.stop_loss


def test_sizing_matches_risk_budget():
    """El numero de acciones debe corresponder exactamente al riesgo presupuestado (rho * capital)."""
    capital = 100_000.0
    atr_value = 10.0
    rho = 0.01

    shares = compute_sizing(capital, atr_value, rho)
    risk_in_dollars = shares * 2 * atr_value

    assert shares == 50
    assert risk_in_dollars == rho * capital


def test_no_simultaneous_positions():
    """La maquina de estados nunca debe mantener dos posiciones abiertas al mismo tiempo."""
    rng = np.random.default_rng(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close = pd.Series(100 + np.cumsum(rng.normal(0, 2.0, size=n)), index=dates)
    high = close + rng.uniform(0.5, 2.5, size=n)
    low = close - rng.uniform(0.5, 2.5, size=n)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, size=n), index=dates)

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    trades = run_backtest(df, capital=100_000.0)

    assert len(trades) > 0

    # Toda entrada ocurre en el open de entry_bar. La posicion anterior debe
    # haberse cerrado antes: en una barra previa, o en el open de la misma
    # barra (gap o señal opuesta). Una salida intrabar o al cierre de t seguida
    # de una entrada al open de t seria un traslape en el tiempo.
    for previous, current in zip(trades, trades[1:]):
        previous_exit = (previous["exit_bar"], PHASE_ORDER[previous["exit_phase"]])
        current_entry = (current["entry_bar"], PHASE_ORDER["open"])
        assert previous_exit <= current_entry


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Construye un DataFrame OHLCV a partir de tuplas (Open, High, Low, Close)."""
    dates = pd.date_range("2024-01-01", periods=len(rows), freq="B")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=dates)
    df["Volume"] = 1_000_000
    return df


def _patch_features(monkeypatch, signal: list[int], atr_value: float = 2.0) -> None:
    """Sustituye el pipeline de indicadores por una señal y un ATR fijos."""
    def fake_features(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signal": signal, "atr_14": atr_value}, index=df.index)

    monkeypatch.setattr(strategy, "compute_features", fake_features)


def test_no_reentry_in_same_bar_after_intrabar_exit(monkeypatch):
    """Si el TP se toca intrabar en t, la siguiente entrada debe ser al open de t+1, no de t."""
    df = _bars([
        (100, 101, 99, 100),     # 0: señal long
        (100, 101, 99, 100),     # 1: entra long al open (SL 96, TP 106)
        (100, 107, 99, 105),     # 2: toca TP intrabar; la señal sigue en long
        (105, 106, 104, 105),    # 3: reentra aqui (SL 101)
        (95, 96, 94, 95),        # 4: gap bajo el SL, cierra la segunda posicion
    ])
    _patch_features(monkeypatch, [1, 1, 1, 0, 0])

    trades = run_backtest(df, capital=10_000.0, max_holding=10)

    assert trades[0]["exit_bar"] == 2
    assert trades[0]["exit_reason"] == "take_profit"
    assert trades[0]["exit_phase"] == "intrabar"
    assert trades[1]["entry_bar"] == 3


def test_gap_through_stop_fills_at_open(monkeypatch):
    """Un open por debajo del SL de un long se llena al open, no al precio del SL."""
    df = _bars([
        (100, 101, 99, 100),     # 0: señal long
        (100, 101, 99, 100),     # 1: entra long al open (SL 96)
        (90, 91, 88, 89),        # 2: abre en 90, por debajo del SL
    ])
    _patch_features(monkeypatch, [1, 0, 0])

    trades = run_backtest(df, capital=10_000.0)

    assert trades[0]["exit_reason"] == "stop_loss"
    assert trades[0]["exit_phase"] == "open"
    assert trades[0]["exit_price"] == pytest.approx(90 * (1 - TOTAL_COST_RATE))


def test_opposite_signal_closes_at_open_before_intrabar_checks(monkeypatch):
    """La señal opuesta cierra al open aunque despues en la barra se toque el SL."""
    df = _bars([
        (100, 101, 99, 100),     # 0: señal long
        (100, 101, 99, 100),     # 1: entra long (SL 96); señal short al cierre
        (100, 101, 95, 97),      # 2: cierra long y abre short al open; el Low toca el SL del long
        (110, 111, 109, 110),    # 3: gap sobre el SL del short (104), lo cierra
    ])
    _patch_features(monkeypatch, [1, -1, 0, 0])

    trades = run_backtest(df, capital=10_000.0)

    assert trades[0]["side"] == "long"
    assert trades[0]["exit_reason"] == "opposite_signal"
    assert trades[0]["exit_price"] == pytest.approx(100 * (1 - TOTAL_COST_RATE))
    assert trades[1]["side"] == "short"
    assert trades[1]["entry_bar"] == 2


def test_max_holding_closes_on_tenth_bar(monkeypatch):
    """La barra de entrada cuenta como 1: sin SL/TP, cierra al cierre de entry_bar + 9."""
    rows = [(100, 101, 99, 100)] * 15
    _patch_features(monkeypatch, [1] + [0] * 14)

    trades = run_backtest(_bars(rows), capital=10_000.0)

    assert trades[0]["entry_bar"] == 1
    assert trades[0]["exit_bar"] == 10
    assert trades[0]["exit_reason"] == "max_holding"
    assert trades[0]["exit_phase"] == "close"


def test_signal_direction_agrees_with_score_and_sma():
    """Long exige Z > 0.3 y Close > SMA(50); short exige Z < -0.3 y Close < SMA(50)."""
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "NVDA_daily.csv",
                     index_col="Date", parse_dates=True)
    features = compute_features(df)

    longs = features[features["signal"] == 1]
    shorts = features[features["signal"] == -1]

    assert len(longs) > 0 and len(shorts) > 0
    assert (longs["z_score"] > 0.3).all()
    assert (df.loc[longs.index, "Close"] > longs["sma_50"]).all()
    assert (shorts["z_score"] < -0.3).all()
    assert (df.loc[shorts.index, "Close"] < shorts["sma_50"]).all()
