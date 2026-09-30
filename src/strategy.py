"""Estrategia de lab_02: entry rule, exit rule, sizing y maquina de estados segun SPEC.md.

Los indicadores viven en src/indicators.py (calculados a mano con
pandas/numpy, sin librerias externas).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.indicators import atr, mfi, roc, rsi, sma

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
TOTAL_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE


@dataclass
class Position:
    """Posicion abierta por la estrategia.

    Atributos
    ---------
    side : str
        "long" o "short".
    shares : int
        Numero de acciones (ver compute_sizing).
    entry_price : float
        Precio al que se abrio la posicion.
    stop_loss : float
        Precio de stop-loss.
    take_profit : float
        Precio de take-profit.
    entry_bar : int
        Indice (posicional) de la barra en la que se abrio la posicion.
    """

    side: str
    shares: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_bar: int


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todo el pipeline de la entry rule (SPEC.md, seccion 3).

    indicadores -> normalizacion -> pesos adaptativos por ATR -> score Z_t
    -> filtro de consenso -> filtro de SMA(50) -> señal. Se regresan todas
    las columnas intermedias para poder auditarlas (prueba de truncamiento).

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".

    Regresa
    -------
    pd.DataFrame
        Columnas: sma_50, rsi_14, mfi_14, roc_10, atr_14, rsi_norm, mfi_norm,
        roc_norm, atr_mean, w_rsi, w_mfi, w_roc, z_score, consensus, signal.
        signal es 1 (long), -1 (short) o 0 (flat).
    """
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    sma_50 = sma(close, 50)
    rsi_14 = rsi(close, 14)
    mfi_14 = mfi(high, low, close, volume, 14)
    roc_10 = roc(close, 10)
    atr_14 = atr(high, low, close, 14)

    rsi_norm = (rsi_14 - 50) / 50
    mfi_norm = (mfi_14 - 50) / 50
    roc_norm = (roc_10 / 10).clip(-1, 1)

    # "Media de ATR" se toma como la media historica hasta la barra actual
    # (expanding), para no usar informacion futura al fijar el regimen.
    atr_mean = atr_14.expanding(min_periods=1).mean()
    high_vol = atr_14 > atr_mean

    w_rsi = pd.Series(np.where(high_vol, 0.25, 0.45), index=df.index)
    w_mfi = pd.Series(np.where(high_vol, 0.50, 0.25), index=df.index)
    w_roc = pd.Series(np.where(high_vol, 0.25, 0.30), index=df.index)

    z_score = w_rsi * rsi_norm + w_mfi * mfi_norm + w_roc * roc_norm

    sign_rsi = np.sign(rsi_norm)
    sign_mfi = np.sign(mfi_norm)
    sign_roc = np.sign(roc_norm)
    consensus = (sign_rsi == sign_mfi) & (sign_mfi == sign_roc) & (sign_rsi != 0)

    # La direccion la da el score y SMA(50) solo la confirma: si no
    # coinciden, la señal queda flat.
    signal = pd.Series(0, index=df.index, dtype=int)
    signal.loc[(z_score > 0.3) & consensus & (close > sma_50)] = 1
    signal.loc[(z_score < -0.3) & consensus & (close < sma_50)] = -1

    return pd.DataFrame({
        "sma_50": sma_50,
        "rsi_14": rsi_14,
        "mfi_14": mfi_14,
        "roc_10": roc_10,
        "atr_14": atr_14,
        "rsi_norm": rsi_norm,
        "mfi_norm": mfi_norm,
        "roc_norm": roc_norm,
        "atr_mean": atr_mean,
        "w_rsi": w_rsi,
        "w_mfi": w_mfi,
        "w_roc": w_roc,
        "z_score": z_score,
        "consensus": consensus,
        "signal": signal,
    })


def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
    """Señal de entrada por barra (SPEC.md, seccion 3).

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".

    Regresa
    -------
    pd.Series
        Señal por barra: 1 (long), -1 (short) o 0 (flat).
    """
    return compute_features(df)["signal"]


def compute_sizing(capital: float, atr_value: float, rho: float = 0.01) -> int:
    """Sizing por risk-parity segun ATR (SPEC.md, seccion 5).

    Q = (rho * capital) / (2 * ATR). El resultado se trunca a un entero de
    acciones (no se pueden comprar fracciones de accion).

    Parametros
    ----------
    capital : float
        Capital disponible (V_t).
    atr_value : float
        ATR(14) vigente en la barra de entrada.
    rho : float
        Presupuesto de riesgo por trade, como fraccion del capital.

    Regresa
    -------
    int
        Numero de acciones a comprar/vender en corto.
    """
    risk_budget = rho * capital
    stop_distance = 2 * atr_value
    return int(risk_budget / stop_distance)


def resolve_exit(position: Position, bar_high: float, bar_low: float, bar_close: float,
                  bar_index: int, max_holding: int = 10) -> tuple[bool, Optional[str], Optional[float]]:
    """Revisa si una posicion debe cerrarse en la barra actual (SPEC.md, seccion 4).

    Si el High y el Low de la barra tocan tanto el take-profit como el
    stop-loss (tie intrabar), se resuelve a favor del stop-loss (conservador).

    Parametros
    ----------
    position : Position
        Posicion abierta a evaluar.
    bar_high, bar_low, bar_close : float
        OHLC de la barra actual (solo se usan High, Low y Close).
    bar_index : int
        Indice posicional de la barra actual.
    max_holding : int
        Numero maximo de barras que se puede mantener la posicion. La barra
        de entrada cuenta como barra 1 (ver SPEC.md, seccion 4).

    Regresa
    -------
    tuple[bool, str | None, float | None]
        (se_cierra, motivo, precio_de_salida). motivo es uno de
        "stop_loss", "take_profit", "max_holding", o None si no se cierra.
    """
    if position.side == "long":
        hit_sl = bar_low <= position.stop_loss
        hit_tp = bar_high >= position.take_profit
    else:
        hit_sl = bar_high >= position.stop_loss
        hit_tp = bar_low <= position.take_profit

    if hit_sl:
        return True, "stop_loss", position.stop_loss
    if hit_tp:
        return True, "take_profit", position.take_profit
    if bar_index - position.entry_bar >= max_holding - 1:
        return True, "max_holding", bar_close

    return False, None, None


def resolve_open_gap(position: Position, bar_open: float) -> tuple[bool, Optional[str], Optional[float]]:
    """Revisa si el open de la barra ya cruzo el SL o el TP (gap).

    Si el precio abre mas alla del stop (o del target), la orden no puede
    llenarse al nivel del SL/TP: se llena al open. Es la ejecucion realista
    y evita el sesgo optimista de llenar al SL en un gap (SPEC.md, seccion 7).

    Parametros
    ----------
    position : Position
        Posicion abierta a evaluar.
    bar_open : float
        Precio de apertura de la barra actual.

    Regresa
    -------
    tuple[bool, str | None, float | None]
        (se_cierra, motivo, precio_de_salida). motivo es "stop_loss" o
        "take_profit"; el precio de salida es el open.
    """
    if position.side == "long":
        hit_sl = bar_open <= position.stop_loss
        hit_tp = bar_open >= position.take_profit
    else:
        hit_sl = bar_open >= position.stop_loss
        hit_tp = bar_open <= position.take_profit

    if hit_sl:
        return True, "stop_loss", bar_open
    if hit_tp:
        return True, "take_profit", bar_open
    return False, None, None


def _adjust_entry_price(raw_price: float, side: str) -> float:
    """Ajusta el precio de entrada por comision y slippage (SPEC.md, seccion 6).

    Un long paga de mas al entrar (precio efectivo mas alto); un short recibe
    de menos (precio efectivo mas bajo).
    """
    if side == "long":
        return raw_price * (1 + TOTAL_COST_RATE)
    return raw_price * (1 - TOTAL_COST_RATE)


def _adjust_exit_price(raw_price: float, side: str) -> float:
    """Ajusta el precio de salida por comision y slippage (SPEC.md, seccion 6).

    Un long recibe de menos al salir; un short paga de mas al cubrir.
    """
    if side == "long":
        return raw_price * (1 - TOTAL_COST_RATE)
    return raw_price * (1 + TOTAL_COST_RATE)


def run_backtest(df: pd.DataFrame, capital: float = 100_000.0, rho: float = 0.01,
                  sl_mult: float = 2.0, tp_mult: float = 3.0, max_holding: int = 10) -> list[dict]:
    """Maquina de estados de la estrategia completa (entry + exit + sizing).

    Mantiene como maximo una posicion abierta a la vez. Cada barra t se
    procesa en orden cronologico (SPEC.md, seccion 7):

    1. Open: si el open ya cruzo el SL/TP (gap), se cierra al open.
    2. Open: si la señal de t-1 es opuesta a la posicion, se cierra al open.
    3. Open: si no hay posicion y la señal de t-1 es != 0, se abre al open.
    4. Intrabar: SL/TP con el High/Low de t (tie -> SL).
    5. Close: salida por holding maximo al cierre de t.

    Como las entradas solo ocurren en el open, una posicion cerrada en los
    pasos 4 o 5 no puede reemplazarse hasta el open de t+1.

    El precio de entrada/salida efectivo incluye comision (0.1%) y slippage
    (0.05%); el SL y el TP se calculan sobre el precio crudo (sin costos).
    El capital se actualiza con el P&L realizado de cada trade y el sizing
    del siguiente trade usa ese capital actualizado.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Open", "High", "Low", "Close", "Volume".
    capital, rho, sl_mult, tp_mult, max_holding
        Ver SPEC.md, secciones 4 y 5.

    Regresa
    -------
    list[dict]
        Un registro por operacion cerrada, con entry_bar, exit_bar, side,
        entry_price, exit_price, shares, exit_reason, exit_phase ("open",
        "intrabar" o "close") y pnl. entry_price y exit_price ya incluyen el
        ajuste por costos. Toda entrada ocurre en el open de entry_bar.
    """
    features = compute_features(df)
    sig = features["signal"].to_numpy()
    atr_vals = features["atr_14"].to_numpy()

    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()

    trades: list[dict] = []
    position: Optional[Position] = None
    equity = capital
    n = len(df)

    def _close(position: Position, exit_bar: int, raw_exit_price: float, reason: str, phase: str) -> float:
        exit_price = _adjust_exit_price(raw_exit_price, position.side)
        if position.side == "long":
            pnl = position.shares * (exit_price - position.entry_price)
        else:
            pnl = position.shares * (position.entry_price - exit_price)

        trades.append({
            "entry_bar": position.entry_bar,
            "exit_bar": exit_bar,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "exit_reason": reason,
            "exit_phase": phase,
            "pnl": pnl,
        })
        return pnl

    for t in range(1, n):
        # 1. Gap en el open que ya cruzo el SL o el TP.
        if position is not None:
            closed, reason, raw_exit_price = resolve_open_gap(position, opens[t])
            if closed:
                equity += _close(position, t, raw_exit_price, reason, "open")
                position = None

        desired_side = int(sig[t - 1])

        # 2. Señal opuesta: se cierra al open.
        if position is not None and desired_side != 0:
            current_side = 1 if position.side == "long" else -1
            if desired_side != current_side:
                equity += _close(position, t, opens[t], "opposite_signal", "open")
                position = None

        # 3. Entrada al open con la señal y el ATR de la barra anterior.
        if position is None and desired_side != 0 and not np.isnan(atr_vals[t - 1]):
            raw_entry_price = opens[t]
            atr_at_entry = atr_vals[t - 1]
            shares = compute_sizing(equity, atr_at_entry, rho)

            if shares > 0:
                side = "long" if desired_side == 1 else "short"
                if side == "long":
                    stop_loss = raw_entry_price - sl_mult * atr_at_entry
                    take_profit = raw_entry_price + tp_mult * atr_at_entry
                else:
                    stop_loss = raw_entry_price + sl_mult * atr_at_entry
                    take_profit = raw_entry_price - tp_mult * atr_at_entry

                position = Position(
                    side=side,
                    shares=shares,
                    entry_price=_adjust_entry_price(raw_entry_price, side),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_bar=t,
                )

        # 4 y 5. SL/TP intrabar (incluida la barra de entrada) y holding maximo al cierre.
        if position is not None:
            closed, reason, raw_exit_price = resolve_exit(position, highs[t], lows[t], closes[t], t, max_holding)
            if closed:
                phase = "close" if reason == "max_holding" else "intrabar"
                equity += _close(position, t, raw_exit_price, reason, phase)
                position = None

    return trades


def compute_win_rate(trades: list[dict]) -> float:
    """Calcula el porcentaje de trades ganadores (SPEC.md, break-even win rate).

    Un trade ganador es aquel donde exit_price > entry_price (long) o
    entry_price > exit_price (short), usando los precios ya ajustados por
    costos que regresa run_backtest.

    Parametros
    ----------
    trades : list[dict]
        Lista de trades regresada por run_backtest.

    Regresa
    -------
    float
        Porcentaje de trades ganadores (0-100). 0.0 si no hay trades.
    """
    if not trades:
        return 0.0

    wins = 0
    for trade in trades:
        if trade["side"] == "long":
            if trade["exit_price"] > trade["entry_price"]:
                wins += 1
        else:
            if trade["entry_price"] > trade["exit_price"]:
                wins += 1

    return 100.0 * wins / len(trades)
