"""Estrategia de lab_02: entry rule, exit rule y sizing segun SPEC.md.

Reutiliza los indicadores de data/indicator_analysis.py (calculados a mano
con pandas/numpy, sin librerias externas).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.indicator_analysis import atr, mfi, roc, rsi, sma  # noqa: E402

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


def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
    """Calcula la señal de entrada (SPEC.md, seccion 3).

    Normaliza RSI(14), MFI(14) y ROC(10) a [-1, 1], pondera con pesos
    adaptativos segun el regimen de volatilidad de ATR(14) (comparado contra
    su media historica hasta la barra actual, sin ver el futuro), aplica un
    filtro de consenso de signos y un filtro direccional de SMA(50).

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".

    Regresa
    -------
    pd.Series
        Señal por barra: 1 (long), -1 (short) o 0 (flat).
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

    w_rsi = np.where(high_vol, 0.25, 0.45)
    w_mfi = np.where(high_vol, 0.50, 0.25)
    w_roc = np.where(high_vol, 0.25, 0.30)

    z_score = w_rsi * rsi_norm + w_mfi * mfi_norm + w_roc * roc_norm

    sign_rsi = np.sign(rsi_norm)
    sign_mfi = np.sign(mfi_norm)
    sign_roc = np.sign(roc_norm)
    consensus = (sign_rsi == sign_mfi) & (sign_mfi == sign_roc) & (sign_rsi != 0)

    entry_condition = (z_score.abs() > 0.3) & consensus

    signal = pd.Series(0, index=df.index, dtype=int)
    signal.loc[entry_condition & (close > sma_50)] = 1
    signal.loc[entry_condition & (close < sma_50)] = -1

    return signal


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

    Mantiene como maximo una posicion abierta a la vez. La señal calculada al
    cierre de la barra t se ejecuta al open de la barra t+1; una señal
    opuesta a la posicion abierta la cierra e inmediatamente abre la nueva.

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
        entry_price, exit_price, shares, exit_reason y pnl. entry_price y
        exit_price ya incluyen el ajuste por costos.
    """
    signal = compute_entry_signal(df)
    atr_14 = atr(df["High"], df["Low"], df["Close"], 14)

    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    sig = signal.to_numpy()
    atr_vals = atr_14.to_numpy()

    trades: list[dict] = []
    position: Optional[Position] = None
    equity = capital
    n = len(df)

    def _close(position: Position, exit_bar: int, raw_exit_price: float, reason: str) -> float:
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
            "pnl": pnl,
        })
        return pnl

    for t in range(1, n):
        if position is not None:
            closed, reason, raw_exit_price = resolve_exit(position, highs[t], lows[t], closes[t], t, max_holding)
            if closed:
                equity += _close(position, t, raw_exit_price, reason)
                position = None

        desired_side = int(sig[t - 1])

        if position is not None and desired_side != 0:
            current_side = 1 if position.side == "long" else -1
            if desired_side != current_side:
                equity += _close(position, t, opens[t], "opposite_signal")
                position = None

        if position is None and desired_side != 0 and not np.isnan(atr_vals[t - 1]):
            raw_entry_price = opens[t]
            atr_at_entry = atr_vals[t - 1]
            shares = compute_sizing(equity, atr_at_entry, rho)

            if shares > 0:
                side = "long" if desired_side == 1 else "short"
                entry_price = _adjust_entry_price(raw_entry_price, side)

                if side == "long":
                    stop_loss = raw_entry_price - sl_mult * atr_at_entry
                    take_profit = raw_entry_price + tp_mult * atr_at_entry
                else:
                    stop_loss = raw_entry_price + sl_mult * atr_at_entry
                    take_profit = raw_entry_price - tp_mult * atr_at_entry

                position = Position(
                    side=side,
                    shares=shares,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_bar=t,
                )

                # El SL/TP puede tocarse en la misma barra en la que se abre
                # la posicion; se revisa de inmediato con el H/L de esa barra.
                closed, reason, raw_exit_price = resolve_exit(position, highs[t], lows[t], closes[t], t, max_holding)
                if closed:
                    equity += _close(position, t, raw_exit_price, reason)
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
