"""Motor de backtest orientado a eventos (barra por barra) de lab_02.

backtest() es una funcion pura: recibe precios, señal, ATR y configuracion,
y regresa la curva de capital por barra y la lista de trades. Reutiliza la
logica de salida y sizing de src/strategy.py (SPEC.md, secciones 4, 5 y 7).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy import (
    BORROW_FEE_ANNUAL,
    TOTAL_COST_RATE,
    Position,
    _adjust_entry_price,
    _adjust_exit_price,
    compute_borrow_fee,
    compute_sizing,
    resolve_exit,
    resolve_open_gap,
)

TRADE_COLUMNS = [
    "entry_bar", "exit_bar", "entry_date", "exit_date", "side", "entry_price",
    "exit_price", "shares", "truncated", "exit_reason", "exit_phase", "borrow_fee", "pnl",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Parametros del backtest (SPEC.md, secciones 4, 5 y 6).

    Atributos
    ---------
    initial_cash : float
        Capital inicial en efectivo.
    rho : float
        Presupuesto de riesgo por trade, como fraccion del capital.
    sl_mult, tp_mult : float
        Multiplos del ATR para el stop-loss y el take-profit.
    max_holding : int
        Barras maximas por posicion (la barra de entrada cuenta como 1).
    cost_rate : float
        Costo por lado (comision + slippage) como fraccion del nocional.
    borrow_fee_annual : float
        Tasa anual de borrow de los shorts sobre el nocional.
    """

    initial_cash: float = 100_000.0
    rho: float = 0.01
    sl_mult: float = 2.0
    tp_mult: float = 3.0
    max_holding: int = 10
    cost_rate: float = TOTAL_COST_RATE
    borrow_fee_annual: float = BORROW_FEE_ANNUAL


@dataclass(frozen=True)
class BacktestResult:
    """Resultado de backtest().

    Atributos
    ---------
    equity : pd.DataFrame
        Una fila por barra (mismo indice que df) con cash, shares (con signo:
        + long, - short) y equity = cash + shares * Close, al cierre de la barra.
    trades : pd.DataFrame
        Una fila por operacion cerrada, columnas TRADE_COLUMNS.
    """

    equity: pd.DataFrame
    trades: pd.DataFrame


def backtest(df: pd.DataFrame, signal: pd.Series, atr: pd.Series,
             config: BacktestConfig) -> BacktestResult:
    """Simula la estrategia barra por barra con estado explicito de caja.

    Cada barra t se procesa en el orden de SPEC.md, seccion 7:

    1. Open: si el open ya cruzo el SL/TP (gap), se cierra al open.
    2. Open: si la señal de t-1 es opuesta a la posicion, se cierra al open.
    3. Open: si no hay posicion y la señal de t-1 es != 0, se abre al open
       con el ATR de t-1 (sizing con V_t = cash, que estando flat es el
       capital realizado).
    4. Intrabar: SL/TP con el High/Low de t (tie -> SL).
    5. Close: salida por holding maximo al cierre de t.

    Contabilidad de caja, con P crudo, q acciones y c = cost_rate:
    entrada long cash -= P*q*(1+c); salida long cash += P*q*(1-c);
    entrada short cash += P*q*(1-c); salida short cash -= P*q*(1+c) + borrow_fee.
    Una posicion abierta al final no se liquida: queda valuada a mercado en
    equity y no aparece en trades. No modifica sus entradas.

    Parametros
    ----------
    df : pd.DataFrame
        Columnas "Open", "High", "Low", "Close" e indice de fechas.
    signal : pd.Series
        Señal al cierre de cada barra: 1, -1 o 0. Mismo indice que df.
    atr : pd.Series
        ATR al cierre de cada barra (NaN en warm-up). Mismo indice que df.
    config : BacktestConfig
        Parametros de sizing, salida y costos.

    Regresa
    -------
    BacktestResult
        equity (cash, shares, equity por barra) y trades.
    """
    if not signal.index.equals(df.index) or not atr.index.equals(df.index):
        raise ValueError("signal y atr deben tener el mismo indice que df")

    sig = signal.to_numpy()
    atr_vals = atr.to_numpy(dtype=float)
    dates = df.index
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    n = len(df)
    c = config.cost_rate

    cash = config.initial_cash
    shares = 0
    position: Optional[Position] = None
    trades: list[dict] = []
    cash_hist = np.empty(n)
    shares_hist = np.zeros(n, dtype=int)

    def _open(t: int, side: str, q: int, atr_at_entry: float, truncated: bool) -> tuple[Position, float, int]:
        """Abre una posicion al open de t; regresa (posicion, delta de caja, shares con signo)."""
        raw = opens[t]
        if side == "long":
            stop_loss = raw - config.sl_mult * atr_at_entry
            take_profit = raw + config.tp_mult * atr_at_entry
            cash_delta, signed = -raw * q * (1 + c), q
        else:
            stop_loss = raw + config.sl_mult * atr_at_entry
            take_profit = raw - config.tp_mult * atr_at_entry
            cash_delta, signed = raw * q * (1 - c), -q

        new_position = Position(
            side=side,
            shares=q,
            entry_price=_adjust_entry_price(raw, side, c),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar=t,
            entry_date=dates[t],
            raw_entry_price=raw,
            truncated=truncated,
        )
        return new_position, cash_delta, signed

    def _close(position: Position, t: int, raw_exit: float, reason: str, phase: str) -> float:
        """Cierra la posicion en t, registra el trade y regresa el delta de caja."""
        q = position.shares
        exit_price = _adjust_exit_price(raw_exit, position.side, c)
        borrow_fee = compute_borrow_fee(position.side, q, position.raw_entry_price,
                                        position.entry_date, dates[t], config.borrow_fee_annual)
        if position.side == "long":
            cash_delta = raw_exit * q * (1 - c)
            pnl = q * (exit_price - position.entry_price)
        else:
            cash_delta = -(raw_exit * q * (1 + c) + borrow_fee)
            pnl = q * (position.entry_price - exit_price) - borrow_fee

        trades.append({
            "entry_bar": position.entry_bar,
            "exit_bar": t,
            "entry_date": position.entry_date,
            "exit_date": dates[t],
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": q,
            "truncated": position.truncated,
            "exit_reason": reason,
            "exit_phase": phase,
            "borrow_fee": borrow_fee,
            "pnl": pnl,
        })
        return cash_delta

    cash_hist[0] = cash

    for t in range(1, n):
        # 1. Gap en el open que ya cruzo el SL o el TP.
        if position is not None:
            closed, reason, raw_exit = resolve_open_gap(position, opens[t])
            if closed:
                cash += _close(position, t, raw_exit, reason, "open")
                position, shares = None, 0

        desired_side = int(sig[t - 1])

        # 2. Señal opuesta: se cierra al open.
        if position is not None and desired_side != 0:
            current_side = 1 if position.side == "long" else -1
            if desired_side != current_side:
                cash += _close(position, t, opens[t], "opposite_signal", "open")
                position, shares = None, 0

        # 3. Entrada al open con la señal y el ATR de la barra anterior.
        if position is None and desired_side != 0 and not np.isnan(atr_vals[t - 1]):
            q, truncated = compute_sizing(cash, atr_vals[t - 1], opens[t], config.rho, c)
            if q > 0:
                side = "long" if desired_side == 1 else "short"
                position, cash_delta, shares = _open(t, side, q, atr_vals[t - 1], truncated)
                cash += cash_delta

        # 4 y 5. SL/TP intrabar (incluida la barra de entrada) y holding maximo al cierre.
        if position is not None:
            closed, reason, raw_exit = resolve_exit(position, highs[t], lows[t], closes[t], t,
                                                    config.max_holding)
            if closed:
                phase = "close" if reason == "max_holding" else "intrabar"
                cash += _close(position, t, raw_exit, reason, phase)
                position, shares = None, 0

        cash_hist[t] = cash
        shares_hist[t] = shares

    equity = pd.DataFrame({
        "cash": cash_hist,
        "shares": shares_hist,
        "equity": cash_hist + shares_hist * closes,
    }, index=df.index)
    return BacktestResult(equity=equity, trades=pd.DataFrame(trades, columns=TRADE_COLUMNS))
