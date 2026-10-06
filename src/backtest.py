"""Motor de backtest orientado a eventos (barra por barra) de lab_02.

backtest() es una funcion pura: recibe precios, señal, ATR y configuracion,
y regresa la curva de capital por barra y la lista de trades. La logica de
salida y sizing vive en este mismo modulo (docs/SPEC.md, secciones 4, 5 y 7).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.signals import compute_features

COMMISSION_RATE = 0.00125
SLIPPAGE_RATE = 0.0005
TOTAL_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE

# Borrow fee de los shorts: tasa anual sobre el nocional, dias calendario / 360.
BORROW_FEE_ANNUAL = 0.0
BORROW_DAY_COUNT = 360


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
        Precio de entrada efectivo, ya ajustado por comision y slippage.
    stop_loss : float
        Precio de stop-loss.
    take_profit : float
        Precio de take-profit.
    entry_bar : int
        Indice (posicional) de la barra en la que se abrio la posicion.
    entry_date : pd.Timestamp
        Fecha de la barra de entrada (para el borrow fee).
    raw_entry_price : float
        Open crudo de la barra de entrada, sin costos (P_e del SPEC).
    truncated : bool
        True si el sizing se recorto por el tope de apalancamiento 1.
    """

    side: str
    shares: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    entry_date: pd.Timestamp
    raw_entry_price: float
    truncated: bool


def compute_sizing(capital: float, atr_value: float, entry_price: float,
                   rho: float = 0.01, cost_rate: float = TOTAL_COST_RATE) -> tuple[float, bool]:
    """Sizing por risk-parity segun ATR con apalancamiento maximo 1 (docs/SPEC.md, seccion 5).

    Q = floor(rho * capital / (2 * ATR)). Si el nocional mas el costo de
    entrada excede el capital (Q * P_e * (1 + c) > V), se recorta a
    Q = floor(V / (P_e * (1 + c))) y se marca como truncado.

    Parametros
    ----------
    capital : float
        Capital disponible (V_t).
    atr_value : float
        ATR(14) de la barra de señal.
    entry_price : float
        Precio crudo de entrada (P_e, open de la barra de ejecucion).
    rho : float
        Presupuesto de riesgo por trade, como fraccion del capital.
    cost_rate : float
        Costo por lado (comision + slippage) como fraccion del nocional.

    Regresa
    -------
    tuple[float, bool]
        (numero de acciones, si se trunco por el tope de apalancamiento).
    """
    shares = rho * capital / (2 * atr_value)
    cost_per_share = entry_price * (1 + cost_rate)

    if shares * cost_per_share > capital:
        return capital / cost_per_share, True
    return shares, False


def compute_borrow_fee(side: str, shares: int, raw_entry_price: float,
                       entry_date: pd.Timestamp, exit_date: pd.Timestamp,
                       fee_annual: float = BORROW_FEE_ANNUAL) -> float:
    """Costo de pedir prestadas las acciones de un short (docs/SPEC.md, seccion 6).

    fee = BORROW_FEE_ANNUAL * Q * P_e * dias / 360, con P_e crudo y dias
    calendario entre la entrada y la salida. Un short abierto y cerrado el
    mismo dia paga 0 (no hay overnight). Los longs no pagan borrow.

    Parametros
    ----------
    side : str
        "long" o "short".
    shares : int
        Numero de acciones de la posicion.
    raw_entry_price : float
        Open crudo de la barra de entrada.
    entry_date, exit_date : pd.Timestamp
        Fechas de las barras de entrada y salida.
    fee_annual : float
        Tasa anual de borrow sobre el nocional.

    Regresa
    -------
    float
        Borrow fee en dolares (0.0 para longs).
    """
    if side != "short":
        return 0.0
    days = (exit_date - entry_date).days
    return fee_annual * shares * raw_entry_price * days / BORROW_DAY_COUNT


def resolve_exit(position: Position, bar_high: float, bar_low: float, bar_close: float,
                 bar_index: int, max_holding: int = 10) -> tuple[bool, Optional[str], Optional[float]]:
    """Revisa si una posicion debe cerrarse en la barra actual (docs/SPEC.md, seccion 4).

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
        de entrada cuenta como barra 1 (ver docs/SPEC.md, seccion 4).

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
    y evita el sesgo optimista de llenar al SL en un gap (docs/SPEC.md, seccion 7).

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


def _adjust_entry_price(raw_price: float, side: str, cost_rate: float = TOTAL_COST_RATE) -> float:
    """Ajusta el precio de entrada por comision y slippage (docs/SPEC.md, seccion 6).

    Un long paga de mas al entrar (precio efectivo mas alto); un short recibe
    de menos (precio efectivo mas bajo).
    """
    if side == "long":
        return raw_price * (1 + cost_rate)
    return raw_price * (1 - cost_rate)


def _adjust_exit_price(raw_price: float, side: str, cost_rate: float = TOTAL_COST_RATE) -> float:
    """Ajusta el precio de salida por comision y slippage (docs/SPEC.md, seccion 6).

    Un long recibe de menos al salir; un short paga de mas al cubrir.
    """
    if side == "long":
        return raw_price * (1 - cost_rate)
    return raw_price * (1 + cost_rate)



TRADE_COLUMNS = [
    "entry_bar", "exit_bar", "entry_date", "exit_date", "side", "entry_price",
    "exit_price", "shares", "truncated", "exit_reason", "exit_phase", "borrow_fee", "pnl",
    "raw_entry_price", "raw_exit_price", "entry_atr",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Parametros del backtest (docs/SPEC.md, secciones 4, 5 y 6).

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

    initial_cash: float = 1_000_000.0
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
        Una fila por operacion cerrada, columnas TRADE_COLUMNS. entry_price y
        exit_price incluyen costos; raw_entry_price y raw_exit_price son los
        precios crudos de ejecucion; entry_atr es el ATR de la barra de señal.
    """

    equity: pd.DataFrame
    trades: pd.DataFrame


def backtest(df: pd.DataFrame, signal: pd.Series, atr: pd.Series,
             config: BacktestConfig, sl_mult: Optional[pd.Series] = None,
             tp_mult: Optional[pd.Series] = None,
             max_holding: Optional[pd.Series] = None) -> BacktestResult:
    """Simula la estrategia barra por barra con estado explicito de caja.

    Cada barra t se procesa en el orden de docs/SPEC.md, seccion 7:

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
    sl_mult, tp_mult, max_holding : pd.Series, opcional
        Parametros de salida por barra (mismo indice que df), p. ej. segun
        el regimen. Se fijan al ENTRAR con el valor de la barra de señal
        t-1 (igual que el ATR) y la posicion los conserva hasta salir.
        None = el escalar de config en todas las barras.

    Regresa
    -------
    BacktestResult
        equity (cash, shares, equity por barra) y trades.
    """
    if not signal.index.equals(df.index) or not atr.index.equals(df.index):
        raise ValueError("signal y atr deben tener el mismo indice que df")

    def _per_bar(values: Optional[pd.Series], default: float, name: str) -> np.ndarray:
        """Parametro por barra como arreglo; el escalar de config si values es None."""
        if values is None:
            return np.full(len(df), default, dtype=float)
        if not values.index.equals(df.index):
            raise ValueError(f"{name} debe tener el mismo indice que df")
        return values.to_numpy(dtype=float)

    sl_vals = _per_bar(sl_mult, config.sl_mult, "sl_mult")
    tp_vals = _per_bar(tp_mult, config.tp_mult, "tp_mult")
    holding_vals = _per_bar(max_holding, config.max_holding, "max_holding")

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
    shares = 0.0
    position: Optional[Position] = None
    position_holding = config.max_holding  # holding maximo fijado al entrar
    trades: list[dict] = []
    cash_hist = np.empty(n)
    shares_hist = np.zeros(n, dtype=float)

    def _open(t: int, side: str, q: float, atr_at_entry: float, truncated: bool) -> tuple[Position, float, float]:
        """Abre una posicion al open de t; regresa (posicion, delta de caja, shares con signo).

        SL y TP usan los multiplos de la barra de señal t-1.
        """
        raw = opens[t]
        sl, tp = sl_vals[t - 1], tp_vals[t - 1]
        if side == "long":
            stop_loss = raw - sl * atr_at_entry
            take_profit = raw + tp * atr_at_entry
            cash_delta, signed = -raw * q * (1 + c), q
        else:
            stop_loss = raw + sl * atr_at_entry
            take_profit = raw - tp * atr_at_entry
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
            "raw_entry_price": position.raw_entry_price,
            "raw_exit_price": raw_exit,
            # La entrada en entry_bar usa el ATR de la barra de señal (entry_bar - 1).
            "entry_atr": atr_vals[position.entry_bar - 1],
        })
        return cash_delta

    cash_hist[0] = cash

    for t in range(1, n):
        # 1. Gap en el open que ya cruzo el SL o el TP.
        if position is not None:
            closed, reason, raw_exit = resolve_open_gap(position, opens[t])
            if closed:
                cash += _close(position, t, raw_exit, reason, "open")
                position, shares = None, 0.0

        desired_side = int(sig[t - 1])

        # 2. Señal opuesta: se cierra al open.
        if position is not None and desired_side != 0:
            current_side = 1 if position.side == "long" else -1
            if desired_side != current_side:
                cash += _close(position, t, opens[t], "opposite_signal", "open")
                position, shares = None, 0.0

        # 3. Entrada al open con la señal y el ATR de la barra anterior.
        if position is None and desired_side != 0 and not np.isnan(atr_vals[t - 1]):
            q, truncated = compute_sizing(cash, atr_vals[t - 1], opens[t], config.rho, c)
            if q > 0:
                side = "long" if desired_side == 1 else "short"
                position, cash_delta, shares = _open(t, side, q, atr_vals[t - 1], truncated)
                position_holding = int(holding_vals[t - 1])
                cash += cash_delta

        # 4 y 5. SL/TP intrabar (incluida la barra de entrada) y holding maximo al cierre.
        if position is not None:
            closed, reason, raw_exit = resolve_exit(position, highs[t], lows[t], closes[t], t,
                                                    position_holding)
            if closed:
                phase = "close" if reason == "max_holding" else "intrabar"
                cash += _close(position, t, raw_exit, reason, phase)
                position, shares = None, 0.0

        cash_hist[t] = cash
        shares_hist[t] = shares

    equity = pd.DataFrame({
        "cash": cash_hist,
        "shares": shares_hist,
        "equity": cash_hist + shares_hist * closes,
    }, index=df.index)
    return BacktestResult(equity=equity, trades=pd.DataFrame(trades, columns=TRADE_COLUMNS))


def run_backtest(df: pd.DataFrame, capital: float = 1_000_000.0, rho: float = 0.01,
                 sl_mult: float = 2.0, tp_mult: float = 3.0, max_holding: int = 10) -> list[dict]:
    """Corre la estrategia completa (entry + exit + sizing) sobre df.

    Calcula los indicadores y la señal con compute_features y delega la
    simulacion en src.backtest.backtest, que es el unico motor (orden de
    eventos de docs/SPEC.md, seccion 7; costos por defecto de la seccion 6).
    Una posicion que sigue abierta al final de la serie no se registra.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Open", "High", "Low", "Close", "Volume" e
        indice de fechas.
    capital, rho, sl_mult, tp_mult, max_holding
        Ver docs/SPEC.md, secciones 4 y 5.

    Regresa
    -------
    list[dict]
        Un registro por operacion cerrada, con entry_bar, exit_bar,
        entry_date, exit_date, side, entry_price, exit_price, shares,
        truncated, exit_reason, exit_phase ("open", "intrabar" o "close"),
        borrow_fee y pnl. entry_price y exit_price ya incluyen el ajuste por
        costos; pnl es neto de comision, slippage y borrow fee. Toda entrada
        ocurre en el open de entry_bar.
    """
    features = compute_features(df)
    config = BacktestConfig(initial_cash=capital, rho=rho, sl_mult=sl_mult,
                            tp_mult=tp_mult, max_holding=max_holding)
    result = backtest(df, features["signal"], features["atr_14"], config)
    return result.trades.to_dict("records")


def compute_win_rate(trades: list[dict]) -> float:
    """Calcula el porcentaje de trades ganadores (docs/SPEC.md, break-even win rate).

    Un trade gana si su pnl neto es > 0. El pnl ya descuenta comision,
    slippage y borrow fee, asi que un trade con precio de salida apenas
    mejor que el de entrada puede contar como perdedor.

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

    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    return 100.0 * wins / len(trades)

