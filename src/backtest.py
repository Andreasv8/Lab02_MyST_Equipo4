"""Motor de backtest orientado a eventos (barra por barra) de lab_02.

backtest() es una funcion pura: recibe precios, señal, ATR y configuracion,
y regresa la curva de capital por barra y la lista de trades. La logica de
salida y sizing vive en este mismo modulo (docs/SPEC.md, secciones 4, 5 y 7).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.signals import THETA0, compute_strategy

# Costos por lado (docs/SPEC.md, seccion 6): comision de 0.125% y slippage 0 en el
# escenario base. El efecto de costos mas altos se mide con la curva de costos.
COMMISSION_RATE = 0.00125
SLIPPAGE_RATE = 0.0
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


def compute_sizing(capital: float, atr_value: float, entry_price: float, rho: float = 0.01,
                   sl_mult: float = 2.0, cost_rate: float = TOTAL_COST_RATE) -> tuple[float, bool]:
    """Calcula cuantas unidades comprar para arriesgar rho del capital (docs/SPEC.md, seccion 5).

    unidades = rho * capital / (sl_mult * ATR): si el precio llega al stop,
    que esta a sl_mult * ATR de la entrada, se pierde rho del capital.
    Sin apalancamiento: si el nocional mas el costo de entrada pasa del
    capital (unidades * P_e * (1 + c) > V), se recorta al capital disponible
    y se marca como truncado.

    Parametros
    ----------
    capital : float
        Capital disponible (V_t).
    atr_value : float
        ATR de la barra de señal.
    entry_price : float
        Precio crudo de entrada (P_e, open de la barra de ejecucion).
    rho : float
        Fraccion del capital que se arriesga por operacion.
    sl_mult : float
        Distancia del stop-loss en multiplos del ATR.
    cost_rate : float
        Costo por lado (comision + slippage) como fraccion del nocional.

    Regresa
    -------
    tuple[float, bool]
        (unidades, si se recorto por falta de capital).
    """
    shares = rho * capital / (sl_mult * atr_value)
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
    "entry_commission", "exit_commission", "entry_slippage", "exit_slippage",
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
    commission_rate : float
        Comision por lado como fraccion del nocional.
    slippage_rate : float
        Slippage por lado como fraccion del nocional.
    borrow_fee_annual : float
        Tasa anual de borrow de los shorts sobre el nocional.
    """

    initial_cash: float = 1_000_000.0
    rho: float = 0.01
    sl_mult: float = 2.0
    tp_mult: float = 3.0
    max_holding: int = 10
    commission_rate: float = COMMISSION_RATE
    slippage_rate: float = SLIPPAGE_RATE
    borrow_fee_annual: float = BORROW_FEE_ANNUAL

    @property
    def cost_rate(self) -> float:
        """Costo total por lado: comision + slippage."""
        return self.commission_rate + self.slippage_rate


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
             max_holding: Optional[pd.Series] = None,
             force_exit: Optional[pd.Series] = None,
             rho: Optional[pd.Series] = None) -> BacktestResult:
    """Simula la estrategia barra por barra con estado explicito de caja.

    Cada barra t se procesa en el orden de docs/SPEC.md, seccion 7:

    1. Open: si el open ya cruzo el SL/TP (gap), se cierra al open.
    1b. Open: si force_exit de t-1 es True, se cierra al open ("regime_exit").
    2. Open: si la señal de t-1 es opuesta a la posicion, se cierra al open.
    3. Open: si no hay posicion y la señal de t-1 es != 0, se abre al open
       con el ATR de t-1 (sizing con V_t = cash, que estando flat es el
       capital realizado).
    4. Intrabar: SL/TP con el High/Low de t (tie -> SL).
    5. Close: salida por holding maximo al cierre de t.

    Contabilidad de caja, con P crudo, q unidades y costo de cada lado
    = (comision + slippage) * P * q:
    entrada long cash -= P*q + costo; salida long cash += P*q - costo;
    entrada short cash += P*q - costo; salida short cash -= P*q + costo + borrow_fee.
    Cada trade guarda esos mismos montos en entry_commission, exit_commission,
    entry_slippage y exit_slippage, asi que las columnas cuadran con lo cobrado.
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
    force_exit : pd.Series de bool, opcional
        Salida forzada por barra (regla R3: el regimen cambia a crisis). Si es
        True en t-1 y hay posicion, se cierra al open de t pagando comision;
        igual que la señal, se decide en t-1 y se ejecuta en t (sin look-ahead).
        None = nunca se fuerza la salida.
    rho : pd.Series, opcional
        Fraccion de capital en riesgo por barra (p. ej. segun el regimen). Se
        toma de la barra de señal t-1, igual que sl_mult. None = config.rho.

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
    rho_vals = _per_bar(rho, config.rho, "rho")
    # Salida forzada por barra; un NaN cuenta como "no forzar".
    if force_exit is None:
        exit_flags = np.zeros(len(df), dtype=bool)
    elif not force_exit.index.equals(df.index):
        raise ValueError("force_exit debe tener el mismo indice que df")
    else:
        exit_flags = force_exit.fillna(False).to_numpy(dtype=bool)

    sig = signal.to_numpy()
    atr_vals = atr.to_numpy(dtype=float)
    dates = df.index
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    n = len(df)
    c = config.cost_rate

    def _side_costs(raw_price: float, q: float) -> tuple[float, float]:
        """Comision y slippage en dolares de una operacion (un lado) de q unidades a raw_price."""
        notional = raw_price * q
        return config.commission_rate * notional, config.slippage_rate * notional

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
        commission, slippage = _side_costs(raw, q)
        if side == "long":
            stop_loss = raw - sl * atr_at_entry
            take_profit = raw + tp * atr_at_entry
            cash_delta, signed = -(raw * q + commission + slippage), q
        else:
            stop_loss = raw + sl * atr_at_entry
            take_profit = raw - tp * atr_at_entry
            cash_delta, signed = raw * q - commission - slippage, -q

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
        # Mismos montos que se cobraron al abrir (entrada) y que se cobran ahora (salida).
        entry_commission, entry_slippage = _side_costs(position.raw_entry_price, q)
        exit_commission, exit_slippage = _side_costs(raw_exit, q)
        if position.side == "long":
            cash_delta = raw_exit * q - exit_commission - exit_slippage
            pnl = q * (exit_price - position.entry_price)
        else:
            cash_delta = -(raw_exit * q + exit_commission + exit_slippage + borrow_fee)
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
            "entry_commission": entry_commission,
            "exit_commission": exit_commission,
            "entry_slippage": entry_slippage,
            "exit_slippage": exit_slippage,
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

        # 1b. Salida forzada por regimen, decidida en t-1 (regla R3).
        if position is not None and exit_flags[t - 1]:
            cash += _close(position, t, opens[t], "regime_exit", "open")
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
            q, truncated = compute_sizing(cash, atr_vals[t - 1], opens[t], rho=rho_vals[t - 1],
                                          sl_mult=sl_vals[t - 1], cost_rate=c)
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


def config_from_params(params: dict, capital: float = 1_000_000.0) -> BacktestConfig:
    """Arma la configuracion del motor a partir de los parametros θ.

    El take-profit se mide en multiplos del stop: tp_mult = rr · sl_mult.
    Recibe el diccionario θ (rho, sl_mult, rr, max_holding) y el capital inicial.
    Regresa un BacktestConfig con los costos por defecto.
    """
    return BacktestConfig(initial_cash=capital, rho=params["rho"], sl_mult=params["sl_mult"],
                          tp_mult=params["rr"] * params["sl_mult"],
                          max_holding=int(params["max_holding"]))


def run_backtest(df: pd.DataFrame, params: dict = THETA0,
                 capital: float = 1_000_000.0) -> list[dict]:
    """Corre la estrategia final completa (señal + salidas + sizing) sobre df.

    Calcula la señal y el ATR de 4h con compute_strategy y simula con
    backtest(), que es el unico motor (orden de eventos de docs/SPEC.md,
    seccion 7). Una posicion que sigue abierta al final no se registra.

    Recibe las velas de 5 min, los parametros θ y el capital inicial.
    Regresa una lista con un diccionario por operacion cerrada (columnas de
    TRADE_COLUMNS). pnl es neto de comision, slippage y borrow fee.
    """
    features = compute_strategy(df, params)
    config = config_from_params(params, capital)
    result = backtest(df, features["signal"], features["atr"], config)
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

