"""Estrategia de lab_02: entry rule, exit rule, sizing y maquina de estados segun SPEC.md.

Los indicadores viven en src/indicators.py (calculados a mano con
pandas/numpy, sin librerias externas).
"""


from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.indicators import adx, atr, chaikin_money_flow, roc

COMMISSION_RATE = 0.00125
SLIPPAGE_RATE = 0.0005
TOTAL_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE

# Borrow fee de los shorts: tasa anual sobre el nocional, dias calendario / 360.
BORROW_FEE_ANNUAL = 0.0
BORROW_DAY_COUNT = 360

# Umbral de Wilder para considerar que el mercado esta en tendencia.
ADX_THRESHOLD = 25


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


def confluence_signal(roc_10: pd.Series, cmf_20: pd.Series, adx_14: pd.Series,
                      adx_threshold: float = ADX_THRESHOLD) -> pd.Series:
    """Regla de entrada por confluencia (SPEC.md, seccion 3).

    long  si ROC > 0 y CMF > 0 y ADX > umbral
    short si ROC < 0 y CMF < 0 y ADX > umbral
    flat  en cualquier otro caso. Las comparaciones son estrictas, asi que
    los NaN del warm-up y los valores exactamente 0 quedan en flat.

    Parametros
    ----------
    roc_10, cmf_20, adx_14 : pd.Series
        Indicadores al cierre de cada barra, con el mismo indice.
    adx_threshold : float
        Umbral de fuerza de tendencia (25, Wilder).

    Regresa
    -------
    pd.Series
        Señal por barra: 1 (long), -1 (short) o 0 (flat).
    """
    trending = adx_14 > adx_threshold
    signal = pd.Series(0, index=roc_10.index, dtype=int)
    signal.loc[(roc_10 > 0) & (cmf_20 > 0) & trending] = 1
    signal.loc[(roc_10 < 0) & (cmf_20 < 0) & trending] = -1
    return signal


def compute_features(df: pd.DataFrame, roc_window: int = 10, cmf_window: int = 20,
                     adx_window: int = 14, adx_threshold: float = ADX_THRESHOLD) -> pd.DataFrame:
    """Calcula los indicadores de la estrategia y la señal (SPEC.md, secciones 2 y 3).

    Los defaults son los del SPEC (θ0). Las ventanas y el umbral se pueden
    cambiar para la optimizacion (Act 07); el ATR queda fijo en 14 porque
    solo define SL/TP y sizing, no la señal.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Close", "High", "Low", "Volume".
    roc_window, cmf_window, adx_window : int
        Ventanas de ROC, CMF y ADX.
    adx_threshold : float
        Umbral de ADX para considerar que hay tendencia.

    Regresa
    -------
    pd.DataFrame
        Columnas: roc_10, cmf_20, adx_14, atr_14, signal. signal es 1
        (long), -1 (short) o 0 (flat). El ATR solo se usa para SL/TP y sizing.
        Los nombres conservan el sufijo del default aunque se usen otras
        ventanas, para no romper a quienes ya leen esas columnas.
    """
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    roc_10 = roc(close, roc_window)
    cmf_20 = chaikin_money_flow(high, low, close, volume, cmf_window)
    adx_14 = adx(high, low, close, adx_window)
    atr_14 = atr(high, low, close, 14)

    return pd.DataFrame({
        "roc_10": roc_10,
        "cmf_20": cmf_20,
        "adx_14": adx_14,
        "atr_14": atr_14,
        "signal": confluence_signal(roc_10, cmf_20, adx_14, adx_threshold),
    })


def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
    """Señal de entrada por barra con la regla de confluencia (SPEC.md, seccion 3).

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


def compute_sizing(capital: float, atr_value: float, entry_price: float,
                   rho: float = 0.01, cost_rate: float = TOTAL_COST_RATE) -> tuple[float, bool]:
    """Sizing por risk-parity segun ATR con apalancamiento maximo 1 (SPEC.md, seccion 5).

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
    """Costo de pedir prestadas las acciones de un short (SPEC.md, seccion 6).

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


def _adjust_entry_price(raw_price: float, side: str, cost_rate: float = TOTAL_COST_RATE) -> float:
    """Ajusta el precio de entrada por comision y slippage (SPEC.md, seccion 6).

    Un long paga de mas al entrar (precio efectivo mas alto); un short recibe
    de menos (precio efectivo mas bajo).
    """
    if side == "long":
        return raw_price * (1 + cost_rate)
    return raw_price * (1 - cost_rate)


def _adjust_exit_price(raw_price: float, side: str, cost_rate: float = TOTAL_COST_RATE) -> float:
    """Ajusta el precio de salida por comision y slippage (SPEC.md, seccion 6).

    Un long recibe de menos al salir; un short paga de mas al cubrir.
    """
    if side == "long":
        return raw_price * (1 - cost_rate)
    return raw_price * (1 + cost_rate)


def run_backtest(df: pd.DataFrame, capital: float = 1_000_000.0, rho: float = 0.01,
                 sl_mult: float = 2.0, tp_mult: float = 3.0, max_holding: int = 10) -> list[dict]:
    """Corre la estrategia completa (entry + exit + sizing) sobre df.

    Calcula los indicadores y la señal con compute_features y delega la
    simulacion en src.backtest.backtest, que es el unico motor (orden de
    eventos de SPEC.md, seccion 7; costos por defecto de la seccion 6).
    Una posicion que sigue abierta al final de la serie no se registra.

    Parametros
    ----------
    df : pd.DataFrame
        Debe incluir columnas "Open", "High", "Low", "Close", "Volume" e
        indice de fechas.
    capital, rho, sl_mult, tp_mult, max_holding
        Ver SPEC.md, secciones 4 y 5.

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
    # Import diferido: src.backtest importa de este modulo.
    from src.backtest import BacktestConfig, backtest

    features = compute_features(df)
    config = BacktestConfig(initial_cash=capital, rho=rho, sl_mult=sl_mult,
                            tp_mult=tp_mult, max_holding=max_holding)
    result = backtest(df, features["signal"], features["atr_14"], config)
    return result.trades.to_dict("records")


def compute_win_rate(trades: list[dict]) -> float:
    """Calcula el porcentaje de trades ganadores (SPEC.md, break-even win rate).

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
