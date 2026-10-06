"""Metricas de desempeño de lab_02 sobre la salida de src.backtest.backtest().

Todas las funciones son puras: reciben la equity por barra y/o el DataFrame
de trades y no leen archivos. Convenciones:
- N = numero de periodos transcurridos = len(equity) - 1; años = N / 252.
- Tasa libre de riesgo = 0.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy import compute_win_rate

BARS_PER_DAY = 288                      # 24 h x 12 barras de 5 minutos
PERIODS_PER_YEAR = BARS_PER_DAY * 365


def _years(equity: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Años cubiertos por la serie: (len(equity) - 1) / periods (NaN si hay menos de 2 barras)."""
    n = len(equity) - 1
    return n / periods if n > 0 else np.nan


def returns(equity: pd.Series) -> pd.Series:
    """Rendimientos simples por barra.

    r_t = E_t / E_{t-1} - 1

    Parametros
    ----------
    equity : pd.Series
        Valor de la cuenta por barra.

    Regresa
    -------
    pd.Series
        Rendimientos, sin la primera barra (no tiene barra previa).
    """
    return (equity / equity.shift(1) - 1).iloc[1:]


def sharpe_ratio(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Sharpe ratio anualizado con tasa libre = 0.

    Sharpe = media(r) / std(r) · sqrt(periods), con std muestral (ddof = 1).

    Parametros
    ----------
    r : pd.Series
        Rendimientos por periodo.
    periods : int
        Periodos por año (252 para barras diarias).

    Regresa
    -------
    float
        Sharpe anualizado; NaN si hay menos de 2 rendimientos o std = 0.
    """
    std = r.std(ddof=1)
    if len(r) < 2 or not std > 0:
        return np.nan
    return r.mean() / std * math.sqrt(periods)


def sortino_ratio(r: pd.Series, periods: int = PERIODS_PER_YEAR, threshold: float = 0.0) -> float:
    """Sortino ratio anualizado (formula de clase).

    Sortino = media(r - τ) / sqrt( (1/T) · Σ min(r_t - τ, 0)^2 ) · sqrt(periods)

    El denominador (downside deviation) divide entre T, el total de
    rendimientos, no solo entre los negativos.

    Parametros
    ----------
    r : pd.Series
        Rendimientos por periodo.
    periods : int
        Periodos por año.
    threshold : float
        Rendimiento minimo aceptable τ por periodo.

    Regresa
    -------
    float
        Sortino anualizado; NaN si no hay rendimientos por debajo de τ.
    """
    excess = r - threshold
    downside = math.sqrt((np.minimum(excess, 0.0) ** 2).mean()) if len(r) > 0 else 0.0
    if not downside > 0:
        return np.nan
    return excess.mean() / downside * math.sqrt(periods)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown por barra sobre el capital (no sobre rendimientos).

    DD_t = E_t / max_{s<=t} E_s - 1   (siempre <= 0)

    Parametros
    ----------
    equity : pd.Series
        Valor de la cuenta por barra.

    Regresa
    -------
    pd.Series
        Drawdown por barra, mismo indice que equity.
    """
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """Maximo drawdown: min_t DD_t (un numero <= 0, p. ej. -0.10 = -10%)."""
    return float(drawdown_series(equity).min())


def cagr(equity: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Tasa de crecimiento anual compuesta.

    CAGR = (E_final / E_inicial)^(periods / N) - 1,  N = len(equity) - 1.

    Parametros
    ----------
    equity : pd.Series
        Valor de la cuenta por barra.
    periods : int
        Periodos por año.

    Regresa
    -------
    float
        CAGR; NaN si hay menos de 2 barras.
    """
    years = _years(equity, periods)
    if not years > 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def calmar_ratio(equity: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Calmar ratio = CAGR / |max drawdown|; NaN si no hubo drawdown."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.nan
    return cagr(equity, periods) / abs(mdd)


def turnover_stats(equity: pd.Series, trades: pd.DataFrame, cost_rate: float,
                   periods: int = PERIODS_PER_YEAR) -> dict:
    """Rotacion anual y costo anual que la estrategia debe superar.

    nocional = Σ raw_entry_price · shares (entradas) + Σ raw_exit_price · shares (salidas)
    turnover_annual = nocional / media(E) / años
    cost_hurdle_annual = turnover_annual · cost_rate + Σ borrow_fee / media(E) / años
    trades_per_year = trades cerrados / años

    Cada pata cuenta en el periodo de equity donde ocurre: la entrada si
    entry_date cae en el rango de fechas de equity, y la salida (con su
    borrow fee y el conteo del trade) si cae exit_date. Asi un trade que
    cruza de un periodo a otro no se cuenta dos veces.

    Parametros
    ----------
    equity : pd.Series
        Equity por barra en dolares (sin rebasar), indice de fechas.
    trades : pd.DataFrame
        Trades de backtest() (usa raw_entry_price, raw_exit_price, shares,
        borrow_fee, entry_date, exit_date).
    cost_rate : float
        Costo por lado como fraccion del nocional.
    periods : int
        Periodos por año.

    Regresa
    -------
    dict
        turnover_annual, cost_hurdle_annual, trades_per_year.
    """
    first, last = equity.index[0], equity.index[-1]
    entries = trades[trades["entry_date"].between(first, last)]
    exits = trades[trades["exit_date"].between(first, last)]

    notional = ((entries["raw_entry_price"] * entries["shares"]).sum()
                + (exits["raw_exit_price"] * exits["shares"]).sum())
    mean_equity = equity.mean()
    years = _years(equity, periods)

    turnover_annual = notional / mean_equity / years
    borrow_annual = exits["borrow_fee"].sum() / mean_equity / years
    return {
        "turnover_annual": turnover_annual,
        "cost_hurdle_annual": turnover_annual * cost_rate + borrow_annual,
        "trades_per_year": len(exits) / years,
    }


def win_rate_stats(trades: pd.DataFrame, sl_mult: float, tp_mult: float, cost_rate: float) -> dict:
    """Win rate empirico vs break-even teorico, sin y con costos (SPEC.md, break-even).

    empirico = fraccion de trades con pnl neto > 0
    r = tp_mult / sl_mult;  p* = 1 / (1 + r)
    k_i = (cost_rate · Q · (P_e + P_x) + borrow_fee) / (sl_mult · ATR · Q),
          con P_e, P_x precios crudos y ATR de la barra de señal
    p*_costos = (1 + media(k)) / (1 + r)

    Parametros
    ----------
    trades : pd.DataFrame
        Trades de backtest().
    sl_mult, tp_mult : float
        Multiplos de ATR del stop-loss y del take-profit.
    cost_rate : float
        Costo por lado.

    Regresa
    -------
    dict
        win_rate, p_star, k_mean, p_star_cost (fracciones, no %). win_rate,
        k_mean y p_star_cost son NaN si no hay trades.
    """
    r = tp_mult / sl_mult
    p_star = 1 / (1 + r)
    if trades.empty:
        return {"win_rate": np.nan, "p_star": p_star, "k_mean": np.nan, "p_star_cost": np.nan}

    round_trip_cost = (cost_rate * trades["shares"] * (trades["raw_entry_price"] + trades["raw_exit_price"])
                       + trades["borrow_fee"])
    risk = sl_mult * trades["entry_atr"] * trades["shares"]
    k_mean = (round_trip_cost / risk).mean()
    return {
        "win_rate": compute_win_rate(trades.to_dict("records")) / 100,
        "p_star": p_star,
        "k_mean": k_mean,
        "p_star_cost": (1 + k_mean) / (1 + r),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    """Numero de trades, P&L promedio y profit factor.

    profit_factor = Σ pnl de ganadores / |Σ pnl de perdedores|

    Parametros
    ----------
    trades : pd.DataFrame
        Trades de backtest() (usa pnl).

    Regresa
    -------
    dict
        n_trades, avg_pnl (NaN sin trades), profit_factor (inf si no hay
        perdedores y si ganadores; NaN sin trades).
    """
    pnl = trades["pnl"]
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses > 0:
        profit_factor = gains / losses
    else:
        profit_factor = np.inf if gains > 0 else np.nan
    return {
        "n_trades": len(trades),
        "avg_pnl": pnl.mean() if len(pnl) else np.nan,
        "profit_factor": profit_factor,
    }


def exposure(equity_df: pd.DataFrame) -> float:
    """Fraccion de barras con posicion abierta al cierre (shares != 0).

    Un trade que abre y cierra en la misma barra no cuenta como exposicion.

    Parametros
    ----------
    equity_df : pd.DataFrame
        Salida equity de backtest() (usa la columna shares).

    Regresa
    -------
    float
        Exposicion en [0, 1].
    """
    return float((equity_df["shares"] != 0).mean())


def buy_and_hold_equity(df: pd.DataFrame, initial_cash: float, cost_rate: float) -> pd.DataFrame:
    """Benchmark buy & hold: compra al primer open con costo y mantiene.

    Q = floor(initial_cash / (Open_0 · (1 + c)))
    cash = initial_cash - Q · Open_0 · (1 + c)
    E_t = cash + Q · Close_t

    Igual que el motor, la posicion no se liquida al final (sin costo de salida).

    Parametros
    ----------
    df : pd.DataFrame
        Columnas "Open" y "Close" con indice de fechas.
    initial_cash : float
        Capital inicial.
    cost_rate : float
        Costo por lado de la compra.

    Regresa
    -------
    pd.DataFrame
        cash, shares y equity por barra (mismo formato que backtest().equity).
    """
    entry_cost = df["Open"].iloc[0] * (1 + cost_rate)
    q = math.floor(initial_cash / entry_cost)
    cash = initial_cash - q * entry_cost
    return pd.DataFrame({
        "cash": cash,
        "shares": q,
        "equity": cash + q * df["Close"],
    }, index=df.index)


def summarize(equity_df: pd.DataFrame, trades: pd.DataFrame, config,
              start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
    """Resumen de metricas de un periodo (completo o sub-periodo).

    Recorta equity_df a [start, end] y rebasa la equity a config.initial_cash
    en la primera barra del periodo (E / E_0 · initial_cash), para comparar
    train y test por separado. Las metricas de razon (Sharpe, DD, CAGR, ...)
    no cambian con el rebase. Los trades del periodo para win rate, profit
    factor y P&L son los que cierran en el periodo (exit_date); en turnover,
    cada pata cuenta en su propio periodo (ver turnover_stats).

    Parametros
    ----------
    equity_df : pd.DataFrame
        Salida equity de backtest() o de buy_and_hold_equity().
    trades : pd.DataFrame
        Trades de backtest() (puede estar vacio, p. ej. para buy & hold).
    config : BacktestConfig
        Usa initial_cash, sl_mult, tp_mult y cost_rate.
    start, end : str, opcional
        Limites del periodo (inclusive). None = desde el inicio / hasta el final.

    Regresa
    -------
    pd.Series
        equity_final, total_return, cagr, sharpe, sortino, max_drawdown,
        calmar, n_trades, trades_per_year, avg_pnl, profit_factor, win_rate,
        p_star, p_star_cost, k_mean, exposure, turnover_annual,
        cost_hurdle_annual.
    """
    period = equity_df.loc[start:end]
    equity = period["equity"]
    rebased = equity / equity.iloc[0] * config.initial_cash
    r = returns(rebased)

    period_trades = trades[trades["exit_date"].between(equity.index[0], equity.index[-1])]
    win = win_rate_stats(period_trades, config.sl_mult, config.tp_mult, config.cost_rate)
    turnover = turnover_stats(equity, trades, config.cost_rate)
    stats = trade_stats(period_trades)

    return pd.Series({
        "equity_final": rebased.iloc[-1],
        "total_return": rebased.iloc[-1] / rebased.iloc[0] - 1,
        "cagr": cagr(rebased),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "max_drawdown": max_drawdown(rebased),
        "calmar": calmar_ratio(rebased),
        "n_trades": stats["n_trades"],
        "trades_per_year": turnover["trades_per_year"],
        "avg_pnl": stats["avg_pnl"],
        "profit_factor": stats["profit_factor"],
        "win_rate": win["win_rate"],
        "p_star": win["p_star"],
        "p_star_cost": win["p_star_cost"],
        "k_mean": win["k_mean"],
        "exposure": exposure(period),
        "turnover_annual": turnover["turnover_annual"],
        "cost_hurdle_annual": turnover["cost_hurdle_annual"],
    })
