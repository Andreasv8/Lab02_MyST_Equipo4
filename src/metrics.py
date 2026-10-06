"""Metricas de desempeño de lab_02 sobre la salida de src.backtest.backtest().

Todas las funciones son puras: reciben la equity por barra y/o el DataFrame
de trades y no leen archivos. Convenciones:
- N = numero de periodos transcurridos = len(equity) - 1; años = N / 252.
- Tasa libre de riesgo = 0.
"""

import math
from dataclasses import replace
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from src.backtest import backtest, compute_win_rate, config_from_params
from src.signals import compute_strategy

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
    """Win rate empirico vs break-even teorico, sin y con costos (docs/SPEC.md, break-even).

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
    q = initial_cash / entry_cost
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


# ---------------------------------------------------------------------------
# Resumen de desempeño y tabla de retornos (PDF 3.2)
# ---------------------------------------------------------------------------

def performance_summary(equity_df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Metricas principales de una curva: Sharpe, Sortino, Calmar, MDD, win rate y mas.

    Recibe la equity por barra (columnas equity y shares, como backtest().equity)
    y los trades (puede estar vacio, como en buy & hold).
    Regresa una Series con sharpe, sortino, calmar, max_drawdown (fraccion <= 0),
    win_rate (% de trades con pnl > 0; NaN sin trades), total_return (fraccion),
    n_trades y exposure (fraccion de barras con posicion).
    """
    equity = equity_df["equity"]
    r = returns(equity)
    n_trades = len(trades)
    win_rate = compute_win_rate(trades.to_dict("records")) if n_trades else np.nan
    return pd.Series({
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate,
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "n_trades": n_trades,
        "exposure": exposure(equity_df),
    })


def returns_table(equity: pd.Series, freq: str) -> pd.Series:
    """Retorno de cada periodo calendario: mensual ("ME"), trimestral ("QE") o anual ("YE").

    retorno = ultimo valor del periodo / ultimo valor del periodo anterior - 1.
    El primer periodo se mide contra el primer valor de la serie (puede ser
    un periodo incompleto).
    Recibe la equity por barra y la frecuencia de pandas.
    Regresa una Series con indice = fin de cada periodo.
    """
    period_end = equity.resample(freq).last().dropna()
    previous = period_end.shift(1)
    previous.iloc[0] = equity.iloc[0]
    return period_end / previous - 1


# ---------------------------------------------------------------------------
# Robustez a costos de transaccion
# ---------------------------------------------------------------------------

def cost_sensitivity(df: pd.DataFrame, params: dict, periods: dict,
                     round_trip_bps: Iterable[float] = range(0, 55, 5)) -> pd.DataFrame:
    """Sharpe y equity final de la estrategia para varios costos de ida y vuelta.

    La señal se calcula una vez con compute_strategy(df, params). Para cada
    costo de ida y vuelta (en bps) se corre el backtest con comision por lado
    = bps / 2 / 10_000 y slippage 0, y cada periodo se mide por separado con
    summarize (equity rebasada al inicio del periodo). Sirve para dibujar la
    curva de retorno neto contra nivel de costo.

    Recibe las velas de 5 min, los parametros θ, los periodos
    {nombre: (inicio, fin)} y la malla de costos en bps.
    Regresa un DataFrame con indice bps y columnas sharpe_<periodo> y
    equity_final_<periodo>.
    """
    features = compute_strategy(df, params)
    base_config = config_from_params(params)

    rows = []
    for bps in round_trip_bps:
        config = replace(base_config, commission_rate=bps / 2 / 10_000, slippage_rate=0.0)
        result = backtest(df, features["signal"], features["atr"], config)
        row = {"bps": bps}
        for name, (start, end) in periods.items():
            summary = summarize(result.equity, result.trades, config, start, end)
            row[f"sharpe_{name}"] = summary["sharpe"]
            row[f"equity_final_{name}"] = summary["equity_final"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("bps")


def break_even_cost(bps: pd.Series, sharpe: pd.Series) -> float:
    """Costo de ida y vuelta en el que el Sharpe cruza 0 por primera vez (break-even).

    Interpola en linea recta entre los dos puntos de la malla donde el Sharpe
    pasa de > 0 a <= 0:  bps* = b_0 + S_0 · (b_1 - b_0) / (S_0 - S_1).

    Recibe los costos en bps (en orden creciente) y el Sharpe en cada costo.
    Regresa el costo de break-even en bps; NaN si el Sharpe no cruza 0 en la
    malla (siempre > 0, o ya <= 0 desde el primer punto).
    """
    b = np.asarray(bps, dtype=float)
    s = np.asarray(sharpe, dtype=float)
    if len(s) == 0 or not s[0] > 0:
        return np.nan
    for i in range(1, len(s)):
        if s[i] <= 0:
            return b[i - 1] + s[i - 1] * (b[i] - b[i - 1]) / (s[i - 1] - s[i])
    return np.nan


# ---------------------------------------------------------------------------
# Analisis de trades: por regimen y diagnostico (docs/SPEC.md, seccion 14.4)
# ---------------------------------------------------------------------------

def trade_returns(trades: pd.DataFrame) -> pd.Series:
    """Retorno de cada trade: pnl neto / nocional de entrada (unidades · precio crudo)."""
    return trades["pnl"] / (trades["shares"] * trades["raw_entry_price"])


def bootstrap_mean_ci(values, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Intervalo de confianza del 95% de la media con bootstrap.

    Remuestrea con reemplazo n_boot veces, calcula la media de cada muestra y
    toma los percentiles 2.5 y 97.5. NaN si hay menos de 2 valores.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def regime_trade_stats(trades: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Metricas de los trades agrupados por regimen de entrada.

    Recibe los trades y el regimen de entrada de cada trade (mismo indice).
    Regresa una tabla por regimen con n_trades, win_rate (%), mean_return
    (retorno promedio por trade) y su IC bootstrap del 95% (ci_low, ci_high).
    """
    rets = trade_returns(trades)
    rows = {}
    for regime in sorted(regimes.dropna().unique()):
        r = rets[regimes == regime]
        ci_low, ci_high = bootstrap_mean_ci(r)
        rows[regime] = {"n_trades": len(r), "win_rate": 100 * (r > 0).mean(), "mean_return": r.mean(),
                        "ci_low": ci_low, "ci_high": ci_high}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("regime")


def kruskal_by_regime(trades: pd.DataFrame, regimes: pd.Series) -> dict:
    """Prueba de Kruskal-Wallis: ¿los retornos por trade difieren entre regimenes?

    Usa los regimenes con al menos 2 trades. Regresa {"h", "p_value", "groups"};
    NaN si hay menos de 2 grupos.
    """
    rets = trade_returns(trades)
    groups = [rets[regimes == r].to_numpy() for r in sorted(regimes.dropna().unique())]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"h": np.nan, "p_value": np.nan, "groups": len(groups)}
    h, p_value = kruskal(*groups)
    return {"h": float(h), "p_value": float(p_value), "groups": len(groups)}


def _group_table(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    """Numero de trades, win rate (%) y PnL total agrupando por una columna."""
    grouped = trades.groupby(column)["pnl"]
    return pd.DataFrame({
        "n_trades": grouped.size(),
        "win_rate": grouped.apply(lambda pnl: 100 * (pnl > 0).mean()),
        "total_pnl": grouped.sum(),
    })


def exit_reason_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades por motivo de salida (stop_loss, take_profit, max_holding, ...)."""
    return _group_table(trades, "exit_reason")


def side_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades largos contra cortos."""
    return _group_table(trades, "side")


def pnl_breakdown(trades: pd.DataFrame) -> pd.Series:
    """De donde viene el PnL: bruto, costos y neto; payoff y win rate de break-even.

    pnl_bruto = pnl neto + comisiones + slippage + borrow fee.
    payoff = ganancia promedio / perdida promedio (en valor absoluto).
    win rate de break-even = 1 / (1 + payoff): con ese % de aciertos, las
    ganancias pagan justo las perdidas.
    """
    pnl = trades["pnl"]
    commissions = (trades["entry_commission"] + trades["exit_commission"]).sum()
    slippage = (trades["entry_slippage"] + trades["exit_slippage"]).sum()
    borrow = trades["borrow_fee"].sum()
    avg_win = pnl[pnl > 0].mean()
    avg_loss = -pnl[pnl < 0].mean()
    payoff = avg_win / avg_loss
    return pd.Series({
        "gross_pnl": pnl.sum() + commissions + slippage + borrow,
        "commissions": commissions,
        "slippage": slippage,
        "borrow_fee": borrow,
        "net_pnl": pnl.sum(),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "win_rate": 100 * (pnl > 0).mean(),
        "break_even_win_rate": 100 / (1 + payoff),
    })


# ---------------------------------------------------------------------------
# Impacto de mercado (PDF seccion 4: advertencia obligatoria)
# ---------------------------------------------------------------------------

def market_impact_table(df: pd.DataFrame, trades_by_curve: dict, break_even_bps: float) -> pd.DataFrame:
    """Estimacion simple del impacto de mercado de nuestras ordenes (con el archivo de train).

    Compara el nocional promedio por trade (unidades · precio de entrada) con
    el volumen en dolares de una barra de 5 min y de una vela de 4h (medianas,
    solo barras con volumen > 0), y estima el costo extra con el modelo de
    raiz cuadrada:

        impacto ≈ σ_5min · sqrt(nocional / volumen_5min)

    σ_5min es la desviacion estandar de los rendimientos de 5 min. El modelo
    es solo un orden de magnitud: supone que la orden se ejecuta en una barra
    y que el volumen observado es todo el que hay.

    Supuestos y limitaciones:
    - La columna Volume ya esta en dolares (Yahoo BTC-USD): multiplicarla por
      el precio daria volumenes absurdos.
    - El volumen falta en ~48% de las barras del archivo crudo (quedan en 0
      tras la limpieza); se usan solo barras con volumen > 0.

    Recibe las velas de 5 min de train, {curva: trades} y el break-even en pb
    por lado de la curva de costos.
    Regresa una fila por curva.
    """
    volume_5min = df.loc[df["Volume"] > 0, "Volume"]
    volume_4h = df["Volume"].resample("4h").sum()
    volume_4h = volume_4h[volume_4h > 0]
    sigma = returns(df["Close"]).std()
    rows = {}
    for name, trades in trades_by_curve.items():
        notional = (trades["shares"] * trades["raw_entry_price"]).mean()
        participation = notional / volume_5min.median()
        impact_bps = sigma * math.sqrt(participation) * 10_000
        rows[name] = {
            "avg_notional_usd": notional,
            "median_volume_5min_usd": volume_5min.median(),
            "median_volume_4h_usd": volume_4h.median(),
            "participation_5min_pct": 100 * participation,
            "participation_4h_pct": 100 * notional / volume_4h.median(),
            "sigma_5min": sigma,
            "impact_bps": impact_bps,
            "break_even_bps": break_even_bps,
            "impact_vs_break_even_pct": 100 * impact_bps / break_even_bps,
            "bars_without_volume_pct": 100 * (df["Volume"] <= 0).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("curve")
