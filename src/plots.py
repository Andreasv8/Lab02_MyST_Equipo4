"""Graficas de lab_02 (matplotlib) para el notebook de la Act 06.

Cada funcion recibe los datos ya calculados y un Axes, y solo dibuja.
Convenciones: un eje y por grafica, estrategia en azul y buy & hold en gris,
rejilla tenue, titulos y ejes con unidades.
"""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

STRATEGY_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#52514e"
REFERENCE_COLOR = "#9a9893"


def _style(ax: plt.Axes) -> None:
    """Rejilla tenue y bordes superior/derecho ocultos."""
    ax.grid(True, color="#d9d8d4", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _split_line(ax: plt.Axes, split_date: pd.Timestamp, label: str = "inicio de test",
                at_top: bool = True) -> None:
    """Linea vertical en el corte train/test, con su etiqueta arriba o abajo."""
    ax.axvline(split_date, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
    y, offset = (1, -12) if at_top else (0, 6)
    ax.annotate(label, xy=(split_date, y), xycoords=("data", "axes fraction"),
                xytext=(4, offset), textcoords="offset points", fontsize=9, color="#3d3c39")


def plot_equity(ax: plt.Axes, strategy: pd.Series, benchmark: pd.Series,
                split_date: pd.Timestamp) -> plt.Axes:
    """Curva de equity de la estrategia vs buy & hold, en escala logaritmica.

    Parametros
    ----------
    ax : plt.Axes
        Ejes donde dibujar.
    strategy, benchmark : pd.Series
        Equity por barra (USD) con indice de fechas.
    split_date : pd.Timestamp
        Primer dia de test (linea vertical).

    Regresa
    -------
    plt.Axes
    """
    ax.plot(strategy.index, strategy, color=STRATEGY_COLOR, linewidth=2, label="Estrategia")
    ax.plot(benchmark.index, benchmark, color=BENCHMARK_COLOR, linewidth=2, label="Buy & hold NVDA")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    _split_line(ax, split_date)
    ax.set_title("Equity: estrategia vs buy & hold (escala log)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Equity (USD, escala log)")
    ax.legend(loc="upper left", frameon=False)
    _style(ax)
    return ax


def plot_drawdown(ax: plt.Axes, strategy_dd: pd.Series, benchmark_dd: pd.Series,
                  split_date: pd.Timestamp) -> plt.Axes:
    """Drawdown sobre el capital (siempre <= 0) de la estrategia y de buy & hold.

    Parametros
    ----------
    ax : plt.Axes
        Ejes donde dibujar.
    strategy_dd, benchmark_dd : pd.Series
        Drawdown por barra como fraccion (<= 0).
    split_date : pd.Timestamp
        Primer dia de test (linea vertical).

    Regresa
    -------
    plt.Axes
    """
    ax.plot(benchmark_dd.index, benchmark_dd * 100, color=BENCHMARK_COLOR, linewidth=1.5,
            label="Buy & hold NVDA")
    ax.plot(strategy_dd.index, strategy_dd * 100, color=STRATEGY_COLOR, linewidth=1.5, label="Estrategia")
    ax.set_ylim(top=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _split_line(ax, split_date, at_top=False)
    ax.set_title("Drawdown sobre el capital")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=False)
    _style(ax)
    return ax


def plot_cost_sensitivity(ax: plt.Axes, sensitivity: pd.DataFrame, assumed_bps: float,
                          break_even: Optional[dict] = None) -> plt.Axes:
    """Sharpe vs costo de ida y vuelta, una linea por periodo (train, test).

    Parametros
    ----------
    ax : plt.Axes
        Ejes donde dibujar.
    sensitivity : pd.DataFrame
        Salida de analysis.cost_sensitivity (indice bps, columnas sharpe_train
        y sharpe_test).
    assumed_bps : float
        Costo de ida y vuelta asumido en el SPEC (linea vertical).
    break_even : dict, opcional
        {periodo: bps de break-even}; se marca con una X en Sharpe = 0. Si es
        NaN, la leyenda indica que no cruza en el rango.

    Regresa
    -------
    plt.Axes
    """
    break_even = break_even or {}
    styles = {"train": "--", "test": "-"}
    bps = sensitivity.index

    for period, linestyle in styles.items():
        be = break_even.get(period, np.nan)
        be_text = f"break-even {be:.1f} bps" if np.isfinite(be) else "no cruza 0 en el rango"
        ax.plot(bps, sensitivity[f"sharpe_{period}"], color=STRATEGY_COLOR, linestyle=linestyle,
                linewidth=2, marker="o", markersize=5, label=f"{period.capitalize()} ({be_text})")
        if np.isfinite(be):
            ax.plot([be], [0], marker="X", markersize=11, color=STRATEGY_COLOR,
                    markeredgecolor="white", markeredgewidth=1.5, linestyle="none")

    ax.axhline(0, color="#3d3c39", linewidth=0.8)
    ax.axvline(assumed_bps, color=REFERENCE_COLOR, linestyle=":", linewidth=1.5)
    ax.annotate(f"costo asumido {assumed_bps:g} bps", xy=(assumed_bps, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=9, color="#3d3c39")
    ax.set_title("Sensibilidad del Sharpe al costo de transacción")
    ax.set_xlabel("Costo de ida y vuelta (bps)")
    ax.set_ylabel("Sharpe anualizado (adimensional)")
    ax.legend(loc="center left", frameon=False)
    _style(ax)
    return ax
