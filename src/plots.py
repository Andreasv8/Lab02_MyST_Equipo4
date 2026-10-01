"""Graficas de lab_02 (matplotlib) para los notebooks de las Act 06 y 07.

Cada funcion recibe los datos ya calculados y un Axes, y solo dibuja. Las
graficas de varios paneles de regimenes (Act 07) crean y regresan su Figure.
Convenciones: un eje y por grafica, estrategia en azul y buy & hold en gris,
regimenes con color fijo por nombre, rejilla tenue, titulos y ejes con unidades.
"""

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

STRATEGY_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#52514e"
REFERENCE_COLOR = "#9a9893"
REGIME_COLORS = {"crisis": "#e34948", "trend": "#2a78d6", "mean_reversion": "#52514e"}
WARMUP_COLOR = "#d9d8d4"
REGIME_PANEL_TITLES = {
    "rules": "Reglas (filtrado)",
    "kmeans": "K-means (filtrado)",
    "hmm": "HMM filtrado (forward)",
    "hmm_viterbi": "HMM Viterbi (usa el futuro, solo comparacion)",
}


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


def plot_elbow(ax: plt.Axes, elbow: pd.Series, chosen_k: int = 3) -> plt.Axes:
    """Curva del codo de K-means: inercia en train contra k, marcando el k elegido.

    Parametros
    ----------
    ax : plt.Axes
        Ejes donde dibujar.
    elbow : pd.Series
        Inercia indexada por k (salida de regimes.kmeans_elbow).
    chosen_k : int
        k usado en el lab (linea vertical).

    Regresa
    -------
    plt.Axes
    """
    ax.plot(elbow.index, elbow.to_numpy(), color=STRATEGY_COLOR, marker="o", linewidth=2)
    ax.axvline(chosen_k, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
    ax.annotate(f"k = {chosen_k}", xy=(chosen_k, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=9, color="#3d3c39")
    ax.set_xticks(list(elbow.index))
    ax.set_title("Curva del codo de K-means (train)")
    ax.set_xlabel("Numero de clusters k")
    ax.set_ylabel("Inercia (features escaladas)")
    _style(ax)
    return ax


def _regime_line(ax: plt.Axes, close: pd.Series, labels: pd.Series) -> None:
    """Cierre como segmentos coloreados por el regimen de la barra de llegada (warm-up en gris)."""
    x = mdates.date2num(close.index.to_pydatetime())
    points = np.column_stack([x, close.to_numpy()])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = [REGIME_COLORS.get(label, WARMUP_COLOR) for label in labels.iloc[1:]]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.4))
    ax.autoscale_view()
    ax.xaxis_date()


def plot_regime_timeline(close: pd.Series, labels_df: pd.DataFrame,
                         split_date: pd.Timestamp) -> plt.Figure:
    """Precio de cierre coloreado por regimen: un panel por metodo filtrado + Viterbi.

    Rejilla 2x2 con eje x compartido: reglas, K-means, HMM filtrado y HMM
    Viterbi (este ultimo usa el futuro; se muestra solo para comparar).

    Parametros
    ----------
    close : pd.Series
        Cierre por barra (USD).
    labels_df : pd.DataFrame
        Salida de regime_labels (columnas rules, kmeans, hmm, hmm_viterbi).
    split_date : pd.Timestamp
        Primer dia de test (linea vertical).

    Regresa
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, REGIME_PANEL_TITLES):
        _regime_line(ax, close, labels_df[method])
        _split_line(ax, split_date)
        ax.set_title(REGIME_PANEL_TITLES[method])
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cierre NVDA (USD)")
    for ax in axes[1, :]:
        ax.set_xlabel("Fecha")
    handles = [Line2D([], [], color=c, linewidth=2.5, label=n) for n, c in REGIME_COLORS.items()]
    handles.append(Line2D([], [], color=WARMUP_COLOR, linewidth=2.5, label="warm-up (sin features)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Regimenes de NVDA por metodo (train + test)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_feature_distributions(features: pd.DataFrame, labels_df: pd.DataFrame,
                               methods: tuple = ("rules", "kmeans", "hmm")) -> plt.Figure:
    """Boxplots de cada feature por regimen; una fila por metodo, una columna por feature.

    Parametros
    ----------
    features : pd.DataFrame
        Features en unidades originales (volatility, trend_r2, autocorr_1).
    labels_df : pd.DataFrame
        Etiquetas por metodo, mismo indice.
    methods : tuple
        Metodos (filas) a graficar.

    Regresa
    -------
    plt.Figure
    """
    regimes = list(REGIME_COLORS)
    units = {"volatility": "volatilidad anualizada", "trend_r2": "R^2 (0-1)",
             "autocorr_1": "autocorrelacion lag-1"}
    fig, axes = plt.subplots(len(methods), len(features.columns),
                             figsize=(13, 3.2 * len(methods)), squeeze=False)
    for i, method in enumerate(methods):
        for j, col in enumerate(features.columns):
            ax = axes[i, j]
            data = [features.loc[labels_df[method] == r, col].dropna() for r in regimes]
            box = ax.boxplot(data, patch_artist=True, widths=0.6,
                             medianprops={"color": "white", "linewidth": 1.5},
                             flierprops={"markersize": 2, "alpha": 0.4})
            for patch, regime in zip(box["boxes"], regimes):
                patch.set_facecolor(REGIME_COLORS[regime])
                patch.set_edgecolor(REGIME_COLORS[regime])
            ax.set_xticks(range(1, len(regimes) + 1), regimes)
            ax.set_title(f"{REGIME_PANEL_TITLES[method]} - {col}", fontsize=10)
            ax.set_ylabel(units.get(col, col))
            _style(ax)
    fig.tight_layout()
    return fig
