"""Graficas de lab_02 (matplotlib) para el reporte.

Cada funcion recibe los datos ya calculados y solo dibuja; las que arman una
figura completa la crean y la regresan para guardarla. Todas llevan titulo,
ejes con nombre y leyenda (PDF 3.6). Convenciones: un eje y por grafica,
colores fijos por curva y por regimen, rejilla tenue.
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
        ax.set_ylabel("Cierre BTCUSDT (USDT)")
    for ax in axes[1, :]:
        ax.set_xlabel("Fecha")
    handles = [Line2D([], [], color=c, linewidth=2.5, label=n) for n, c in REGIME_COLORS.items()]
    handles.append(Line2D([], [], color=WARMUP_COLOR, linewidth=2.5, label="warm-up (sin features)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Regimenes de BTCUSDT por metodo (train + test)")
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


# ---------------------------------------------------------------------------
# Resultados de la estrategia (PDF 3.6)
# ---------------------------------------------------------------------------

# Color fijo por curva; las que no esten aqui usan el azul de la estrategia.
CURVE_COLORS = {"con régimen": STRATEGY_COLOR, "solo global": "#e8a33d",
                "θ_final global": STRATEGY_COLOR, "buy & hold": BENCHMARK_COLOR}
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _usd(ax: plt.Axes) -> None:
    """Eje y en dolares con separador de miles."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))


def plot_portfolio(curves: dict, benchmark: pd.Series, title: str) -> plt.Figure:
    """Valor del portafolio en el tiempo de cada curva contra el benchmark.

    Recibe {nombre: equity por barra} y la equity del buy & hold (mismo periodo).
    Regresa la figura.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(benchmark.index, benchmark, color=BENCHMARK_COLOR, linewidth=1.5, label="buy & hold")
    for name, equity in curves.items():
        ax.plot(equity.index, equity, color=CURVE_COLORS.get(name, STRATEGY_COLOR), linewidth=1.8, label=name)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor del portafolio (USD)")
    _usd(ax)
    ax.legend(loc="upper left", frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_drawdown(curves: dict, title: str) -> plt.Figure:
    """Drawdown (caida desde el maximo previo, en %) de cada curva.

    Recibe {nombre: equity por barra}. Regresa la figura.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, equity in curves.items():
        drawdown = (equity / equity.cummax() - 1) * 100
        ax.plot(drawdown.index, drawdown, color=CURVE_COLORS.get(name, STRATEGY_COLOR),
                linewidth=1.4, label=name)
    ax.set_ylim(top=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def _monthly_grid(monthly: pd.Series) -> pd.DataFrame:
    """Retornos mensuales como tabla año x mes, en %."""
    grid = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month,
                         "ret": monthly.to_numpy() * 100})
    return grid.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))


def _bar_returns(ax: plt.Axes, values: pd.Series, labels: list, title: str) -> None:
    """Barras de retorno en %: verde si gana, rojo si pierde."""
    colors = ["#2e8b57" if v >= 0 else "#e34948" for v in values]
    ax.bar(labels, values * 100, color=colors, label="retorno del periodo")
    ax.axhline(0, color=REFERENCE_COLOR, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Retorno (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    _style(ax)


def plot_returns_table(monthly: pd.Series, quarterly: pd.Series, annual: pd.Series,
                       title: str) -> plt.Figure:
    """Tabla de retornos: heatmap mensual (año x mes, con el numero en cada celda),
    mas barras de retornos trimestrales y anuales.

    Recibe las salidas de metrics.returns_table. Regresa la figura.
    """
    grid = _monthly_grid(monthly)
    fig = plt.figure(figsize=(12, 7))
    ax_map = fig.add_subplot(2, 1, 1)
    limit = max(np.nanmax(np.abs(grid.to_numpy())), 1e-9)
    image = ax_map.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for i, year in enumerate(grid.index):
        for j in range(12):
            value = grid.iloc[i, j]
            if np.isfinite(value):
                ax_map.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=8)
    ax_map.set_xticks(range(12), MONTHS)
    ax_map.set_yticks(range(len(grid.index)), [str(y) for y in grid.index])
    ax_map.set_xlabel("Mes")
    ax_map.set_ylabel("Año")
    ax_map.set_title(f"{title}: retornos mensuales")
    fig.colorbar(image, ax=ax_map, label="Retorno mensual (%)")

    ax_q = fig.add_subplot(2, 2, 3)
    quarter_labels = [f"{d.year} T{(d.month - 1) // 3 + 1}" for d in quarterly.index]
    _bar_returns(ax_q, quarterly, quarter_labels, "Retornos trimestrales")
    ax_y = fig.add_subplot(2, 2, 4)
    _bar_returns(ax_y, annual, [str(d.year) for d in annual.index], "Retornos anuales")
    fig.tight_layout()
    return fig
