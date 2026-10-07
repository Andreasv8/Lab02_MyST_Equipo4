"""Graficas de lab_02 (matplotlib) para el reporte.

Cada funcion recibe los datos ya calculados y solo dibuja; las que arman una
figura completa la crean y la regresan para guardarla. Todas llevan titulo,
ejes con nombre y leyenda (PDF 3.6). Convenciones: un eje y por grafica,
colores fijos por curva y por regimen, rejilla tenue.
"""

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


def _regime_line(ax: plt.Axes, close: pd.Series, labels: pd.Series) -> None:
    """Cierre como segmentos coloreados por el regimen de la barra de llegada (warm-up en gris)."""
    x = mdates.date2num(close.index.to_pydatetime())
    points = np.column_stack([x, close.to_numpy()])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = [REGIME_COLORS.get(label, WARMUP_COLOR) for label in labels.iloc[1:]]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.4))
    ax.autoscale_view()
    ax.xaxis_date()


def _regime_handles(with_warmup: bool = False) -> list:
    """Entradas de leyenda con el color de cada regimen."""
    handles = [Line2D([], [], color=c, linewidth=6, label=n) for n, c in REGIME_COLORS.items()]
    if with_warmup:
        handles.append(Line2D([], [], color=WARMUP_COLOR, linewidth=6, label="sin regimen (calentamiento)"))
    return handles


def plot_regime_timeline(close: pd.Series, labels_df: pd.DataFrame, split_date: pd.Timestamp) -> plt.Figure:
    """Precio de cierre coloreado por regimen: reglas arriba (el metodo elegido), K-means y HMM abajo.

    Recibe el cierre por hora, las etiquetas por hora (columnas rules, kmeans,
    hmm) y la fecha donde empieza el periodo fuera de muestra (linea vertical).
    Regresa la figura.
    """
    titles = {"rules": "Reglas (metodo elegido)", "kmeans": "K-means (solo comparacion)",
              "hmm": "HMM filtrado (solo comparacion)"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    for ax, method in zip(axes, titles):
        _regime_line(ax, close, labels_df[method])
        _split_line(ax, split_date, label="inicio fuera de muestra")
        ax.set_title(titles[method])
        ax.set_ylabel("Cierre BTCUSDT (USDT)")
        _style(ax)
    axes[-1].set_xlabel("Fecha")
    axes[0].legend(handles=_regime_handles(with_warmup=True), loc="upper left", frameon=False)
    fig.suptitle("Regimenes de BTCUSDT por metodo (archivo de train)")
    fig.tight_layout()
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
            ax.set_xlabel("Régimen")
            ax.set_ylabel(units.get(col, col))
            _style(ax)
    fig.legend(handles=_regime_handles(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Distribucion de las variables de regimen por regimen y metodo")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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


# ---------------------------------------------------------------------------
# Robustez y salidas de regimen
# ---------------------------------------------------------------------------

def plot_sensitivity(table: pd.DataFrame) -> plt.Figure:
    """Calmar con cada parametro en -20%, base y +20% (barras agrupadas).

    Recibe la salida de optimize.sensitivity_table. Las barras que quedaron
    "en el limite" se marcan con un asterisco.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(table))
    width = 0.27
    bars = [("calmar_minus_20", "-20%", "#e8a33d", "at_limit_minus_20"),
            ("calmar_base", "base (θ_final)", STRATEGY_COLOR, None),
            ("calmar_plus_20", "+20%", "#2e8b57", "at_limit_plus_20")]
    for k, (col, label, color, limit_col) in enumerate(bars):
        positions = x + (k - 1) * width
        ax.bar(positions, table[col], width, color=color, label=label)
        if limit_col is not None:
            for pos, value, at_limit in zip(positions, table[col], table[limit_col]):
                if at_limit:
                    ax.annotate("*", (pos, value), ha="center", va="bottom", fontsize=12)
    ax.axhline(0, color=REFERENCE_COLOR, linewidth=0.8)
    ax.set_xticks(x, table["param"], rotation=30)
    ax.set_title("Sensibilidad ±20% de θ_final global (Calmar en todo el archivo de train; * = en el limite)")
    ax.set_xlabel("Parametro movido (uno a la vez)")
    ax.set_ylabel("Calmar")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_cost_curve(curves: dict, real_bps: float = 12.5) -> plt.Figure:
    """Retorno neto de la curva OOS contra la comision por lado, una linea por curva.

    Recibe {nombre: salida de optimize.oos_cost_curve}. Marca la comision
    real con una linea vertical y el retorno 0 con una horizontal.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in curves.items():
        ax.plot(curve.index, curve["total_return"] * 100, marker="o", markersize=3, linewidth=1.8,
                color=CURVE_COLORS.get(name, STRATEGY_COLOR), label=name)
    ax.axvline(real_bps, color=REFERENCE_COLOR, linestyle="--", linewidth=1,
               label=f"comision real ({real_bps} pb por lado)")
    ax.axhline(0, color="#1f1e1c", linewidth=0.8)
    ax.set_title("Retorno neto OOS contra costo de transaccion (mismas entradas, solo cambia la comision)")
    ax.set_xlabel("Comision por lado (pb)")
    ax.set_ylabel("Retorno neto total (%)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_correlation(corr_votes: pd.DataFrame, corr_indicators: pd.DataFrame) -> plt.Figure:
    """Dos heatmaps con numeros: correlacion de los votos y de los indicadores (velas de 4h)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, corr, title in [(axes[0], corr_votes, "Votos"), (axes[1], corr_indicators, "Indicadores")]:
        image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=30)
        ax.set_yticks(range(len(corr)), corr.index)
        ax.set_title(f"Correlacion entre {title.lower()} (4h, train)")
        ax.set_xlabel(title)
        ax.set_ylabel(title)
        fig.colorbar(image, ax=ax, label="Correlacion de Pearson")
    fig.tight_layout()
    return fig


def _regime_runs(regimes: pd.Series) -> list[tuple]:
    """Rachas consecutivas de un mismo regimen: (inicio, fin, regimen)."""
    clean = regimes.dropna()
    run_id = (clean != clean.shift()).cumsum()
    return [(group.index[0], group.index[-1], group.iloc[0]) for _, group in clean.groupby(run_id)]


def plot_portfolio_regimes(equity: pd.Series, regimes: pd.Series, benchmark: pd.Series,
                           title: str) -> plt.Figure:
    """Valor del portafolio OOS con el regimen de cada momento como franja de color de fondo."""
    fig, ax = plt.subplots(figsize=(13, 5))
    for start, end, regime in _regime_runs(regimes):
        ax.axvspan(start, end, color=REGIME_COLORS[regime], alpha=0.15, linewidth=0)
    ax.plot(benchmark.index, benchmark, color=BENCHMARK_COLOR, linewidth=1.2, label="buy & hold")
    ax.plot(equity.index, equity, color="#1f1e1c", linewidth=1.8, label="con régimen")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor del portafolio (USD)")
    _usd(ax)
    curve_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=curve_handles + _regime_handles(), loc="upper left", frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_portfolio_panels(panels: dict) -> plt.Figure:
    """Valor del portafolio en varios periodos, un panel por periodo, cada uno con buy & hold.

    Recibe {titulo del panel: (curvas {nombre: equity}, equity del buy & hold)}.
    Regresa la figura.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
    for ax, (title, (curves, benchmark)) in zip(axes[0], panels.items()):
        ax.plot(benchmark.index, benchmark, color=BENCHMARK_COLOR, linewidth=1.5, label="buy & hold")
        for name, equity in curves.items():
            ax.plot(equity.index, equity, color=CURVE_COLORS.get(name, STRATEGY_COLOR), linewidth=1.8, label=name)
        ax.set_title(title)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor del portafolio (USD)")
        ax.tick_params(axis="x", rotation=30)
        _usd(ax)
        ax.legend(loc="upper left", frameon=False)
        _style(ax)
    fig.tight_layout()
    return fig
