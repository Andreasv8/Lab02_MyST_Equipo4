"""Prueba de humo de las graficas de resultados: se arman y tienen titulo, ejes y leyenda (PDF 3.6)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import returns_table
from src.plots import plot_drawdown, plot_portfolio, plot_returns_table

INDEX = pd.date_range("2022-08-01", "2023-12-24", freq="6h")
RNG = np.random.default_rng(42)
EQUITY = pd.Series(1_000_000 * np.exp(np.cumsum(RNG.normal(0, 0.01, len(INDEX)))), index=INDEX)
BENCHMARK = pd.Series(1_000_000 * np.exp(np.cumsum(RNG.normal(0, 0.01, len(INDEX)))), index=INDEX)


def _check_axes(ax, needs_legend: bool = True) -> None:
    assert ax.get_title()
    assert ax.get_xlabel() and ax.get_ylabel()
    if needs_legend:
        assert ax.get_legend() is not None


def test_portfolio_and_drawdown_have_labels():
    fig = plot_portfolio({"con régimen": EQUITY}, BENCHMARK, "Portafolio")
    _check_axes(fig.axes[0])
    fig = plot_drawdown({"con régimen": EQUITY, "buy & hold": BENCHMARK}, "Drawdown")
    _check_axes(fig.axes[0])


def test_returns_table_figure_has_labels():
    fig = plot_returns_table(returns_table(EQUITY, "ME"), returns_table(EQUITY, "QE"),
                             returns_table(EQUITY, "YE"), "Retornos")
    heatmap, quarterly, annual = fig.axes[0], fig.axes[2], fig.axes[3]   # axes[1] es la barra de color
    _check_axes(heatmap, needs_legend=False)       # el heatmap usa barra de color con etiqueta
    _check_axes(quarterly)
    _check_axes(annual)
    assert len(heatmap.texts) == 17                # un numero por mes (ago 2022 a dic 2023)
