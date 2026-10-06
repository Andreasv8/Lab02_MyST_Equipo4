"""Prueba de humo de las graficas de resultados: se arman y tienen titulo, ejes y leyenda (PDF 3.6)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import returns_table
from src.plots import (plot_correlation, plot_cost_curve, plot_drawdown, plot_feature_distributions,
                       plot_portfolio, plot_portfolio_regimes, plot_regime_timeline,
                       plot_returns_table, plot_sensitivity)

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


REGIMES = pd.Series(np.resize(["trend"] * 40 + ["crisis"] * 20 + ["mean_reversion"] * 40, len(INDEX)),
                    index=INDEX)


def test_robustness_figures_have_labels():
    sensitivity = pd.DataFrame({"param": ["sl_mult", "rr"], "calmar_minus_20": [0.1, -0.2],
                                "calmar_base": [0.2, 0.2], "calmar_plus_20": [0.3, 0.1],
                                "at_limit_minus_20": [False, True], "at_limit_plus_20": [False, False]})
    _check_axes(plot_sensitivity(sensitivity).axes[0])
    curve = pd.DataFrame({"total_return": [0.1, 0.0, -0.1]}, index=[0.0, 12.5, 25.0])
    _check_axes(plot_cost_curve({"con régimen": curve}).axes[0])
    corr = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    fig = plot_correlation(corr, corr)
    for ax in fig.axes[:2]:                              # los dos heatmaps (despues van las barras de color)
        _check_axes(ax, needs_legend=False)


def test_regime_figures_have_labels():
    labels = pd.DataFrame({"rules": REGIMES, "kmeans": REGIMES, "hmm": REGIMES})
    fig = plot_regime_timeline(EQUITY, labels, INDEX[len(INDEX) // 2])
    assert all(ax.get_title() and ax.get_ylabel() for ax in fig.axes)
    assert fig.axes[-1].get_xlabel()                     # eje x compartido: la fecha va abajo
    assert fig.axes[0].get_legend() is not None
    features = pd.DataFrame({"volatility": EQUITY.pct_change(), "trend_r2": 0.5, "autocorr_1": 0.0})
    fig = plot_feature_distributions(features, labels)
    assert fig.legends and fig.axes[0].get_xlabel()
    fig = plot_portfolio_regimes(EQUITY, REGIMES, BENCHMARK, "Portafolio con régimen")
    _check_axes(fig.axes[0])
