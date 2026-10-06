"""Reproduce los resultados de lab_02 (BTCUSDT 5 min, Nivel B).

Solo orquesta: toda la logica vive en src/. Este paso usa unicamente el
archivo de train (btc_project_train.csv):
1. Walk-forward completo (docs/SPEC.md, seccion 12).
2. Curvas fuera de muestra: con regimen, solo global y buy & hold.
3. Metricas, tablas de retornos, degradacion train -> OOS y θ_final.
4. θ_final global sobre todo el archivo de train (dentro de muestra).
5. Figuras en docs/figures y tablas en docs/tables.

Uso: python main.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # solo guarda archivos, no abre ventanas

import pandas as pd

from src.backtest import COMMISSION_RATE, BacktestResult, backtest, config_from_params
from src.data import load_train
from src.metrics import buy_and_hold_equity, performance_summary, returns_table
from src.optimize import (CAPITAL, N_TRIALS, REGIME_NAMES, degradation_table, run_oos_both,
                          run_walk_forward, theta_final, window_table)
from src.plots import plot_drawdown, plot_portfolio, plot_returns_table
from src.signals import compute_strategy

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "docs" / "tables"
FIGURES = ROOT / "docs" / "figures"


def _save(table: pd.DataFrame, name: str, index: bool = True) -> None:
    """Guarda una tabla en docs/tables/<name>.csv."""
    table.to_csv(TABLES / f"{name}.csv", index=index)


def buy_and_hold(df: pd.DataFrame) -> BacktestResult:
    """Buy & hold en las barras de df, con la comision de entrada y sin trades."""
    equity = buy_and_hold_equity(df, CAPITAL, COMMISSION_RATE)
    return BacktestResult(equity=equity, trades=pd.DataFrame(columns=["pnl"]))


def run_walk_forward_step(df: pd.DataFrame) -> list[dict]:
    """Corre el walk-forward y guarda la tabla por ventana y el tiempo de corrida."""
    results, seconds, n_configs = run_walk_forward(df, n_trials=N_TRIALS)
    table = window_table(results)
    _save(table, "walk_forward_windows", index=False)
    run_info = pd.Series({
        "windows": len(results),
        "trials_per_study": N_TRIALS,
        "configurations": n_configs,
        "seconds": round(seconds, 1),
        "cpu_cores": os.cpu_count(),
    }, name="value")
    _save(run_info.to_frame(), "run_info")
    return results


def build_oos_curves(df: pd.DataFrame, results: list[dict]) -> dict:
    """Curvas OOS continuas: con regimen, solo global y buy & hold en las mismas barras."""
    oos = run_oos_both(df, results)
    curves = {"con régimen": oos["regimen"], "solo global": oos["global"]}
    curves["buy & hold"] = buy_and_hold(df.loc[oos["regimen"].equity.index])
    return curves


def save_metrics(curves: dict, name: str) -> pd.DataFrame:
    """performance_summary de cada curva -> docs/tables/<name>.csv."""
    table = pd.DataFrame({label: performance_summary(r.equity, r.trades) for label, r in curves.items()})
    _save(table, name)
    return table


def save_returns(curves: dict) -> dict:
    """Retornos mensuales, trimestrales y anuales de cada curva (una columna por curva)."""
    tables = {}
    for freq, name in [("ME", "monthly"), ("QE", "quarterly"), ("YE", "annual")]:
        tables[name] = pd.DataFrame({label: returns_table(r.equity["equity"], freq)
                                     for label, r in curves.items()})
        _save(tables[name], f"returns_{name}")
    return tables


def save_degradation(results: list[dict], curves: dict) -> pd.DataFrame:
    """Degradacion train -> OOS por ventana y su resumen."""
    oos = {"con_regimen": curves["con régimen"], "solo_global": curves["solo global"]}
    by_window, summary = degradation_table(results, oos)
    _save(by_window, "degradation")
    _save(summary, "degradation_summary", index=False)
    return summary


def save_theta_final(results: list[dict]) -> dict:
    """θ_final de cada estudio (docs/SPEC.md, seccion 13) -> theta_final.csv."""
    final = theta_final(results)
    table = pd.DataFrame({name: theta for name, theta in final.items() if theta is not None}).T
    _save(table.rename_axis("study"), "theta_final")
    return final


def run_train_analysis(df: pd.DataFrame, theta: dict) -> dict:
    """θ_final global sobre todo el archivo de train, contra buy & hold.

    Es DENTRO de muestra: θ_final sale de las ventanas de este mismo archivo.
    """
    features = compute_strategy(df, theta)
    strategy = backtest(df, features["signal"], features["atr"], config_from_params(theta, CAPITAL))
    curves = {"θ_final global": strategy, "buy & hold": buy_and_hold(df)}
    save_metrics(curves, "metrics_train")
    return curves


def save_figures(oos_curves: dict, train_curves: dict, returns: dict) -> None:
    """Figuras del reporte en docs/figures."""
    strategies = {k: v.equity["equity"] for k, v in oos_curves.items() if k != "buy & hold"}
    benchmark = oos_curves["buy & hold"].equity["equity"]
    figures = {
        "portfolio_oos": plot_portfolio(strategies, benchmark,
                                        "Walk-forward fuera de muestra: valor del portafolio"),
        "drawdown_oos": plot_drawdown({**strategies, "buy & hold": benchmark},
                                      "Walk-forward fuera de muestra: drawdown"),
        "portfolio_train": plot_portfolio(
            {"θ_final global": train_curves["θ_final global"].equity["equity"]},
            train_curves["buy & hold"].equity["equity"],
            "θ_final global en todo el archivo de train (dentro de muestra)"),
        "returns_oos": plot_returns_table(returns["monthly"]["con régimen"], returns["quarterly"]["con régimen"],
                                          returns["annual"]["con régimen"],
                                          "Walk-forward OOS con régimen"),
    }
    for name, fig in figures.items():
        fig.savefig(FIGURES / f"{name}.png", dpi=130)


def print_summary(results: list[dict]) -> None:
    """Imprime las tablas principales y cuantas ventanas usaron R5 por regimen."""
    pd.set_option("display.width", 160)
    for name in ["run_info", "metrics_oos", "metrics_train", "degradation_summary", "theta_final"]:
        print(f"\n=== {name} ===")
        print(pd.read_csv(TABLES / f"{name}.csv", index_col=0).to_string())
    table = window_table(results)
    r5 = table[table["study"].isin(REGIME_NAMES)].groupby("study")["used_r5"].sum()
    print("\n=== ventanas que usaron R5, por regimen ===")
    print(r5.to_string())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_train()

    results = run_walk_forward_step(df)
    oos_curves = build_oos_curves(df, results)
    save_metrics(oos_curves, "metrics_oos")
    returns = save_returns(oos_curves)
    save_degradation(results, oos_curves)
    final = save_theta_final(results)
    train_curves = run_train_analysis(df, final["global"])
    save_figures(oos_curves, train_curves, returns)
    print_summary(results)


if __name__ == "__main__":
    main()
