"""Reproduce los resultados de lab_02 (BTCUSDT 5 min, Nivel B).

Solo orquesta: toda la logica vive en src/. Este paso usa unicamente el
archivo de train (btc_project_train.csv):
1. Walk-forward completo (docs/SPEC.md, seccion 12).
2. Curvas fuera de muestra: con regimen, solo global y buy & hold.
3. Metricas, tablas de retornos, degradacion train -> OOS y θ_final.
4. θ_final global sobre todo el archivo de train (dentro de muestra).
5. Robustez pre-registrada (docs/SPEC.md, seccion 14), salidas de regimen
   del Nivel B y diagnostico de las perdidas.
6. Figuras en docs/figures y tablas en docs/tables.

Uso: python main.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # solo guarda archivos, no abre ventanas

import pandas as pd

from src.backtest import COMMISSION_RATE, BacktestResult, backtest, config_from_params
from src.data import load_train
from src.metrics import (buy_and_hold_equity, exit_reason_table, kruskal_by_regime, performance_summary,
                         pnl_breakdown, regime_trade_stats, returns_table, side_table)
from src.optimize import (CAPITAL, N_TRIALS, REGIME_NAMES, build_oos_inputs, cost_summary,
                          degradation_table, entry_regimes, oos_cost_curve, run_oos, run_walk_forward,
                          sensitivity_table, single_vote_table, theta_final, transitions_table, window_table)
from src.plots import (plot_correlation, plot_cost_curve, plot_drawdown, plot_feature_distributions,
                       plot_portfolio, plot_portfolio_regimes, plot_regime_timeline, plot_returns_table,
                       plot_sensitivity)
from src.regimes import (FIT_END, FIT_START, fit_regime_models, hourly, regime_features, regime_labels,
                         regime_validation)
from src.signals import compute_strategy, indicator_correlation

ROOT = Path(__file__).resolve().parent
# Periodos de la validacion de regimen (ajuste y fuera de muestra, dentro del archivo de train).
REGIME_PERIODS = {"ajuste": (FIT_START, FIT_END), "fuera_de_muestra": ("2023-05-15", "2023-12-31")}
# Nombre de cada curva OOS en los archivos.
FILE_NAMES = {"con régimen": "con_regimen", "solo global": "solo_global"}
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


def build_oos_curves(df: pd.DataFrame, results: list[dict]) -> tuple[dict, dict]:
    """Curvas OOS continuas (con regimen, solo global y buy & hold) y sus entradas por barra."""
    inputs = {"con régimen": build_oos_inputs(df, results, use_regimes=True),
              "solo global": build_oos_inputs(df, results, use_regimes=False)}
    curves = {name: run_oos(df, table) for name, table in inputs.items()}
    curves["buy & hold"] = buy_and_hold(df.loc[inputs["con régimen"].index])
    return curves, inputs


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
        _save_figure(fig, name)


def _save_figure(fig, name: str) -> None:
    """Guarda una figura en docs/figures/<name>.png."""
    fig.savefig(FIGURES / f"{name}.png", dpi=130)


def robustness_sensitivity(df: pd.DataFrame, theta: dict) -> None:
    """SPEC 14.1: sensibilidad ±20% de θ_final global en todo el archivo de train."""
    table = sensitivity_table(df, theta)
    _save(table, "sensitivity", index=False)
    _save_figure(plot_sensitivity(table), "sensitivity")


def robustness_costs(df: pd.DataFrame, inputs: dict) -> None:
    """SPEC 14.2: curva de costos con las mismas entradas por barra de cada curva OOS."""
    curves = {name: oos_cost_curve(df, table) for name, table in inputs.items()}
    flat = pd.concat({FILE_NAMES[name]: curve for name, curve in curves.items()}, axis=1)
    flat.columns = [f"{metric}_{name}" for name, metric in flat.columns]
    _save(flat, "cost_curve")
    summary = pd.DataFrame({FILE_NAMES[name]: cost_summary(curve) for name, curve in curves.items()})
    _save(summary, "cost_summary")
    _save_figure(plot_cost_curve(curves), "cost_curve")


def robustness_single_vote(df: pd.DataFrame, theta: dict) -> None:
    """SPEC 14.3: un solo indicador contra la regla 2 de 3."""
    _save(single_vote_table(df, theta), "single_vote")


def robustness_regime_trades(curves: dict, inputs: dict) -> None:
    """SPEC 14.4: trades OOS de la curva con regimen, por regimen de entrada, y Kruskal-Wallis."""
    trades = curves["con régimen"].trades
    regimes = entry_regimes(trades, inputs["con régimen"])
    _save(regime_trade_stats(trades, regimes), "regime_trades")
    _save(pd.Series(kruskal_by_regime(trades, regimes), name="value").to_frame(), "kruskal")


def robustness_correlation(df: pd.DataFrame, theta: dict) -> None:
    """SPEC 14.5: correlacion de votos e indicadores en velas de 4h (train)."""
    corr_votes, corr_indicators = indicator_correlation(df, theta)
    _save(corr_votes, "correlation_votes")
    _save(corr_indicators, "correlation_indicators")
    _save_figure(plot_correlation(corr_votes, corr_indicators), "correlation")


def regime_outputs(df: pd.DataFrame, results: list[dict], curves: dict, inputs: dict) -> None:
    """Salidas de regimen del Nivel B: validacion, linea de tiempo, distribuciones y transiciones."""
    models = fit_regime_models(df, FIT_START, FIT_END)
    _save(regime_validation(df, models, REGIME_PERIODS), "regime_validation")

    labels = hourly(regime_labels(df, models))
    features = hourly(regime_features(df))
    close = hourly(df["Close"].to_frame())["Close"]
    out_of_sample = pd.Timestamp(REGIME_PERIODS["fuera_de_muestra"][0])
    _save_figure(plot_regime_timeline(close, labels, out_of_sample), "regime_timeline")
    _save_figure(plot_feature_distributions(features, labels), "regime_features")

    regime_inputs = inputs["con régimen"]
    _save_figure(plot_portfolio_regimes(curves["con régimen"].equity["equity"], regime_inputs["regime"],
                                        curves["buy & hold"].equity["equity"],
                                        "Walk-forward OOS con régimen: portafolio y régimen vigente"),
                 "portfolio_regimes")
    _save(transitions_table(regime_inputs, curves["con régimen"].trades, results), "transitions")


def loss_diagnostics(curves: dict) -> None:
    """Por que pierde: salidas, largos contra cortos, PnL bruto contra costos y payoff."""
    strategies = {FILE_NAMES[name]: curves[name].trades for name in FILE_NAMES}
    _save(pd.concat({name: exit_reason_table(t) for name, t in strategies.items()}, names=["curve"]),
          "diag_exit_reason")
    _save(pd.concat({name: side_table(t) for name, t in strategies.items()}, names=["curve"]), "diag_side")
    _save(pd.DataFrame({name: pnl_breakdown(t) for name, t in strategies.items()}), "diag_pnl")


def run_robustness(df: pd.DataFrame, results: list[dict], curves: dict, inputs: dict, final: dict) -> None:
    """Robustez pre-registrada (SPEC 14), salidas de regimen y diagnostico de perdidas."""
    robustness_sensitivity(df, final["global"])
    robustness_costs(df, inputs)
    robustness_single_vote(df, final["global"])
    robustness_regime_trades(curves, inputs)
    robustness_correlation(df, final["global"])
    regime_outputs(df, results, curves, inputs)
    loss_diagnostics(curves)


def print_summary(results: list[dict]) -> None:
    """Imprime las tablas principales y cuantas ventanas usaron R5 por regimen."""
    pd.set_option("display.width", 160)
    for name in ["run_info", "metrics_oos", "metrics_train", "degradation_summary", "theta_final",
                 "sensitivity", "cost_summary", "single_vote", "regime_trades", "kruskal",
                 "correlation_votes", "correlation_indicators", "regime_validation", "transitions",
                 "diag_exit_reason", "diag_side", "diag_pnl"]:
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
    oos_curves, oos_inputs = build_oos_curves(df, results)
    save_metrics(oos_curves, "metrics_oos")
    returns = save_returns(oos_curves)
    save_degradation(results, oos_curves)
    final = save_theta_final(results)
    train_curves = run_train_analysis(df, final["global"])
    save_figures(oos_curves, train_curves, returns)
    run_robustness(df, results, oos_curves, oos_inputs, final)
    print_summary(results)


if __name__ == "__main__":
    main()
