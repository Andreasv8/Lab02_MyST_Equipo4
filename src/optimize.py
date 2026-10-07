"""Optimizacion por ventana y walk-forward de lab_02 (PDF 3.2, 3.3 y 3.4).

En cada ventana se entrena 1 mes y se prueba la semana siguiente:
1. Regimenes por reglas con umbral ajustado hasta el fin del train (rule_regimes).
2. Cuatro estudios de Optuna: "global" y uno por regimen. El objetivo es el
   Calmar del train.
3. La semana de test se opera con los parametros de su ventana y del regimen de
   cada barra (reglas de transicion R1-R5 en docs/SPEC.md).

Todas las semanas de test se juntan en una sola curva fuera de muestra (OOS)
con un backtest continuo.
"""

import math
import time
from dataclasses import replace
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd

from src.backtest import COMMISSION_RATE, BacktestResult, backtest, config_from_params
from src.data import TEST_END, TEST_START
from src.metrics import break_even_cost, calmar_ratio
from src.regimes import HOURS_PER_MONTH, REGIME_NAMES, hourly, rule_regimes
from src.signals import THETA0, compute_strategy

SEED = 42
N_TRIALS = 100                    # por estudio (PDF: 100 a 200 por ventana y por regimen)
N_STARTUP_TRIALS = 30             # primero aleatorio, despues TPE

# Minimo de operaciones para que un trial cuente. Son bajos porque en 1 mes de
# train hay ~15 señales; es una limitacion que se declara en el reporte.
MIN_TRADES_GLOBAL = 5
MIN_TRADES_REGIME = 3
INVALID_SCORE = -1e6              # valor de un trial invalido

FIRST_TRAIN_START = "2022-07-01"  # el primer mes de datos sirve de calentamiento
WARMUP = pd.Timedelta(days=60)    # historia previa para calcular indicadores
BAR = pd.Timedelta(minutes=5)
CAPITAL = 1_000_000.0
STUDIES = ["global", *REGIME_NAMES]
PARAM_COLUMNS = ["sl_mult", "tp_mult", "max_holding", "rho"]

# Parametros del espacio de busqueda (los que se optimizan) y cuales son enteros.
SEARCH_PARAMS = ["ema_fast", "slow_ratio", "roc_window", "bb_window", "bb_threshold",
                 "adx_threshold", "sl_mult", "rr", "max_holding", "rho"]
INT_PARAMS = ["ema_fast", "roc_window", "bb_window", "max_holding"]
FIXED_PARAMS = {"bb_std": 2, "adx_window": 14, "atr_window": 14}
# Limites del espacio de busqueda (los mismos de suggest_params; un test lo revisa).
SEARCH_BOUNDS = {
    "ema_fast": (5, 30), "slow_ratio": (2.0, 6.0), "roc_window": (6, 42), "bb_window": (10, 40),
    "bb_threshold": (0.55, 0.90), "adx_threshold": (15.0, 35.0), "sl_mult": (1.0, 4.0),
    "rr": (1.0, 8.0), "max_holding": (288, 4032), "rho": (0.005, 0.02),
}
REAL_COST_BPS = 12.5             # comision real por lado: 0.125% = 12.5 pb


# ---------------------------------------------------------------------------
# 1. Ventanas (PDF 3.3)
# ---------------------------------------------------------------------------

def make_windows(index: pd.DatetimeIndex, first_train_start: str = FIRST_TRAIN_START,
                 train_months: int = 1, test_days: int = 7) -> list[dict]:
    """Arma las ventanas del walk-forward: train de 1 mes y test de la semana siguiente.

    Las semanas de test avanzan 7 dias exactos y no se traslapan. Cada train es
    el mes justo antes de su semana de test (train_start = test_start - 1 mes),
    asi que el train dura de 28 a 31 dias segun el mes. Todos los intervalos
    son [inicio, fin): la semana de test empieza donde termina el train.
    Se queda la ultima ventana cuya semana de test termina dentro del archivo.

    Recibe el indice de las barras de 5 min y la fecha de inicio del primer train.
    Regresa una lista de dicts con train_start, train_end, test_start y test_end.
    """
    end_of_data = index[-1] + BAR
    test_start = pd.Timestamp(first_train_start) + pd.DateOffset(months=train_months)
    windows = []
    while test_start + pd.Timedelta(days=test_days) <= end_of_data:
        windows.append({
            "train_start": test_start - pd.DateOffset(months=train_months),
            "train_end": test_start,
            "test_start": test_start,
            "test_end": test_start + pd.Timedelta(days=test_days),
        })
        test_start += pd.Timedelta(days=test_days)
    return windows


# ---------------------------------------------------------------------------
# 2. Optimizacion de una ventana (PDF 3.2)
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """Pide a Optuna un juego de parametros θ dentro del espacio de busqueda.

    Busca ventanas, umbrales, SL, TP y tamaño; bb_std, adx_window y atr_window
    quedan fijos. La EMA lenta se define como multiplo de la rapida para que
    siempre sea mas lenta.
    Recibe el trial. Regresa un dict con las mismas llaves que THETA0 (mas slow_ratio).
    """
    ema_fast = trial.suggest_int("ema_fast", 5, 30)
    slow_ratio = trial.suggest_float("slow_ratio", 2.0, 6.0)
    return {
        "ema_fast": ema_fast,
        "ema_slow": int(round(ema_fast * slow_ratio)),
        "slow_ratio": slow_ratio,
        "roc_window": trial.suggest_int("roc_window", 6, 42),
        "bb_window": trial.suggest_int("bb_window", 10, 40),
        "bb_std": 2,
        "bb_threshold": trial.suggest_float("bb_threshold", 0.55, 0.90),
        "adx_window": 14,
        "adx_threshold": trial.suggest_float("adx_threshold", 15.0, 35.0),
        "atr_window": 14,
        "sl_mult": trial.suggest_float("sl_mult", 1.0, 4.0),
        "rr": trial.suggest_float("rr", 1.0, 8.0),
        "max_holding": trial.suggest_int("max_holding", 288, 4032),   # 1 a 14 dias
        "rho": trial.suggest_float("rho", 0.005, 0.02),
    }


def train_score(df_warm: pd.DataFrame, train_index: pd.DatetimeIndex, labels: pd.Series,
                params: dict, regime: Optional[str]) -> tuple[float, int, float]:
    """Calmar, numero de trades y retorno total de θ en las barras de train.

    Los indicadores se calculan con el calentamiento (df_warm) y el backtest
    solo corre en las barras de train. Si regime no es None, solo se permiten
    entradas en barras de señal de ese regimen.
    Regresa (calmar, n_trades, retorno total como fraccion).
    """
    features = compute_strategy(df_warm, params).loc[train_index]
    signal = features["signal"]
    if regime is not None:
        signal = signal.where(labels.loc[train_index] == regime, 0)
    result = backtest(df_warm.loc[train_index], signal, features["atr"],
                      config_from_params(params, CAPITAL))
    equity = result.equity["equity"]
    return calmar_ratio(equity), len(result.trades), equity.iloc[-1] / equity.iloc[0] - 1


def run_study(df_warm: pd.DataFrame, train_index: pd.DatetimeIndex, labels: pd.Series,
              regime: Optional[str], n_trials: int, seed: int) -> Optional[dict]:
    """Un estudio de Optuna que maximiza el Calmar de train.

    Un trial con menos trades que el minimo, o con Calmar no finito, es
    invalido y recibe INVALID_SCORE.
    Regresa el mejor trial valido como {"params", "calmar", "n_trades", "total_return"}, o None.
    """
    min_trades = MIN_TRADES_GLOBAL if regime is None else MIN_TRADES_REGIME

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        calmar, n_trades, total_return = train_score(df_warm, train_index, labels, params, regime)
        trial.set_user_attr("params", params)
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("total_return", total_return)
        if n_trades < min_trades or not np.isfinite(calmar):
            return INVALID_SCORE
        return calmar

    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    if best.value <= INVALID_SCORE:
        return None
    return {"params": best.user_attrs["params"], "calmar": best.value,
            "n_trades": best.user_attrs["n_trades"], "total_return": best.user_attrs["total_return"]}


def optimize_window(df: pd.DataFrame, window: dict, number: int, n_trials: int = N_TRIALS) -> dict:
    """Corre los 4 estudios (global y uno por regimen) de una ventana.

    Solo usa datos anteriores a train_end, asi que no hay look-ahead. Los
    regimenes se ajustan con ventana expansiva hasta la ultima barra de train.
    Recibe todos los precios, la ventana, su numero (para la semilla) y los trials por estudio.
    Regresa un dict con la ventana, el mejor trial de cada estudio y los trials corridos.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # corre dentro de joblib
    df_fit = df[df.index < window["train_end"]]
    labels, _ = rule_regimes(df_fit, df_fit.index[-1])
    df_warm = df_fit[df_fit.index >= window["train_start"] - WARMUP]
    train_index = df_warm.index[df_warm.index >= window["train_start"]]

    best = {}
    for name in STUDIES:
        regime = None if name == "global" else name
        best[name] = run_study(df_warm, train_index, labels, regime, n_trials, SEED + number)
    return {"number": number, **window, "best": best, "n_trials": len(STUDIES) * n_trials}


def select_params(best: dict) -> tuple[dict, dict]:
    """Elige los parametros de cada estudio aplicando la regla R5.

    Un regimen sin trial valido usa los parametros globales de la ventana. Si
    tampoco el global tiene trial valido, la semana no se opera (todo None).
    Recibe {estudio: mejor trial o None}.
    Regresa ({estudio: params o None}, {regimen: True si uso R5}).
    """
    if best["global"] is None:
        return {name: None for name in STUDIES}, {regime: False for regime in REGIME_NAMES}

    global_params = best["global"]["params"]
    params, used_r5 = {"global": global_params}, {}
    for regime in REGIME_NAMES:
        used_r5[regime] = best[regime] is None
        params[regime] = global_params if used_r5[regime] else best[regime]["params"]
    return params, used_r5


def run_walk_forward(df: pd.DataFrame, n_trials: int = N_TRIALS, n_jobs: int = -1,
                     windows: Optional[list[dict]] = None) -> tuple[list[dict], float, int]:
    """Optimiza todas las ventanas en paralelo con joblib.

    Recibe los precios, los trials por estudio, los procesos (-1 = todos los
    nucleos) y, opcional, las ventanas (por defecto make_windows).
    Regresa (resultado por ventana, segundos que tardo, configuraciones evaluadas).
    """
    windows = make_windows(df.index) if windows is None else windows
    start = time.perf_counter()
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(optimize_window)(df, window, number, n_trials)
        for number, window in enumerate(windows))
    seconds = time.perf_counter() - start
    n_configs = sum(result["n_trials"] for result in results)
    return results, seconds, n_configs


def window_table(results: list[dict]) -> pd.DataFrame:
    """Tabla con una fila por (ventana, estudio).

    Columnas: fechas, si el estudio tuvo trial valido, Calmar y trades de
    train del mejor trial, si se uso R5, si la semana se opera y los
    parametros elegidos.
    """
    rows = []
    for result in results:
        params, used_r5 = select_params(result["best"])
        for name in STUDIES:
            best = result["best"][name]
            rows.append({
                "window": result["number"],
                "train_start": result["train_start"],
                "test_start": result["test_start"],
                "study": name,
                "valid": best is not None,
                "calmar_train": best["calmar"] if best else np.nan,
                "n_trades_train": best["n_trades"] if best else 0,
                "used_r5": used_r5.get(name, False),
                "operated": params["global"] is not None,
                **(params[name] or {}),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Curva fuera de muestra (OOS)
# ---------------------------------------------------------------------------

def crisis_entries(labels: pd.Series) -> pd.Series:
    """True solo en las barras donde el regimen CAMBIA a crisis (regla R3)."""
    is_crisis = labels == "crisis"
    return is_crisis & ~is_crisis.shift(1, fill_value=False)


def _bar_params(params: dict) -> dict:
    """Parametros de salida y tamaño que el motor usa por barra (tp_mult = rr · sl_mult)."""
    return {"sl_mult": params["sl_mult"], "tp_mult": params["rr"] * params["sl_mult"],
            "max_holding": params["max_holding"], "rho": params["rho"]}


def empty_inputs(regimes: pd.Series) -> pd.DataFrame:
    """Entradas del motor sin operar: señal 0, ATR NaN y THETA0 como relleno."""
    inputs = pd.DataFrame({"regime": regimes, "signal": 0, "atr": np.nan}, index=regimes.index)
    for col, value in _bar_params(THETA0).items():
        inputs[col] = value
    return inputs


def fill_inputs(inputs: pd.DataFrame, bars: pd.Index, features: pd.DataFrame,
                params: dict) -> pd.DataFrame:
    """Pone en las barras dadas la señal y el ATR de features y los parametros de params.

    Regresa una copia de inputs con esas barras llenas.
    """
    out = inputs.copy()
    out.loc[bars, "signal"] = features.loc[bars, "signal"]
    out.loc[bars, "atr"] = features.loc[bars, "atr"]
    for col, value in _bar_params(params).items():
        out.loc[bars, col] = value
    return out


def week_inputs(df: pd.DataFrame, result: dict, use_regimes: bool = True) -> pd.DataFrame:
    """Señal, ATR y parametros por barra de la semana de test de una ventana.

    Con use_regimes, cada barra usa los parametros del regimen vigente en ella
    (con R5 ya aplicado); sin regimen (calentamiento) no se opera. Sin
    use_regimes, toda la semana usa los parametros globales.
    Los regimenes usan el mismo umbral que en el ajuste (fit hasta la ultima
    barra de train), y los indicadores se calculan con 60 dias de calentamiento.
    """
    df_hist = df[df.index < result["test_end"]]
    last_train_bar = df_hist.index[df_hist.index < result["train_end"]][-1]
    labels, _ = rule_regimes(df_hist, last_train_bar)
    df_warm = df_hist[df_hist.index >= result["test_start"] - WARMUP]
    week_index = df_hist.index[df_hist.index >= result["test_start"]]

    inputs = empty_inputs(labels.loc[week_index])
    params, _ = select_params(result["best"])
    for name in (REGIME_NAMES if use_regimes else ["global"]):
        if params[name] is None:
            continue                     # la semana no se opera
        features = compute_strategy(df_warm, params[name]).loc[week_index]
        bars = week_index if name == "global" else week_index[inputs["regime"] == name]
        inputs = fill_inputs(inputs, bars, features, params[name])
    return inputs


def build_oos_inputs(df: pd.DataFrame, results: list[dict], use_regimes: bool = True) -> pd.DataFrame:
    """Junta las semanas de test en una sola tabla por barra para el backtest OOS.

    force_exit es True solo donde el regimen cambia a crisis (R3), y solo
    cuando se usa la capa de regimen.
    """
    inputs = pd.concat([week_inputs(df, result, use_regimes) for result in results])
    inputs["force_exit"] = crisis_entries(inputs["regime"]) if use_regimes else False
    return inputs


def run_oos(df: pd.DataFrame, inputs: pd.DataFrame, capital: float = CAPITAL,
            commission_rate: float = COMMISSION_RATE) -> BacktestResult:
    """Un solo backtest continuo sobre todas las semanas de test, con parametros por barra.

    Las posiciones pueden cruzar de una semana a otra y conservan los
    parametros con los que entraron (R1). commission_rate es la comision por
    lado (se cambia solo en la curva de costos).
    """
    # Los escalares de θ no se usan: sl, tp, holding y rho van por barra.
    config = replace(config_from_params(THETA0, capital), commission_rate=commission_rate)
    return backtest(df.loc[inputs.index], inputs["signal"], inputs["atr"], config,
                    sl_mult=inputs["sl_mult"], tp_mult=inputs["tp_mult"],
                    max_holding=inputs["max_holding"], force_exit=inputs["force_exit"],
                    rho=inputs["rho"])


# ---------------------------------------------------------------------------
# 4. θ_final y degradacion train -> OOS
# ---------------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    """Redondeo con .5 hacia arriba (Python redondea .5 al par: round(12.5) = 12)."""
    return int(math.floor(x + 0.5))


def _ema_slow(ema_fast: int, slow_ratio: float) -> int:
    """EMA lenta = round(ema_fast · slow_ratio); siempre al menos ema_fast + 1."""
    return max(int(round(ema_fast * slow_ratio)), ema_fast + 1)


def theta_from_values(values: dict) -> dict:
    """Arma un θ completo a partir de los parametros del espacio de busqueda.

    Redondea los enteros, recalcula ema_slow y agrega los parametros fijos.
    """
    theta = {name: values[name] for name in SEARCH_PARAMS}
    for name in INT_PARAMS:
        theta[name] = _round_half_up(theta[name])
    theta["ema_slow"] = _ema_slow(theta["ema_fast"], theta["slow_ratio"])
    return {**theta, **FIXED_PARAMS}


def theta_final(results: list[dict]) -> dict:
    """θ_final de cada estudio segun docs/SPEC.md, seccion 13.

    Mediana de cada parametro del espacio de busqueda sobre los mejores trials
    validos de las ventanas (las ventanas sin trial valido en ese estudio no
    cuentan; no se usa R5). Enteros redondeados y ema_slow recalculada.
    Regresa {estudio: θ, o None si ninguna ventana tuvo trial valido}.
    """
    final = {}
    for name in STUDIES:
        chosen = [r["best"][name]["params"] for r in results if r["best"][name] is not None]
        if not chosen:
            final[name] = None
            continue
        medians = {param: float(np.median([theta[param] for theta in chosen])) for param in SEARCH_PARAMS}
        final[name] = theta_from_values(medians)
    return final


def weekly_oos_returns(equity: pd.Series, results: list[dict]) -> pd.Series:
    """Retorno de cada semana de test sobre una curva OOS continua.

    retorno = equity al cierre de la ultima barra de la semana / equity al
    cierre de la ultima barra de la semana anterior - 1 (la primera semana se
    mide contra el valor inicial de la curva).
    Regresa una Series indexada por numero de ventana.
    """
    weekly = {}
    start_value = equity.iloc[0]
    for r in results:
        week = equity[(equity.index >= r["test_start"]) & (equity.index < r["test_end"])]
        weekly[r["number"]] = week.iloc[-1] / start_value - 1
        start_value = week.iloc[-1]
    return pd.Series(weekly, name="oos_weekly_return")


def degradation_table(results: list[dict], oos: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cuanto de la ventaja de train sobrevive fuera de muestra (PDF, pregunta 2).

    Por ventana: Calmar, trades y retorno semanal promedio del mejor trial
    global en train, y el retorno de su semana OOS en cada curva.

    Resumen, para cada curva OOS, con dos comparaciones:
    - Calmar: mediana del Calmar de train contra el Calmar de la curva OOS.
      OJO: no son directamente comparables. El Calmar de train se anualiza
      desde ~4 semanas y su drawdown maximo es el de un solo mes; el de la
      curva OOS cubre ~17 meses, donde el drawdown maximo es mucho mas grande.
      Por eso tambien se compara en la misma escala:
    - Retorno semanal: retorno total del mejor trial global / semanas de la
      ventana, promediado, contra el promedio de los retornos semanales OOS.
    proporcion_que_sobrevive = OOS / train en cada comparacion.

    Recibe los resultados por ventana y {nombre de curva: BacktestResult}.
    Regresa (tabla por ventana, resumen con una fila por curva y comparacion).
    """
    rows = []
    for r in results:
        best = r["best"]["global"]
        weeks = (r["train_end"] - r["train_start"]) / pd.Timedelta(days=7)
        rows.append({
            "window": r["number"],
            "test_start": r["test_start"],
            "calmar_train": best["calmar"] if best else np.nan,
            "n_trades_train": best["n_trades"] if best else 0,
            "weekly_return_train": best["total_return"] / weeks if best else np.nan,
        })
    table = pd.DataFrame(rows).set_index("window")
    for name, result in oos.items():
        table[f"weekly_return_oos_{name}"] = weekly_oos_returns(result.equity["equity"], results)

    summary = []
    for name, result in oos.items():
        train_calmar = table["calmar_train"].median()
        oos_calmar = calmar_ratio(result.equity["equity"])
        train_weekly = table["weekly_return_train"].mean()
        oos_weekly = table[f"weekly_return_oos_{name}"].mean()
        summary.append({"curve": name, "comparison": "calmar", "train": train_calmar,
                        "oos": oos_calmar, "share_survives": oos_calmar / train_calmar})
        summary.append({"curve": name, "comparison": "weekly_return", "train": train_weekly,
                        "oos": oos_weekly, "share_survives": oos_weekly / train_weekly})
    return table, pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# 5. Robustez (docs/SPEC.md, seccion 14)
# ---------------------------------------------------------------------------

def shift_param(name: str, base: float, factor: float) -> tuple[float, bool]:
    """Mueve un parametro por un factor (0.8 o 1.2) con las reglas de SPEC 14.1.

    Los enteros se redondean y se mueven al menos 1 unidad. Todo valor se
    recorta a los limites del espacio de busqueda.
    Regresa (valor nuevo, True si quedo igual al base: "en el limite").
    """
    low, high = SEARCH_BOUNDS[name]
    value = base * factor
    if name in INT_PARAMS:
        value = _round_half_up(value)
        if value == base:
            value = base + (1 if factor > 1 else -1)
    value = min(max(value, low), high)
    return value, value == base


def train_run(df: pd.DataFrame, theta: dict, single_vote: Optional[str] = None) -> tuple[float, int]:
    """Calmar y numero de trades de θ sobre todo df (un solo backtest)."""
    features = compute_strategy(df, theta, single_vote)
    result = backtest(df, features["signal"], features["atr"], config_from_params(theta, CAPITAL))
    return calmar_ratio(result.equity["equity"]), len(result.trades)


def sensitivity_table(df: pd.DataFrame, theta: dict) -> pd.DataFrame:
    """Sensibilidad ±20% de cada parametro de θ, uno a la vez (SPEC 14.1).

    Sirve para ver si θ esta en una meseta (el Calmar cambia poco) o en un
    pico (cambia mucho: señal de sobreajuste). ema_slow se recalcula con cada
    cambio de ema_fast o slow_ratio.
    Regresa una fila por parametro con los valores -20%, base y +20%, el
    Calmar en cada uno y si el valor movido quedo "en el limite".
    """
    base_calmar, _ = train_run(df, theta)
    rows = []
    for name in SEARCH_PARAMS:
        row = {"param": name, "value_base": theta[name], "calmar_base": base_calmar}
        for label, factor in [("minus_20", 0.8), ("plus_20", 1.2)]:
            value, at_limit = shift_param(name, theta[name], factor)
            values = {param: theta[param] for param in SEARCH_PARAMS}
            values[name] = value
            calmar, _ = train_run(df, theta_from_values(values))
            row.update({f"value_{label}": value, f"calmar_{label}": calmar, f"at_limit_{label}": at_limit})
        rows.append(row)
    columns = ["param", "value_minus_20", "value_base", "value_plus_20", "calmar_minus_20",
               "calmar_base", "calmar_plus_20", "at_limit_minus_20", "at_limit_plus_20"]
    return pd.DataFrame(rows)[columns]


def single_vote_table(df: pd.DataFrame, theta: dict) -> pd.DataFrame:
    """Un solo indicador contra la regla 2 de 3, con θ sobre todo df (SPEC 14.3)."""
    rows = {}
    for label, vote in [("EMA sola", "ema"), ("ROC sola", "roc"), ("BB sola", "bb"), ("2 de 3", None)]:
        calmar, n_trades = train_run(df, theta, vote)
        rows[label] = {"n_trades": n_trades, "calmar": calmar}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("rule")


def oos_cost_curve(df: pd.DataFrame, inputs: pd.DataFrame,
                   bps_grid=np.arange(0, 50.01, 2.5)) -> pd.DataFrame:
    """Retorno neto y Calmar de la curva OOS para cada comision por lado (SPEC 14.2).

    Usa EXACTAMENTE las mismas entradas por barra de la curva OOS (señal, ATR,
    sl, tp, holding, rho y force_exit); solo cambia la comision. No se
    vuelve a optimizar ni se recalcula la señal.
    Regresa una tabla indexada por la comision por lado en pb.
    """
    rows = {}
    for bps in bps_grid:
        equity = run_oos(df, inputs, commission_rate=bps / 10_000).equity["equity"]
        rows[float(bps)] = {"total_return": equity.iloc[-1] / equity.iloc[0] - 1,
                            "calmar": calmar_ratio(equity)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("commission_bps")


def cost_summary(curve: pd.DataFrame, real_bps: float = REAL_COST_BPS) -> pd.Series:
    """Break-even (comision donde el retorno neto cruza 0) y margen contra la comision real.

    Si la estrategia ya pierde con comision 0, no hay break-even: se marca
    loses_without_costs y gross_return_0bps es el resultado bruto.
    """
    gross = curve["total_return"].iloc[0]
    break_even = break_even_cost(curve.index, curve["total_return"])
    return pd.Series({
        "gross_return_0bps": gross,
        "return_at_real_cost": curve.loc[real_bps, "total_return"],
        "break_even_bps": break_even,
        "margin_bps": break_even - real_bps,
        "loses_without_costs": bool(gross <= 0),
    })


def entry_regimes(trades: pd.DataFrame, inputs: pd.DataFrame) -> pd.Series:
    """Regimen de la barra de entrada de cada trade de una curva OOS."""
    return pd.Series(inputs["regime"].to_numpy()[trades["entry_bar"].to_numpy()], index=trades.index)


def transitions_table(inputs: pd.DataFrame, trades: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    """Transiciones de regimen en la curva OOS y uso de R3 y R5, por regimen.

    - transitions_in_per_month: veces por mes que se entra al regimen (pasos de 1 hora).
    - regime_exits: trades cerrados por R3 (motivo regime_exit), por regimen de entrada.
    - r5_windows: ventanas donde ese regimen uso los parametros globales (R5).
    La fila "total" lleva todas las transiciones por mes.
    """
    labels = hourly(inputs["regime"]).dropna()
    months = len(labels) / HOURS_PER_MONTH
    changed = labels != labels.shift()
    changed.iloc[0] = False
    exits = trades[trades["exit_reason"] == "regime_exit"]
    exit_regimes = entry_regimes(exits, inputs)
    table = window_table(results)
    rows = {}
    for regime in REGIME_NAMES:
        rows[regime] = {
            "transitions_in_per_month": (changed & (labels == regime)).sum() / months,
            "regime_exits": int((exit_regimes == regime).sum()),
            "r5_windows": int(table.loc[table["study"] == regime, "used_r5"].sum()),
        }
    rows["total"] = {"transitions_in_per_month": changed.sum() / months,
                     "regime_exits": len(exits), "r5_windows": sum(r["r5_windows"] for r in rows.values())}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("regime")


# ---------------------------------------------------------------------------
# 6. Evaluacion final en el archivo de test (docs/SPEC.md, seccion 13)
# ---------------------------------------------------------------------------

def paste_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Pega train y test en una sola serie ordenada.

    Los dos archivos comparten la barra 2023-12-31 00:00; se queda la de train
    (el archivo con el que se ajusto todo).
    """
    pasted = pd.concat([df_train, df_test]).sort_index(kind="stable")
    return pasted[~pasted.index.duplicated(keep="first")]


def final_test_inputs(df_train: pd.DataFrame, df_test: pd.DataFrame, final: dict, use_regimes: bool = True,
                      test_start: str = TEST_START, test_end: str = TEST_END) -> pd.DataFrame:
    """Entradas por barra de la evaluacion final, tal como dice docs/SPEC.md, seccion 13.

    - Se pegan train y test; indicadores y regimenes se calculan sobre la
      serie pegada (solo usan el pasado).
    - El umbral de regimen se ajusta con todo train (fit_end = ultima barra de train).
    - Con use_regimes cada barra usa θ_final de su regimen (R5: si un regimen
      no tiene θ_final, usa el global) y force_exit marca el cambio a crisis
      (R3). Sin use_regimes, toda la ventana usa θ_final global.
    - Solo se regresan las barras de test_start a test_end: el backtest no
      ve nada antes y no puede haber entradas antes de test_start.

    Limitacion: la ventana de regimen de 1 semana cuenta barras, no tiempo.
    Por el hueco de 122 dias, en la primera semana de mayo de 2024 esa ventana
    todavia incluye barras del 31 de diciembre de 2023; lo mismo pasa con las
    velas de 4h, las EMAs y el ATR justo despues del hueco. Es causal (solo
    usa el pasado), pero mezcla dos epocas distintas del mercado.

    Recibe train, test y θ_final por estudio ({"global", "crisis", ...}).
    Regresa la tabla por barra (regime, signal, atr, sl_mult, tp_mult,
    max_holding, rho, force_exit) indexada por las barras de test.
    """
    pasted = paste_train_test(df_train, df_test)
    labels, _ = rule_regimes(pasted, df_train.index[-1])
    test_index = pasted.loc[test_start:test_end].index

    inputs = empty_inputs(labels.loc[test_index])
    for name in (REGIME_NAMES if use_regimes else ["global"]):
        theta = final[name] if final.get(name) is not None else final["global"]
        features = compute_strategy(pasted, theta).loc[test_index]
        bars = test_index if name == "global" else test_index[(inputs["regime"] == name).to_numpy()]
        inputs = fill_inputs(inputs, bars, features, theta)
    inputs["force_exit"] = crisis_entries(inputs["regime"]) if use_regimes else False
    return inputs


def run_final_test(df_train: pd.DataFrame, df_test: pd.DataFrame, final: dict,
                   capital: float = CAPITAL) -> tuple[dict, dict]:
    """Corre la evaluacion final: curva con regimen y curva solo global (un backtest cada una).

    Regresa ({nombre: BacktestResult}, {nombre: entradas por barra}).
    """
    pasted = paste_train_test(df_train, df_test)
    inputs = {"con régimen": final_test_inputs(df_train, df_test, final, use_regimes=True),
              "solo global": final_test_inputs(df_train, df_test, final, use_regimes=False)}
    curves = {name: run_oos(pasted, table, capital) for name, table in inputs.items()}
    return curves, inputs
