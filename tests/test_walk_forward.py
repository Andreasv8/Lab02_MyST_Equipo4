"""Pruebas de src/walk_forward.py (Act 07 v2, ACT07_ROBUST.md).

Datos: BTC de 5 minutos hasta el fin de test (src/splits.py); validation no se usa.
No se corren los 200 trials: la meseta y la regla a priori se prueban con
estudios sinteticos.
"""

import math
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import backtest
from src.data import load_btc
from src.optimization import THETA0, theta_config, theta_signal
from src.splits import SPLITS
from src.walk_forward import (
    BLOCKS,
    chain_equity,
    fit_block_models,
    operates,
    plateau_theta,
    prepare_block,
    run_block,
    run_block_regimes,
    top_trials,
    walk_forward_regimes_run,
    walk_forward_run,
)

TEST_END = SPLITS["test"][1]
DF = load_btc(str(Path(__file__).resolve().parents[1] / "data" / "btc_project_train.csv")).loc[:TEST_END]
MODELS = [fit_block_models(DF, block.fit_end) for block in BLOCKS]

# θ con salida distinta de θ0 para que el backtest no sea el de siempre.
THETA = {**THETA0, "sl_mult": 1.5, "rr": 2.0, "max_holding": 288}
REGIMES = [None, "crisis", "trend", "mean_reversion"]
BLOCK_IDS = range(len(BLOCKS))


def _corrupt_after(df: pd.DataFrame, end: str) -> pd.DataFrame:
    """Copia de df con los precios y volumen posteriores al dia end multiplicados por ruido."""
    rng = np.random.default_rng(42)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    out = df.copy()
    out[cols] = out[cols].astype(float)
    after = out.index.normalize() > pd.Timestamp(end)
    noise = rng.uniform(0.5, 1.5, size=(after.sum(), 1))
    out.loc[after, cols] = out.loc[after, cols].to_numpy() * noise
    return out


def _assert_models_equal(a, b):
    pd.testing.assert_series_equal(a.scaler.mean, b.scaler.mean, check_exact=True)
    pd.testing.assert_series_equal(a.scaler.std, b.scaler.std, check_exact=True)
    assert a.vol_threshold == b.vol_threshold
    assert a.hmm_names == b.hmm_names
    assert a.hmm_choice == b.hmm_choice
    # El EM del HMM usa BLAS multihilo y el orden de las sumas puede variar entre
    # corridas (diferencias relativas ~1e-9, absolutas ~1e-12, p. ej. en probabilidades
    # iniciales de 1e-107); un look-ahead real daria diferencias grandes.
    for attr in ["startprob_", "transmat_", "means_", "covars_"]:
        np.testing.assert_allclose(getattr(a.hmm, attr), getattr(b.hmm, attr), rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Anti look-ahead
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("i", BLOCK_IDS)
def test_block_models_ignore_data_after_fit_end(i):
    """Modelos del bloque identicos con df completo, df.loc[:fit_end] y futuro corrompido."""
    fit_end = BLOCKS[i].fit_end
    _assert_models_equal(MODELS[i], fit_block_models(DF.loc[:fit_end], fit_end))
    _assert_models_equal(MODELS[i], fit_block_models(_corrupt_after(DF, fit_end), fit_end))


@pytest.mark.parametrize("i", BLOCK_IDS)
@pytest.mark.parametrize("regime", REGIMES)
def test_eval_block_ignores_data_after_eval_end(i, regime):
    """Etiquetas, equity y trades del tramo identicos si se borra o corrompe todo lo posterior."""
    block = BLOCKS[i]
    full = prepare_block(DF, block, MODELS[i])
    run_full = run_block(full, THETA, regime)
    for df_alt in [DF.loc[:block.eval_end], _corrupt_after(DF, block.eval_end)]:
        alt = prepare_block(df_alt, block, MODELS[i])
        pd.testing.assert_series_equal(full.labels, alt.labels)
        run_alt = run_block(alt, THETA, regime)
        pd.testing.assert_frame_equal(run_full.equity, run_alt.equity, check_exact=True)
        pd.testing.assert_frame_equal(run_full.trades, run_alt.trades, check_exact=True)


def test_eval_window_is_inside_block():
    """El tramo de evaluacion empieza y termina dentro de sus fechas."""
    for block, models in zip(BLOCKS, MODELS):
        data = prepare_block(DF, block, models)
        assert data.df_eval.index[0] >= pd.Timestamp(block.eval_start)
        assert data.df_eval.index[-1].normalize() <= pd.Timestamp(block.eval_end)
        assert data.labels.notna().all()


@pytest.mark.parametrize("i", BLOCK_IDS)
@pytest.mark.parametrize("regime", REGIMES)
def test_sliced_backtest_matches_masked_full_backtest(i, regime):
    """Decision 4: backtest sobre df.loc[:eval_end] con entradas solo con barra de señal en el
    tramo (y en el regimen) == backtest recortado al tramo."""
    block = BLOCKS[i]
    data = prepare_block(DF, block, MODELS[i])
    df_trunc = DF.loc[:block.eval_end]
    signal, atr = theta_signal(df_trunc, THETA0)
    allowed = pd.Series(df_trunc.index >= pd.Timestamp(block.eval_start), index=df_trunc.index)
    if regime is not None:
        allowed &= (data.labels == regime).reindex(df_trunc.index, fill_value=False)
    full = backtest(df_trunc, signal.where(allowed, 0), atr, theta_config(THETA))

    sliced = run_block(data, THETA, regime)
    pd.testing.assert_frame_equal(full.equity.loc[block.eval_start:], sliced.equity, check_exact=True)
    bar_cols = ["entry_bar", "exit_bar"]
    pd.testing.assert_frame_equal(full.trades.drop(columns=bar_cols),
                                  sliced.trades.drop(columns=bar_cols), check_exact=True)


def test_regime_entries_only_in_regime():
    """Con regime = trend, toda entrada tiene etiqueta trend en la barra de señal.

    Con entradas solo al cambiar la tendencia de 4h hay pocos trades por bloque,
    asi que se exige al menos uno en los 3 bloques juntos.
    """
    n_trades = 0
    for block, models in zip(BLOCKS, MODELS):
        data = prepare_block(DF, block, models)
        trades = run_block(data, THETA, "trend").trades
        assert (data.labels.iloc[trades["entry_bar"] - 1] == "trend").all()
        n_trades += len(trades)
    assert n_trades > 0


# ---------------------------------------------------------------------------
# Encadenado de equity
# ---------------------------------------------------------------------------

def test_chain_equity_compounds_returns_hand_case():
    """Tramos [100, 110] y [100, 90] con capital 100: +10% y luego -10% sobre 110 -> 99."""
    idx1 = pd.to_datetime(["2023-01-02", "2023-01-03"])
    idx2 = pd.to_datetime(["2023-07-03", "2023-07-05"])
    chained = chain_equity([pd.Series([100.0, 110.0], index=idx1),
                            pd.Series([100.0, 90.0], index=idx2)], 100.0)
    np.testing.assert_allclose(chained.to_numpy(), [100.0, 110.0, 110.0, 99.0])
    assert list(chained.index) == list(idx1.append(idx2))


def test_chain_equity_first_bar_against_initial_cash():
    """Si la primera barra de un tramo ya no vale initial_cash, ese rendimiento cuenta."""
    idx1 = pd.to_datetime(["2023-01-02", "2023-01-03"])
    idx2 = pd.to_datetime(["2023-07-03", "2023-07-05"])
    chained = chain_equity([pd.Series([100.0, 120.0], index=idx1),
                            pd.Series([105.0, 105.0], index=idx2)], 100.0)
    np.testing.assert_allclose(chained.to_numpy(), [100.0, 120.0, 126.0, 126.0])


# ---------------------------------------------------------------------------
# Meseta y regla a priori
# ---------------------------------------------------------------------------

DISTRIBUTIONS = {
    "sl_mult": optuna.distributions.FloatDistribution(1.0, 3.0),
    "rr": optuna.distributions.FloatDistribution(1.0, 3.0),
    "max_holding": optuna.distributions.IntDistribution(5, 20),
}


def _synthetic_study(rows: list) -> optuna.Study:
    """Estudio con trials (value, sl_mult, rr, max_holding) dados a mano."""
    study = optuna.create_study(direction="maximize")
    for value, sl, rr, hold in rows:
        study.add_trial(optuna.trial.create_trial(
            params={"sl_mult": sl, "rr": rr, "max_holding": hold},
            distributions=DISTRIBUTIONS, value=value))
    return study


def test_plateau_hand_case():
    """20 trials -> top 10% = 2 mejores con J finito; mediana = promedio; 12.5 -> 13."""
    rows = [(5.0, 1.2, 2.0, 12), (4.0, 1.6, 3.0, 13)]           # top 2
    rows += [(1.0, 3.0, 1.0, 20)] * 8                              # peores finitos
    rows += [(-math.inf, 1.0, 1.0, 5)] * 10                        # sin trades suficientes
    study = _synthetic_study(rows)
    assert [t.value for t in top_trials(study)] == [5.0, 4.0]
    theta = plateau_theta(study)
    assert theta["sl_mult"] == pytest.approx(1.4)
    assert theta["rr"] == pytest.approx(2.5)
    assert theta["max_holding"] == 13
    signal_params = ["timeframe", "ema_fast", "ema_slow", "adx_threshold", "entry_on_change"]
    assert {k: theta[k] for k in signal_params} == {k: THETA0[k] for k in signal_params}


def test_plateau_median_odd_top():
    """Top 3 (30 trials): mediana del medio; 3 trials con hold 8, 9, 10 -> 9."""
    rows = [(9.0, 2.0, 1.0, 8), (8.0, 1.0, 2.0, 10), (7.0, 3.0, 3.0, 9)]
    rows += [(0.5, 1.0, 1.0, 20)] * 27
    theta = plateau_theta(_synthetic_study(rows))
    assert (theta["sl_mult"], theta["rr"], theta["max_holding"]) == (2.0, 2.0, 9)


def test_plateau_fewer_finite_than_top():
    """Solo 1 trial finito de 20: se usa ese; sin finitos -> None."""
    rows = [(2.0, 1.5, 1.5, 7)] + [(-math.inf, 1.0, 1.0, 5)] * 19
    study = _synthetic_study(rows)
    assert len(top_trials(study)) == 1
    assert plateau_theta(study)["max_holding"] == 7
    assert plateau_theta(_synthetic_study([(-math.inf, 1.0, 1.0, 5)] * 5)) is None


@pytest.mark.parametrize("best, expected", [(0.3, True), (0.0, False), (-0.2, False),
                                            (-math.inf, False)])
def test_operates_rule(best, expected):
    """Regla a priori: se opera solo si el mejor J > 0."""
    study = _synthetic_study([(best, 1.0, 1.0, 5), (min(best, -1.0), 2.0, 2.0, 6)])
    assert operates(study) is expected


# ---------------------------------------------------------------------------
# Estrategia combinada por regimen en el walk-forward (aclaracion 10)
# ---------------------------------------------------------------------------

THETA_CRISIS = {**THETA0, "sl_mult": 1.2, "rr": 2.5, "max_holding": 144}
THETA_TREND = {**THETA0, "sl_mult": 2.8, "rr": 1.9, "max_holding": 576}
COMBINED = {"crisis": THETA_CRISIS, "trend": THETA_TREND, "mean_reversion": None}
BLOCKS_DATA = [prepare_block(DF, block, models) for block, models in zip(BLOCKS, MODELS)]


@pytest.mark.parametrize("regime", ["crisis", "trend"])
def test_combined_with_one_regime_equals_regime_run(regime):
    """Con un solo regimen con θ, la combinada == walk_forward_run(.., θ, regimen)."""
    theta = COMBINED[regime]
    only = {name: (theta if name == regime else None) for name in COMBINED}
    combined = walk_forward_regimes_run(BLOCKS_DATA, only)
    single = walk_forward_run(BLOCKS_DATA, theta, regime)
    pd.testing.assert_frame_equal(combined.equity, single.equity, check_exact=True)
    pd.testing.assert_frame_equal(combined.trades, single.trades, check_exact=True)


@pytest.mark.parametrize("i", BLOCK_IDS)
def test_combined_entries_use_their_regime_theta(i):
    """Toda entrada tiene etiqueta crisis o trend en su barra de señal; en las salidas por SL
    intrabar, |salida - entrada| = sl_mult del regimen de la barra de señal · ATR."""
    data = BLOCKS_DATA[i]
    trades = run_block_regimes(data, COMBINED).trades
    signal_labels = data.labels.iloc[trades["entry_bar"] - 1].to_numpy()
    assert set(signal_labels) <= {"crisis", "trend"}
    sl = np.array([COMBINED[label]["sl_mult"] for label in signal_labels])
    stops = trades["exit_reason"] == "stop_loss"
    if stops.any():
        moved = (trades.loc[stops, "raw_exit_price"] - trades.loc[stops, "raw_entry_price"]).abs()
        gap = trades.loc[stops, "exit_phase"] == "open"
        expected = sl[stops.to_numpy()] * trades.loc[stops, "entry_atr"]
        np.testing.assert_allclose(moved[~gap], expected[~gap], rtol=1e-9)


@pytest.mark.parametrize("i", BLOCK_IDS)
def test_combined_ignores_data_after_eval_end(i):
    """La combinada del tramo es identica si se borra o corrompe todo lo posterior a eval_end."""
    block = BLOCKS[i]
    full = run_block_regimes(BLOCKS_DATA[i], COMBINED)
    for df_alt in [DF.loc[:block.eval_end], _corrupt_after(DF, block.eval_end)]:
        alt = run_block_regimes(prepare_block(df_alt, block, MODELS[i]), COMBINED)
        pd.testing.assert_frame_equal(full.equity, alt.equity, check_exact=True)
        pd.testing.assert_frame_equal(full.trades, alt.trades, check_exact=True)
