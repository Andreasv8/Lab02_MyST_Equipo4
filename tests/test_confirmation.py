"""Prueba 2 del PDF (seccion 3.7): regla de confirmacion 2 de 3 con filtro de ADX."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals import (MIN_VOTES, confirmation_signal, confirmed_state, entry_signal,
                         strategy_votes)

ADX_THRESHOLD = 20


def _state(vote_ema: int, vote_roc: int, vote_bb: int, adx: float = 30.0) -> int:
    """Estado de una sola barra con votos dados a mano."""
    votes = pd.DataFrame({"vote_ema": [vote_ema], "vote_roc": [vote_roc], "vote_bb": [vote_bb]})
    return int(confirmed_state(votes, pd.Series([adx]), ADX_THRESHOLD).iloc[0])


@pytest.mark.parametrize("votes, expected", [
    ((1, 0, 0), 0),       # 1 voto a favor de largo: no abre
    ((0, -1, 0), 0),      # 1 voto a favor de corto: no abre
    ((1, 1, 0), 1),       # 2 a favor de largo: abre largo
    ((1, 1, 1), 1),       # 3 a favor de largo: abre largo
    ((-1, 0, -1), -1),    # 2 a favor de corto: abre corto
    ((-1, -1, -1), -1),   # 3 a favor de corto: abre corto
    ((1, -1, 0), 0),      # 1 a favor y 1 en contra: no abre
    ((1, -1, 1), 1),      # 2 a favor y 1 en contra: abre
])
def test_two_of_three_rule(votes, expected):
    assert _state(*votes) == expected


@pytest.mark.parametrize("votes", [(1, 1, 1), (-1, -1, -1)])
def test_low_adx_blocks_entry(votes):
    """Con ADX en o bajo el umbral no se abre, aunque los 3 votos coincidan."""
    assert _state(*votes, adx=ADX_THRESHOLD) == 0
    assert _state(*votes, adx=10.0) == 0
    assert _state(*votes, adx=np.nan) == 0


def test_signal_only_on_state_change():
    """La señal aparece solo en la barra donde cambia el estado."""
    state = pd.Series([0, 1, 1, 1, 0, -1, -1, 1, 1])
    expected = [0, 1, 0, 0, 0, -1, 0, 1, 0]
    assert entry_signal(state).tolist() == expected


def test_signal_on_first_bar_if_state_starts_active():
    """Si el estado arranca en +1, la primera barra cuenta como cambio desde 0."""
    assert entry_signal(pd.Series([1, 1, 0])).tolist() == [1, 0, 0]


def test_bollinger_vote_uses_threshold():
    """vote_bb = +1 si %B > u, -1 si %B < 1 - u y 0 en medio o en NaN."""
    percent_b = pd.Series([0.8, 0.2, 0.5, 0.7, np.nan])
    zeros = pd.Series(0.0, index=percent_b.index)
    votes = strategy_votes(zeros, zeros, zeros, percent_b, bb_threshold=0.7)
    assert votes["vote_bb"].tolist() == [1, -1, 0, 0, 0]
    assert (votes["vote_ema"] == 0).all() and (votes["vote_roc"] == 0).all()


@pytest.mark.parametrize("votes, expected", [
    ((0, 0, 0), 0),      # nadie vota
    ((1, 0, 0), 0),      # 1 a favor de largo: no abre
    ((0, -1, 0), 0),     # 1 a favor de corto: no abre
    ((1, -1, 0), 0),     # 1 largo y 1 corto: no abre
    ((1, 1, 0), 1),      # 2 de 3 largos: abre largo
    ((1, 0, 1), 1),
    ((1, 1, -1), 1),     # 2 largos contra 1 corto: abre largo
    ((1, 1, 1), 1),      # 3 de 3
    ((-1, -1, 0), -1),   # 2 de 3 cortos: abre corto
    ((-1, 1, -1), -1),
    ((-1, -1, -1), -1),
])
def test_confirmation_rule_two_of_three(votes, expected):
    """Regla 2 de 3 sola, sin el filtro de ADX (confirmation_signal)."""
    assert MIN_VOTES == 2
    votes_df = pd.DataFrame([votes], columns=["vote_ema", "vote_roc", "vote_bb"])
    assert confirmation_signal(votes_df).iloc[0] == expected
