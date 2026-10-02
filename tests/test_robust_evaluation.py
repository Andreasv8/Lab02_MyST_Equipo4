"""Pruebas de src/robust_evaluation.py: criterios pre-registrados con casos a mano.

No evalua test: solo funciones puras sobre cifras inventadas.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization import THETA0
from src.robust_evaluation import drop, drop_criterion, drop_table, edge_params


def test_drop_hand_case():
    """J 5 -> 1: caida absoluta 4, relativa 0.8."""
    assert drop(5.0, 1.0) == pytest.approx((4.0, 0.8))


def test_drop_criterion_both_must_hold():
    """v1 4 -> 2 (abs 2, rel 0.5); v2 2 -> 0.5 (abs 1.5, rel 0.75): abs cumple, rel no -> no cumple."""
    out = drop_criterion(4.0, 2.0, 2.0, 0.5)
    assert out["abs cumple"] and not out["rel cumple"]
    assert out["cumple"] is False


def test_drop_criterion_holds():
    """v1 6 -> 1 (abs 5, rel 0.83); v2 4 -> 3 (abs 1, rel 0.25): cumple."""
    out = drop_criterion(6.0, 1.0, 4.0, 3.0)
    assert out["cumple"] is True
    assert out["v2 caída rel"] == pytest.approx(0.25)


def test_drop_table_reads_calmar_columns():
    """v1 usa Calmar train/test; v2 usa J walk-forward y Calmar test."""
    table = pd.DataFrame({"v1 θ* train": [5.0], "v1 θ* test": [1.0], "v2 θ* robusto test": [3.0],
                          "v1 θ*_régimen train": [7.0], "v1 θ*_régimen test": [6.0],
                          "v2 θ*_régimen robusto test": [1.0]}, index=["calmar"])
    out = drop_table(table, {"θ*": 4.0, "θ*_régimen": 5.0})
    assert out.loc["θ*", "cumple"] and not out.loc["θ*_régimen", "cumple"]
    assert out.loc["θ*_régimen", "v2 caída abs"] == pytest.approx(4.0)


def test_edge_params_hand_case():
    """10% del ancho: sl 2.80 en [1,3] -> borde (0.2 de 3); 2.79 no; hold 5 y 6.5 en [5,20] -> borde; 7 no."""
    thetas = {
        "a": {**THETA0, "sl_mult": 2.80, "rr": 2.0, "max_holding": 5},
        "b": {**THETA0, "sl_mult": 2.79, "rr": 1.2, "max_holding": 7},
        "c": {**THETA0, "sl_mult": 2.0, "rr": 2.0, "max_holding": 6.5},
        "off": None,
    }
    edge = edge_params(thetas)["en_borde"]
    assert edge["a", "sl_mult"] and edge["a", "max_holding"] and not edge["a", "rr"]
    assert not edge["b", "sl_mult"] and edge["b", "rr"] and not edge["b", "max_holding"]
    assert edge["c", "max_holding"]
    assert "off" not in edge.index.get_level_values(0)
