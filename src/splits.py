"""Division temporal train/test/validation de lab_02 (docs/SPEC.md, seccion 1).

Los periodos son bloques contiguos (sin aleatorizar). Los limites viven solo
aqui para que el analisis de indicadores, el backtest y los notebooks usen
exactamente las mismas fechas.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = str(ROOT / "data" / "btc_project_train.csv")
TEST_PATH = str(ROOT / "data" / "btc_project_test.csv")

SPLITS = {
    "train": ("2022-06-01", "2023-05-14"),
    "test": ("2023-05-15", "2023-09-06"),
    "validation": ("2023-09-07", "2023-12-31"),
    "final": ("2024-05-02", "2024-06-03"),
}


def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Regresa las filas de df que caen en el periodo indicado.

    Parametros
    ----------
    df : pd.DataFrame
        Datos con indice de fechas (DatetimeIndex).
    name : str
        "train", "test" o "validation".

    Regresa
    -------
    pd.DataFrame
        Subconjunto de df entre las fechas del periodo (ambas inclusive).
    """
    start, end = SPLITS[name]
    return df.loc[start:end]
