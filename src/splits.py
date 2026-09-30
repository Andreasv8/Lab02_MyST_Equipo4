"""Division temporal train/test/validation de lab_02 (SPEC.md, seccion 1).

Los periodos son bloques contiguos (sin aleatorizar). Los limites viven solo
aqui para que el analisis de indicadores, el backtest y los notebooks usen
exactamente las mismas fechas.
"""

import pandas as pd

SPLITS = {
    "train": ("2021-09-15", "2024-09-14"),
    "test": ("2024-09-15", "2025-09-14"),
    "validation": ("2025-09-15", "2026-09-15"),
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
