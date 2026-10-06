"""Division temporal train/test/validation de lab_02 (SPEC.md, seccion 1).

Los periodos son bloques contiguos (sin aleatorizar). Los limites viven solo
aqui para que el analisis de indicadores, el backtest y los notebooks usen
exactamente las mismas fechas.
"""

import pandas as pd

TRAIN_PATH = "data/btc_project_train.csv"
TEST_PATH = "data/btc_project_test.csv"

SPLITS = {
    "train": ("2022-06-01", "2023-12-30"),
    # El archivo de test trae un dia suelto (2023-12-31) y un hueco de 122 dias.
    # Se usa solo el tramo continuo de mayo de 2024.
    "test": ("2024-05-02", "2024-06-03"),
    "validation": ("2024-06-04", "2024-12-31"),
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
