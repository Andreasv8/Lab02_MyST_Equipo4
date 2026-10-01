"""Calcula indicadores tecnicos sobre NVDA_daily.csv y analiza su correlacion.

Todos los indicadores se calculan manualmente con pandas/numpy (sin la
libreria ta), en src/indicators.py, y entran en forma estacionaria (distancias,
razones, osciladores acotados) para que la correlacion no refleje solo la
tendencia comun del precio. El ATR no participa: se reserva para SL/TP y sizing.

La correlacion se mide SOLO sobre el periodo de train, para que la seleccion
de features no vea test ni validation (data snooping). Genera un heatmap
(indicator_corr.png), imprime la matriz numerica y la tabla de pares con
|correlacion| >= CORR_THRESHOLD.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indicators import build_stationary_indicators  # noqa: E402
from src.splits import SPLITS, get_split  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
INPUT_CSV = DATA_DIR / "NVDA_daily.csv"
OUTPUT_PNG = DATA_DIR / "indicator_corr.png"

# Umbral de clase: con |r| >= 0.70 dos indicadores miden lo mismo.
CORR_THRESHOLD = 0.70


def high_correlation_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pares de indicadores con |correlacion| >= threshold.

    Parametros
    ----------
    corr : pd.DataFrame
        Matriz de correlacion cuadrada.
    threshold : float
        Umbral de |correlacion| para considerar dos indicadores redundantes.

    Regresa
    -------
    pd.DataFrame
        Columnas ind_a, ind_b, corr; un renglon por par (triangulo superior),
        ordenado por |corr| descendente.
    """
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["ind_a", "ind_b", "corr"]
    pairs = pairs[pairs["corr"].abs() >= threshold]
    return pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path, title: str) -> None:
    """Genera y guarda un heatmap de una matriz de correlacion."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(INPUT_CSV, index_col="Date", parse_dates=True)

    # Se calcula sobre toda la serie (warm-up de las ventanas) y despues se
    # recorta a train; los indicadores son causales, asi que train no ve el futuro.
    indicators = get_split(build_stationary_indicators(df), "train")
    corr = indicators.corr()

    # El warm-up de las ventanas (hasta 50 barras) recorta el inicio de train,
    # porque los datos empiezan el mismo dia que el split.
    start, end = indicators.index[0].date(), indicators.index[-1].date()
    print(f"Train (split {SPLITS['train'][0]} a {SPLITS['train'][1]}): con indicadores validos "
          f"{start} a {end} ({len(indicators)} barras, {indicators.shape[1]} candidatos)\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(corr.round(2))

    pairs = high_correlation_pairs(corr, CORR_THRESHOLD)
    print(f"\nPares con |correlacion| >= {CORR_THRESHOLD:.2f} ({len(pairs)}):")
    print(pairs.round(3).to_string(index=False))

    title = f"Correlacion entre indicadores (estacionarios) -- NVDA, train {start} a {end}"
    plot_correlation_heatmap(corr, OUTPUT_PNG, title)
    print(f"\nHeatmap guardado en {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
