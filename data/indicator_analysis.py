"""Calcula indicadores tecnicos sobre NVDA_daily.csv y analiza su correlacion.

Todos los indicadores se calculan manualmente con pandas/numpy (sin la
libreria ta), en src/indicators.py. La correlacion se mide SOLO sobre el
periodo de train, para que la seleccion de features no vea test ni
validation (data snooping). Genera un heatmap (indicator_corr.png) e
imprime la matriz numerica en consola.
"""

import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indicators import build_indicators  # noqa: E402
from src.splits import get_split  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
INPUT_CSV = DATA_DIR / "NVDA_daily.csv"
OUTPUT_PNG = DATA_DIR / "indicator_corr.png"


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Genera y guarda un heatmap de una matriz de correlacion."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlacion entre indicadores tecnicos -- NVDA (train)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(INPUT_CSV, index_col="Date", parse_dates=True)

    # Se calcula sobre toda la serie (warm-up de las ventanas) y despues se
    # recorta a train; los indicadores son causales, asi que train no ve el futuro.
    indicators = get_split(build_indicators(df), "train")
    corr = indicators.corr()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(corr.round(3))

    plot_correlation_heatmap(corr, OUTPUT_PNG)
    print(f"\nHeatmap guardado en {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
