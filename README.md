# Lab02_MyST_Equipo4

## Integrantes

- Andrea Santoyo
- Isabela Torres

## Nivel de alcance

B (BTCUSDT 5 minutos, con detección de régimen).

## Descripción

PENDIENTE

## Instalación

Requiere Python 3.12 o superior (probado con 3.14.5).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducir resultados

```bash
python main.py
```

PENDIENTE: `main.py` aún no existe.

## Pruebas

```bash
python -m pytest -q
```

## Semilla

| Constante | Archivo | Uso |
|---|---|---|
| `SEED = 42` | `src/optimize.py` | Semilla del `TPESampler` de Optuna. Cada ventana del walk-forward usa `SEED + número de ventana`. |
| `SEED = 42` | `src/regimes.py` | `random_state` de K-means (solo para comparar regímenes). |
| `HMM_SEEDS = range(10)` | `src/regimes.py` | El HMM (solo para comparar) se ajusta con las semillas 0–9 y se queda el de mayor log-likelihood. |

## Estructura del repositorio

```
.
├── README.md
├── main.py
├── requirements.txt
├── .gitignore
├── .claude/
│   └── CLAUDE.md
├── data/
│   ├── btc_project_train.csv
│   └── btc_project_test.csv
├── docs/
│   ├── SPEC.md
│   ├── figures/         (figuras que genera main.py)
│   └── tables/          (tablas que genera main.py)
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── backtest.py
│   ├── data.py
│   ├── metrics.py
│   ├── optimize.py
│   ├── plots.py
│   ├── regimes.py
│   └── signals.py
└── tests/
    ├── golden/
    │   └── backtest_scenario.csv
    ├── test_accounting.py
    ├── test_backtest.py
    ├── test_causality.py
    ├── test_confirmation.py
    ├── test_data.py
    ├── test_execution.py
    ├── test_final_test.py
    ├── test_metrics.py
    ├── test_optimize.py
    ├── test_plots.py
    ├── test_regimes.py
    └── test_robustness.py
```

## Uso de herramientas de IA

PENDIENTE
