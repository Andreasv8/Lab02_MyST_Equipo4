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
| `SEED = 42` | `src/optimization.py` | Semilla del `TPESampler` de Optuna; `src/walk_forward.py` la importa para su propia optimización. |
| `SEED = 42` | `src/regimes.py` | `random_state` de K-means. |
| `HMM_SEEDS = range(10)` | `src/regimes.py` | El HMM se ajusta con las semillas 0–9 y se conserva el de mayor log-likelihood en train. |

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── .claude/
│   └── CLAUDE.md
├── data/
│   ├── btc_project_train.csv
│   └── btc_project_test.csv
├── docs/
│   └── SPEC.md
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── backtest.py
│   ├── data.py
│   ├── metrics.py
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
    ├── test_metrics.py
    └── test_regimes.py
```

## Uso de herramientas de IA

PENDIENTE
