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
│   ├── ACT07_ROBUST.md
│   └── SPEC.md
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── backtest.py
│   ├── data.py
│   ├── ema_search.py
│   ├── indicators.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── plots.py
│   ├── regime_analysis.py
│   ├── regimes.py
│   ├── robust_evaluation.py
│   ├── splits.py
│   ├── strategy.py
│   └── walk_forward.py
└── tests/
    ├── golden/
    │   └── backtest_scenario.csv
    ├── test_analysis.py
    ├── test_backtest.py
    ├── test_ema_adx.py
    ├── test_metrics.py
    ├── test_optimization.py
    ├── test_regime_analysis.py
    ├── test_regimes.py
    ├── test_robust_evaluation.py
    ├── test_signals.py
    ├── test_strategy.py
    ├── test_truncation.py
    └── test_walk_forward.py
```

## Uso de herramientas de IA

PENDIENTE
