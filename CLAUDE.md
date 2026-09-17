# CLAUDE.md -- lab_02

Hereda las convenciones globales del CLAUDE.md de la raiz del repo.

---

# Decisiones tomadas

- Ticker: NVDA, datos diarios (1d) de 2024-01-01 a 2026-09-15.
- Fuente: yfinance, descargado con `data/download_data.py`. Los notebooks
  NO llaman a yfinance directo; cargan `data/NVDA_daily.csv`.
- La logica de estrategia vive en `src/strategy.py`; el notebook solo importa,
  ejecuta y presenta resultados.
- Ver `SPEC.md` para el detalle de la actividad (pendiente de completar).

---

# Estructura de carpetas

```
lab_02/
├── CLAUDE.md
├── SPEC.md
├── data/
│   ├── download_data.py
│   └── NVDA_daily.csv
├── src/
│   ├── __init__.py
│   └── strategy.py
└── tests/
    └── test_strategy.py
```
