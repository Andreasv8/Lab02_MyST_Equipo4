# CLAUDE.md -- lab_02

Hereda las convenciones globales del CLAUDE.md de la raiz del repo.

---

# Decisiones tomadas

- Ticker: NVDA, datos diarios (1d), 5 años: 2021-09-15 a 2026-09-15
  (el profesor pidió 5 años para datos diarios). Split train/test/validation
  en `src/splits.py`.
- Fuente: yfinance, descargado con `data/download_data.py`. Los notebooks
  NO llaman a yfinance directo; cargan `data/NVDA_daily.csv`.
- Los indicadores viven en `src/indicators.py`; la logica de estrategia en
  `src/strategy.py`. El notebook solo importa, ejecuta y presenta resultados.
- Ver `SPEC.md` para la especificacion completa de la estrategia.

---

# Estructura de carpetas

```
lab_02/
├── CLAUDE.md
├── SPEC.md
├── Act_06_backtest.ipynb  <-- entregable Act 06 (train + test; validation no se carga)
├── data/
│   ├── download_data.py
│   ├── indicator_analysis.py   <-- correlacion de indicadores (solo train)
│   ├── indicator_corr.png
│   └── NVDA_daily.csv
├── src/
│   ├── __init__.py
│   ├── analysis.py         <-- sensibilidad a costos, break-even, desglose de trades
│   ├── backtest.py         <-- motor orientado a eventos (cash, shares, equity)
│   ├── indicators.py
│   ├── metrics.py          <-- metricas puras (Sharpe, Sortino, DD, CAGR, turnover, ...)
│   ├── plots.py            <-- graficas del notebook (matplotlib)
│   ├── splits.py
│   └── strategy.py
└── tests/
    ├── golden/backtest_scenario.csv
    ├── test_analysis.py
    ├── test_backtest.py
    ├── test_metrics.py
    ├── test_strategy.py
    └── test_truncation.py
```
