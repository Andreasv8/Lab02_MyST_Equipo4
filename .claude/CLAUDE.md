# CLAUDE.md — Lab02_MyST_Equipo4

## Proyecto

Lab 02 de Microestructura y Sistemas de Trading (ITESO), **Nivel B**: estrategia sobre
**BTCUSDT en velas de 5 minutos**, con detección de régimen. Equipo 4.
La especificación de la estrategia está en `docs/SPEC.md`.

## Estructura

```
.
├── README.md
├── requirements.txt
├── data/            btc_project_train.csv, btc_project_test.csv (congelados en el repo)
├── docs/            SPEC.md
├── notebooks/       analysis.ipynb
├── src/             data, signals, backtest, metrics, regimes, plots, ... (toda la lógica)
└── tests/           test_*.py (pytest)
```

## Reglas

- Toda la lógica vive en `src/`. El notebook solo importa, ejecuta y grafica.
- Sin look-ahead: la señal en t solo usa datos hasta t y se ejecuta en t+1.
- El archivo de test (`btc_project_test.csv`) se usa una sola vez, al final.
- Claude no hace commits: los hace el equipo.
- Las pruebas (`python -m pytest -q`) deben pasar antes de dar un paso por terminado.

## Regla de estilo

- Código simple: funciones cortas, nombres claros, nada rebuscado.
- Comentarios cortos en español que digan qué hace cada bloque y por qué.
- Docstrings en español con palabras sencillas: qué hace, qué recibe, qué regresa.
- Textos (SPEC, README) en español sencillo y natural, usando los términos de clase:
  look-ahead, causalidad, señal en t y ejecución en t+1, costos de transacción,
  break-even, Calmar, walk-forward, meseta contra pico, sobreajuste.
