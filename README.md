# Lab02_MyST_Equipo4

## Integrantes

- Andrea Santoyo
- Isabela Torres

## Nivel de alcance

B (BTCUSDT 5 minutos, con detección de régimen).

## Descripción

Estrategia de trading para BTCUSDT en velas de 5 minutos con tres indicadores calculados en velas de
4 horas (EMA, ROC y %B de Bollinger) y una regla de confirmación **2 de 3** con filtro de ADX; la señal
se calcula en t y se ejecuta en t+1, con comisión de 0.125% por lado. Es **Nivel B**: un régimen de
mercado por reglas (crisis, tendencia, reversión a la media) cambia los parámetros y cierra posiciones
al entrar a crisis. Los parámetros se optimizan con un **walk-forward** de 73 ventanas (train de 1 mes,
test de 1 semana, 29,200 configuraciones); el walk-forward, la evaluación final y los análisis de
robustez se pre-registraron en `docs/SPEC.md` antes de correrlos.
**Resultado principal:** la estrategia no tiene ventaja real; fuera de muestra pierde −33.7% con
régimen y −9.7% solo global mientras el buy & hold gana +84.5%, y la ventaja de train es ajuste a la
muestra.

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

Tarda ~1.5 minutos con 10 núcleos (el walk-forward corre en paralelo). Genera todas las tablas en
`docs/tables/` (CSV) y las figuras en `docs/figures/` (PNG), y al final imprime un resumen. Incluye la
evaluación final sobre `btc_project_test.csv`: el resultado es el mismo en cada corrida (todo es
determinista con las semillas de abajo).

El análisis y las respuestas a las preguntas del PDF están en `notebooks/analysis.ipynb`, que solo lee
esas tablas y figuras.

## Pruebas

```bash
python -m pytest -q
```

Incluye las 4 pruebas obligatorias del PDF: causalidad de la señal (`tests/test_causality.py`), regla
de confirmación (`tests/test_confirmation.py`), contabilidad del motor (`tests/test_accounting.py`) y
causalidad del régimen (`tests/test_regimes.py::test_regime_label_does_not_change_with_future_data`).

## Semilla

| Constante | Archivo | Uso |
|---|---|---|
| `SEED = 42` | `src/optimize.py` | Semilla del `TPESampler` de Optuna. Cada ventana del walk-forward usa `SEED + número de ventana`. |
| `SEED = 42` | `src/regimes.py` | `random_state` de K-means (solo para comparar regímenes). |
| `HMM_SEEDS = range(10)` | `src/regimes.py` | El HMM (solo para comparar) se ajusta con las semillas 0–9 y se queda el de mayor log-likelihood. |

## Resultados principales

| Periodo | Curva | Retorno | Sharpe | Máx. drawdown | Trades |
|---|---|---:|---:|---:|---:|
| Walk-forward OOS (train, 2022-08-01 a 2023-12-25) | Con régimen | −33.7% | −1.91 | −34.9% | 162 |
| | Solo global | −9.7% | −0.48 | −21.6% | 137 |
| | Buy & hold | +84.5% | 1.31 | −37.9% | — |
| Test (2024-05-02 a 2024-06-03) | Con régimen | +4.8% | 3.55* | −4.3% | 8 |
| | Solo global | +1.7% | 1.89* | −3.3% | 6 |
| | Buy & hold | +16.4% | 4.52* | −7.9% | — |

\* Sharpe anualizado desde un solo mes: se infla (el Sharpe mensual con régimen es ~1.0). Con 8 trades
el mes de test no alcanza para concluir nada.

Otros números clave: la ventaja de train no sobrevive (+1.76% por semana en train contra −0.53% y
−0.12% fuera de muestra); el break-even de costos de la versión solo global es 5.6 pb por lado contra
una comisión real de 12.5 pb; la sensibilidad ±20% muestra un pico, no una meseta.

## Limitaciones

- **Costos:** la comisión real (12.5 pb por lado) es más del doble del break-even (5.6 pb) y el
  impacto de mercado estimado suma ~2 pb más por lado.
- **Parámetros inestables:** pico en la sensibilidad y ventaja de train que no sobrevive fuera de muestra.
- **Ejecución idealizada:** llenado completo al open, sin latencia, slippage 0 y sin costo de
  financiamiento de los cortos.
- **Pocos trades:** con 1 mes de train hay ~15 señales; los mínimos por estudio (5 global, 3 por
  régimen) son bajos, y en crisis se usaron los parámetros globales en 59 de 73 ventanas.
- **Datos:** el volumen falta en ~48% de las barras; el archivo de test trae un día suelto y un hueco
  de 122 días, así que indicadores y regímenes mezclan diciembre de 2023 con mayo de 2024; la primera
  ventana solo tiene 30 días de calentamiento.
- **Régimen:** ningún método llega a silhouette > 0.4 (BTC cambia de régimen de forma gradual).

## Uso de herramientas de IA

Usamos Claude (Anthropic) en dos formas:

- **Claude Code**, para escribir código y pruebas bajo nuestras instrucciones: los módulos de `src/`
  (señal, motor de backtest, métricas, regímenes, optimización y walk-forward, gráficas), `main.py`,
  las pruebas de `tests/` y el armado del notebook.
- **Claude**, para planear cada paso, revisar los planes antes de ejecutarlos y revisar los resultados.
  También ayudó a redactar la documentación (`docs/SPEC.md`, este README y el texto del notebook).

Todas las decisiones de diseño las tomamos nosotras: la estrategia y sus indicadores, la regla 2 de 3,
las reglas de transición entre regímenes y el pre-registro de los análisis antes de correrlos.
Revisamos cada cambio antes de hacer commit, y las dos podemos explicar cualquier línea del código.

## Estructura del repositorio

```
.
├── README.md
├── main.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── btc_project_train.csv
│   └── btc_project_test.csv
├── docs/
│   ├── SPEC.md          (especificación y pre-registro)
│   ├── presentacion.pdf (presentación)
│   ├── reporte.md       (reporte ejecutivo)
│   ├── reporte.pdf      (reporte ejecutivo en PDF)
│   ├── figures/         (figuras que genera main.py)
│   └── tables/          (tablas que genera main.py)
├── notebooks/
│   └── analysis.ipynb   (análisis y figuras; sin lógica)
├── src/
│   ├── __init__.py
│   ├── backtest.py      (motor por eventos, costos y sizing)
│   ├── data.py          (carga de datos y velas de 4h sin look-ahead)
│   ├── metrics.py       (métricas, retornos, diagnóstico e impacto de mercado)
│   ├── optimize.py      (Optuna, walk-forward, robustez y evaluación final)
│   ├── plots.py         (figuras)
│   ├── regimes.py       (variables, clasificadores y validación del régimen)
│   └── signals.py       (indicadores, votos y regla 2 de 3)
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
