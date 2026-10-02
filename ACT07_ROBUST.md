# Act 07 v2 (robusta) — Pre-registro

- Fecha: 2026-10-02
- Rama: `feat/lab02-act07-robust`
- Estado: **pre-registro. Nada de lo descrito aquí se ha corrido todavía.**

Este documento fija el diseño de la v2 ANTES de escribir código o ver resultados.
Lo que se decida después de ver test no cambia este diseño: se documenta como v3.

---

## 1. Motivo

La v1 (`src/optimization.py`, `Act_07_regimes.ipynb`) optimizó con Optuna (500 trials) los 6
parámetros de θ = (roc_window, cmf_window, adx_threshold, sl_mult, rr, max_holding) para cada
uno de los 3 regímenes HMM (**18 parámetros**) además de θ* único, evaluando J en un solo periodo
(todo train). Mostró señales de overfitting:

- Caída fuerte de J entre train y test.
- θ* pegado a la restricción MIN_TRADES (la búsqueda premia configuraciones con los trades
  justos para ser válidas).
- adx_threshold en el borde del rango de búsqueda.

Se aplica lo visto en clase:

1. **Menos parámetros**: la señal queda fija; solo se optimiza la salida.
2. **Walk-forward**: J se mide fuera de muestra, no en el mismo tramo con el que se ajusta.
3. **Mesetas en lugar de picos**: el θ final es la mediana de una región buena, no el mejor trial.

---

## 2. Qué queda fijo

| Componente | Valor | Fuente |
|---|---|---|
| Señal | ROC(10) > 0, CMF(20) > 0, ADX(14) > 25 (y simétrico en short) | SPEC §3 |
| ATR | 14 | SPEC §2 |
| Sizing | risk parity ρ = 1%, apalancamiento ≤ 1 | SPEC §5 |
| Costos | 0.15% por lado + borrow 0.50% anual en shorts | SPEC §6 |
| Features de régimen | volatility, trend_r2, autocorr_1 (ventana 63) | `src/regimes.py` |
| Clasificador | HMM gaussiano de 3 estados, etiqueta **filtrada** (causal) | `src/regimes.py` |
| Semilla | 42 (Optuna) | CLAUDE.md raíz |

---

## 3. Qué se optimiza

| Parámetro | Tipo | Rango |
|---|---|---|
| sl_mult | float | [1, 3] |
| rr | float | [1, 3] |
| max_holding | int | [5, 20] |

tp_mult = sl_mult · rr.

- **θ*_régimen robusto**: un estudio por régimen HMM (crisis, trend, mean_reversion), 3
  parámetros cada uno → **9 parámetros** (vs 18 en v1). En el estudio del régimen j solo se
  permiten entradas cuya barra de señal tenga etiqueta filtrada == j.
- **θ* robusto**: mismo esquema sin régimen → **3 parámetros**. Sirve para comparar.

**Regla a priori por régimen**: si el mejor J_j ≤ 0, o ningún trial alcanza 10 trades, en el
régimen j **no se opera**.

---

## 4. Walk-forward (ventana creciente, dentro de train)

"inicio" = 2021-09-15 (inicio de train; el warm-up de indicadores y features se descarta como en v1).

| Bloque | Ajuste | Evaluación (OOS) |
|---|---|---|
| 1 | inicio → 2023-03-14 | 2023-03-15 → 2023-09-14 |
| 2 | inicio → 2023-09-14 | 2023-09-15 → 2024-03-14 |
| 3 | inicio → 2024-03-14 | 2024-03-15 → 2024-09-14 |

El último tramo de evaluación termina en el fin de train (2024-09-14). Test y validation no se
usan en ninguna parte de la búsqueda.

---

## 5. Anti look-ahead

En cada bloque se **reajustan solo con datos ≤ fin de su tramo de ajuste**:

- el scaler de las features de régimen (`fit_scaler`),
- el umbral p90 de volatilidad (`fit_rule_thresholds`),
- el HMM, con la misma cascada de v1 (`fit_hmm`: full → diag → diag+prior, semillas 0..9,
  mayor log-likelihood, duración esperada ≥ 5 barras).

Las etiquetas del tramo de evaluación son las filtradas (forward) de ese modelo: en la barra t
solo usan datos hasta t. **Nunca se usa un modelo ajustado con datos posteriores al tramo
evaluado.**

---

## 6. Función objetivo y búsqueda

- **J = Calmar ratio** (`calmar_ratio`, CAGR / |max DD|) de la equity fuera de muestra: los 3
  tramos de evaluación concatenados.
- **Optuna**: 200 trials por estudio = 100 random (startup) + 100 TPE,
  `TPESampler(n_startup_trials=100, seed=42)`.
- Mínimo de trades (contados en los 3 tramos OOS juntos): 10 por régimen, 20 para θ* robusto.
  Si no se alcanza, J = −inf.
- Cada trial evalúa el mismo θ en los 3 bloques, cada bloque con sus propios modelos de régimen.

---

## 7. Selección final (meseta)

- Se ordenan los trials por J y se toma el **top 10%** (20 de 200).
- **θ final = mediana por parámetro** de ese top 10%.
- Se reporta también el J walk-forward del θ mediana, para comprobar que cae en la meseta.

---

## 8. Evaluación final

**Una sola corrida en test** (2024-09-15 → 2025-09-14), con los modelos de régimen ajustados con
todo train (igual que v1). Se comparan:

| Estrategia | Origen |
|---|---|
| Buy & hold | referencia |
| θ0 | SPEC (ROC 10, CMF 20, ADX 25, sl 2, rr 1.5, holding 10) |
| v1 θ* | Act 07 v1 |
| v1 θ*_régimen | Act 07 v1 |
| v2 θ* robusto | este documento |
| v2 θ*_régimen robusto | este documento |

- Métricas: las de `summarize` (`src/metrics.py`), como en Act 06 y v1.
- **Se reportan todas aunque v2 salga peor.**
- **Validation no se carga.**

### Qué se reporta además

- θ final de cada estudio y regímenes apagados por la regla a priori.
- Distribución del top 10% de cada parámetro (¿meseta o pico?).
- J walk-forward (OOS) vs J en test.
- Cascada HMM elegida y centroides de cada bloque.

### Criterios de lectura (fijados ahora)

v2 se considera **más robusta** que v1 si:

1. la caída J walk-forward → J test de v2 es menor que la caída J train → J test de v1, y
2. ningún parámetro del θ final queda en el borde de su rango.

Un mejor Calmar en test **no** es requisito ni se persigue.

### Lo que NO se hará después de ver test

Re-optimizar, cambiar rangos, bloques, semillas, número de trials, el % de la meseta o estos
criterios. Cualquier cambio se documenta aparte como v3.

---

## 9. Decisiones derivadas

No vienen explícitas en el pedido original; se fijan aquí para que no se decidan después de
ver resultados.

1. **Estudios separados por régimen** (3 estudios × 3 parámetros), como en v1; no un estudio
   conjunto de 9.
2. **θ* robusto** usa mínimo **20 trades** OOS (igual que MIN_TRADES de v1).
3. El mínimo de trades se cuenta sobre los **3 tramos de evaluación juntos**.
4. **Construcción de la equity OOS**: en cada bloque el backtest corre con datos ≤ fin del
   tramo de evaluación, pero solo se permiten **entradas** con barra de señal dentro del tramo
   de evaluación (y, por régimen, con etiqueta filtrada == j). Cada tramo arranca plano; una
   posición abierta al final del tramo queda valuada a mercado (convención actual del motor).
   Los rendimientos diarios de los 3 tramos se encadenan en una sola equity y sobre ella se
   calcula Calmar.
5. **Meseta**: el top 10% son los 20 mejores trials con J finito; si hay menos de 20 finitos se
   usan los disponibles y se reporta cuántos. `max_holding` (mediana) se redondea al entero más
   cercano.
6. **Nombres de estados por bloque** con `name_states` (mayor volatilidad = crisis, etc.). Como
   el HMM se reajusta, un "trend" del bloque 1 puede no ser idéntico al del bloque 3; por eso se
   reportan centroides y cascada de cada bloque.
7. **En test** se usan scaler, p90 y HMM ajustados con todo train (= `fit_regime_models` de v1),
   que respeta la regla anti look-ahead.
8. **Implementación prevista**: módulo nuevo `src/walk_forward.py` con sus tests y notebook
   `Act_07_robust.ipynb`. El código y el notebook de v1 no se modifican.
9. **Redondeo de max_holding** (aclarada antes de correr nada): con 20 trials la mediana es el
   promedio del 10º y el 11º y puede quedar en x.5; se redondea **half-up**,
   `floor(x + 0.5)` (12.5 → 13).

---

## 10. Aclaraciones antes de correr

Fijadas el 2026-10-02, después del walk-forward y **antes de evaluar test**. Precisan cómo se
calculan los criterios de lectura de la sección 8; no cambian ni el diseño ni los θ.

10. **J walk-forward de v2 θ*_régimen**: Calmar de la equity OOS encadenada de la estrategia
    **combinada** (crisis y trend con su θ de la meseta, mean_reversion apagado), con los modelos
    de régimen de cada bloque. Es el análogo del J en train de v1 θ*_régimen (también combinada).
11. **Caída del criterio 1**: v2 debe caer menos que v1 en **absoluta** (J_ref − J_test) **y** en
    **relativa** ((J_ref − J_test) / J_ref). Si solo una se cumple, el criterio no se cumple.
    J_ref = J walk-forward en v2 y Calmar en train en v1; J_test = Calmar en test.
12. **Borde del criterio 2**: un parámetro está en el borde si su distancia al límite más cercano
    es ≤ **10% del ancho de su rango** (0.2 en sl_mult y rr; 1.5 en max_holding).
