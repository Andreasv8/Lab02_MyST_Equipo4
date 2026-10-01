# SPEC — Lab 02: Trading Strategy Skeleton (NVDA)

## 1. Universe and frequency
- Activo: NVDA, barras diarias (1d), yfinance (`data/download_data.py` → `data/NVDA_daily.csv`).
  Rango 2021-09-15 a 2026-09-15, 1,255 barras. Precios ajustados por el split 10:1 de jun-2024.
- Split por bloques contiguos (`src/splits.py`, sin aleatorizar):

| Periodo | Fechas | Barras |
|---|---|---|
| Train | 2021-09-15 a 2024-09-14 | 754 |
| Test | 2024-09-15 a 2025-09-14 | 249 |
| Validation | 2025-09-15 a 2026-09-15 | 252 |

- Los indicadores se calculan sobre la serie completa (son causales) y luego se recortan por periodo.
  Warm-up: ADX(14) es el último en estar disponible (2021-10-21, barra 27); el train efectivo de la
  estrategia empieza ahí (728 barras). Test y validation no pierden barras.
- Validation no se usa para nada (ni selección, ni ajuste, ni gráficas) hasta la evaluación final.

## 2. Features
Selección: 17 candidatos estacionarios, correlación de Pearson **solo en train**; si |r| ≥ 0.70
se conserva uno del par (`data/indicator_analysis.py`, `data/indicator_corr.png`).

| Indicador | Ventana | Familia | Uso |
|---|---|---|---|
| ROC | 10 | momentum | dirección: cambio % del precio en 10 días |
| CMF | 20 | volumen | dirección: acumulación/distribución ponderada por volumen |
| ADX | 14 | tendencia | filtro de fuerza (sin dirección) |
| ATR | 14 | volatilidad | solo SL/TP y sizing; no entra en la selección ni en la señal |

- Evidencia en train: |r| ROC–CMF = 0.54, ROC–ADX = 0.25, CMF–ADX = 0.14 → **max |r| = 0.54 < 0.70**.
- Set anterior descartado: RSI–MFI = 0.79, RSI–ROC = 0.83, MFI–ROC = 0.74, los tres > 0.70.
- No se usa la estrategia de referencia del curso (cruce SMA 20/50 + RSI 30/70 + MACD por mayoría).

## 3. Entry rule (confluencia, AND)
Con valores al cierre de la barra t:

    señal_t = +1  si ROC_t > 0  y CMF_t > 0  y ADX_t > 25
    señal_t = −1  si ROC_t < 0  y CMF_t < 0  y ADX_t > 25
    señal_t =  0  en cualquier otro caso (incluye NaN de warm-up y valores exactamente 0)

- 0 en ROC y CMF es el cambio de signo: precio arriba/abajo de hace 10 días y flujo de volumen
  comprador/vendedor neto. ADX > 25 es el umbral de Wilder (1978) para "mercado en tendencia".
- Los umbrales son valores estándar de la literatura; no se optimizan ni se eligieron con resultados.

## 4. Exit rule
- Sobre el precio de entrada crudo P_e (open de entrada, sin costos) y ATR_t de la barra de señal:
  long SL = P_e − 2·ATR_t, TP = P_e + 3·ATR_t; short SL = P_e + 2·ATR_t, TP = P_e − 3·ATR_t.
  Risk-reward r = 3/2 = 1.5.
- Señal opuesta (señal_t = −lado actual): cierra al open de t+1 y abre en la nueva dirección
  en ese mismo open. Una señal 0 no cierra la posición.
- Holding máximo 10 barras; la barra de entrada cuenta como 1 → cierre al Close de entry_bar + 9.
- Tie intrabar (la barra toca SL y TP): se ejecuta SL.
- Gap: si el open ya cruzó el SL o el TP, la salida se llena al open, no al nivel.

## 5. Sizing
Risk parity por ATR con ρ = 1% del capital y apalancamiento máximo 1:

    Q = floor( 0.01 · V_t / (2 · ATR_t) )
    si Q · P_e · (1 + c) > V_t:  Q = floor( V_t / (P_e · (1 + c)) )   y el trade se marca truncated = True
    si Q ≤ 0: no se opera

- c = costo por lado = comisión + spread/slippage = 0.15% (sección 6), para que el nocional más
  el costo de entrada no exceda el capital.
- V_t = capital (cash) realizado al momento de la entrada; no hay margen, el tope aplica igual a
  long y short (nocional ≤ V_t). P_e = open de t+1.
- El tope se activa solo si ATR_t/P_e < ~0.5%; en train el mínimo de ATR/Close es 2.4%, así que
  se espera que casi nunca se active, pero la regla y el registro existen.

## 6. Costs
Por lado (entrada y salida), como fracción del nocional:

| Concepto | Valor | Fuente / justificación |
|---|---|---|
| Comisión | 0.10% | IBKR Pro Fixed cobra $0.005/acción (máx. 1% del nocional), que a los precios reales de NVDA es < 0.01%; 0.10% es una cota conservadora y en porcentaje no depende del ajuste por split |
| Spread + slippage | 0.05% | La mitad del spread cotizado de NVDA es ~1 centavo (< 0.01%); 0.05% deja margen para el slippage al open |
| Borrow fee (solo shorts) | 0.50% anual | NVDA es general collateral (fácil de pedir prestado); cargo = 0.005 · Q · P_e · días / 360, con P_e crudo y días = fecha de salida − fecha de entrada (calendario); un short abierto y cerrado el mismo día paga 0 (no hay overnight) |

- Sin impacto de mercado: el nocional máximo es V_t ≈ $100k, frente a un volumen diario mediano
  en train de ~463M acciones (ajustadas) ≈ $13B → < 0.001% del volumen diario.

## 7. Conventions
- La señal y el ATR se calculan al cierre de t; la ejecución es al open de t+1.
- Orden de eventos dentro de cada barra:
  1. Open: salida por gap que cruzó SL/TP (se llena al open).
  2. Open: salida por señal opuesta (al open).
  3. Open: entrada nueva si no hay posición.
  4. Intrabar: SL/TP con High/Low (tie → SL), incluida la barra de entrada.
  5. Close: salida por holding máximo.
- Una sola posición abierta a la vez. Las entradas solo ocurren en el open, así que una
  salida en los pasos 4 o 5 no se reemplaza hasta el open de t+1.
- La transición long ↔ short en la misma barra solo ocurre por señal opuesta (paso 2 + paso 3).

## Break-even win rate
- Sin costos: p* = 1/(1+r) = 1/2.5 = **40%**. La estrategia debe superarlo.
- Con costos: ganancia neta = r·R − C, pérdida neta = −R − C, con R = 2·ATR·Q (riesgo) y C =
  costo ida y vuelta. Si se igualan a cero, p* = (1 + k)/(1 + r), con k = C/R.
- Ejemplo con la mediana de train (ATR/Close = 4.37%): C = 2 × 0.15% = 0.30% del precio,
  R = 8.74% → k = 0.034 → p* = 1.034/2.5 ≈ **41.4%**. En el percentil 10 de ATR/Close (2.97%),
  k = 0.051 → p* ≈ 42.0%. El borrow fee de un short de 10 barras (~14 días) suma ~0.02% → k + 0.002.
