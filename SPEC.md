# SPEC — Lab 02: Trading Strategy Skeleton

## 1. Universe and frequency

Activo: NVDA (NVIDIA Corp), barras diarias (1d), datos de yfinance. Rango: 5 años, 2021-09-15 a 2026-09-15 (~1,255 barras).

División temporal por chunks contiguos (sin aleatorizar), 60/20/20 aprox.:
- Train: 2021-09-15 a 2024-09-14 (3 años)
- Test: 2024-09-15 a 2025-09-14 (1 año)
- Validation: 2025-09-15 a 2026-09-15 (1 año)

Las fechas viven en `src/splits.py`. Los indicadores se calculan sobre la serie completa (son causales) y después se recortan por periodo, para no perder el warm-up de 50 barras al inicio de test y validation.

## 2. Features

Indicadores seleccionados a partir de una matriz de correlación de 12 candidatos (`data/indicator_analysis.py`), descartando pares con correlación > 0.85. La matriz se calcula **solo sobre train**, para que la selección no vea test ni validation.

| Indicador | Tipo | Familia | Uso |
|-----------|------|---------|-----|
| SMA(50) | overlay | tendencia | filtro direccional |
| RSI(14) | oscilador | momentum | velocidad del cambio de precio |
| MFI(14) | oscilador | volumen-momentum | RSI ponderado por volumen |
| ROC(10) | oscilador | momentum | cambio porcentual a 10 períodos |
| ATR(14) | — | volatilidad | sizing, SL/TP, y régimen de pesos adaptativos (no genera señales directamente) |

## 3. Entry rule

Score ponderado adaptativo por régimen de volatilidad.

Normalización de osciladores a [-1, 1]:
```
RSI_norm = (RSI - 50) / 50
MFI_norm = (MFI - 50) / 50
ROC_norm = clip(ROC / 10, -1, 1)
```

Pesos adaptativos según ATR:
```
Si ATR_t > media(ATR):  w_rsi=0.25, w_mfi=0.50, w_roc=0.25
Si no:                  w_rsi=0.45, w_mfi=0.25, w_roc=0.30
```

media(ATR) se calcula como expanding mean hasta la barra actual (sin look-ahead).

Score: Z_t = w_rsi × RSI_norm + w_mfi × MFI_norm + w_roc × ROC_norm

Filtro de consenso: si RSI_norm, MFI_norm y ROC_norm no comparten el mismo signo, señal = 0 (flat).

Si algún indicador normalizado es exactamente 0, se considera sin dirección y el consenso falla (señal = flat).

Señal final:
- Si |Z_t| > 0.3 y pasa consenso y Close > SMA(50): señal = +1 (long)
- Si |Z_t| > 0.3 y pasa consenso y Close < SMA(50): señal = -1 (short)
- Cualquier otro caso: señal = 0 (flat)

## 4. Exit rule

- Stop-loss: 2 × ATR(14) desde el precio de entrada
- Take-profit: 3 × ATR(14) desde el precio de entrada
- Risk-reward ratio: 1.5:1
- Señal opuesta: cierra posición actual y abre en la nueva dirección
- Holding máximo: 10 barras. Si no toca SL ni TP, cierra al cierre de la barra 10
- La barra de entrada cuenta como barra 1, por lo que la posición se cierra al cierre de la barra 10 desde la entrada (entry_bar + 9).
- Tie intrabar (High ≥ TP y Low ≤ SL en la misma barra): se ejecuta SL (conservador)
- Gap: si el open ya cruzó el SL (o el TP), la salida se llena al open, no al nivel del SL/TP

## 5. Sizing

Risk-parity por ATR. Presupuesto de riesgo: ρ = 1% del capital por trade.
```
Q = (ρ × V_t) / (2 × ATR_t)
```
Donde V_t es el capital disponible y 2×ATR es la distancia al SL.

Q se trunca a entero (no se compran fracciones). Si Q ≤ 0, el trade se omite.

## 6. Costs

- Comisión: 0.1% por transacción (entrada y salida)
- Slippage: 0.05% por transacción
- Borrow fee: 0 (simetría long/short)

Fuente: comisiones estándar de brokers retail (Interactive Brokers, similar). Slippage conservador para activo líquido.

## 7. Conventions

- Señal al cierre de barra t, se ejecuta al open de barra t+1. El ATR usado para SL/TP y sizing también es el de t.
- Orden de eventos dentro de cada barra:
  1. Open: salida por gap que cruzó SL/TP (se llena al open).
  2. Open: salida por señal opuesta (al open).
  3. Open: entrada nueva si no hay posición.
  4. Intrabar: SL/TP con High/Low (tie → SL).
  5. Close: salida por holding máximo.
- Ties intrabar: SL tiene prioridad sobre TP
- Una sola posición abierta a la vez. Como las entradas solo ocurren en el open, una posición cerrada intrabar o al cierre de t no se reemplaza hasta el open de t+1.
- Transición directa long a short (o viceversa) permitida en la misma barra, solo por señal opuesta (cierre y apertura al mismo open)

## Break-even win rate

Con ratio 1.5:1: WR_be = 1 / (1 + 1.5) = 0.40 = 40%. La estrategia debe superar este umbral.
