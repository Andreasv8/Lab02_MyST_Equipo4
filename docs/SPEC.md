# SPEC — Estrategia del Lab 02 (BTCUSDT 5 min, Nivel B)

Este documento describe la estrategia tal como está programada en `src/`.
Si el código y este texto no coinciden, es un error y hay que corregirlo.

## 1. Datos y timeframe

- Datos: velas de 5 minutos de BTCUSDT (`data/btc_project_train.csv` y `data/btc_project_test.csv`).
- Los indicadores **no** se calculan en 5 minutos. Se calculan en velas de **4 horas** armadas con
  `resample_ohlc` y se pasan de regreso a 5 minutos con `align_to_base` (`src/data.py`).
- **Sin look-ahead:** una vela de 4h solo se usa cuando ya cerró. La vela que empieza a las 08:00
  se conoce al cierre de la barra de 5 min de las 11:55, no antes.
- **Por qué 4h:** el costo de ida y vuelta es 0.25% (0.125% al abrir y 0.125% al cerrar). En 5 minutos
  el ATR es mucho más chico que ese costo: la comisión es más grande que el stop y ninguna operación
  puede ganar en promedio. En 4h el movimiento típico ya es varias veces el costo.

  ATR(14) / Close mediano en train (con `atr()` de `src/signals.py`):

  | Timeframe | ATR mediano | ATR / costo de ida y vuelta (0.25%) |
  |-----------|-------------|-------------------------------------|
  | 5 min     | 0.079%      | 0.3×                                |
  | 1 hora    | 0.51%       | 2.0×                                |
  | 4 horas   | 1.21%       | 4.8×                                |

## 2. Los tres votos

Cada indicador vota +1 (largo), -1 (corto) o 0 (no vota). Todos se calculan en velas de 4h.

| Voto | Familia | Regla |
|------|---------|-------|
| `vote_ema` | Tendencia | signo(EMA_rápida − EMA_lenta) |
| `vote_roc` | Momento | signo(ROC_n) |
| `vote_bb` | Volatilidad | +1 si %B > u; −1 si %B < 1 − u; 0 en otro caso |

%B es la posición del precio dentro de las bandas de Bollinger (0 = banda inferior, 1 = banda superior)
y u es un umbral entre 0.5 y 1. Durante el calentamiento (indicadores en NaN) el voto es 0.

## 3. Regla 2 de 3 y señal de entrada

Con L_t = número de votos +1 y S_t = número de votos −1:

```
estado_t = +1  si L_t >= 2 y ADX_t > umbral
estado_t = -1  si S_t >= 2 y ADX_t > umbral
estado_t =  0  en otro caso
```

El ADX **no vota**: es un filtro de fuerza de tendencia. Con un solo voto a favor no se abre.

La señal solo aparece cuando el estado cambia:

```
señal_t = estado_t  si estado_t != estado_(t-1)
señal_t = 0         si no cambió
```

Así no se vuelve a entrar en cada barra mientras dura la misma tendencia (eso pagaría comisión una y
otra vez). **Señal en t, ejecución en t+1:** la señal se calcula al cierre de la barra t de 5 min y la
orden se ejecuta al open de la barra t+1.

Código: `strategy_votes`, `confirmed_state`, `entry_signal` y `compute_strategy` en `src/signals.py`.

## 4. Salidas

Una posición se cierra por lo primero que pase:

- **Stop-loss** a `sl_mult · ATR(4h)` del precio de entrada.
- **Take-profit** a `rr · stop`, es decir a `rr · sl_mult · ATR(4h)` (en el motor: `tp_mult = rr · sl_mult`).
- **Holding máximo:** `max_holding` barras de 5 min (la barra de entrada cuenta como la 1); se sale al cierre.
- **Señal contraria:** si llega una señal opuesta a la posición, se cierra al open siguiente.

Convenciones:

- Si en la misma vela se tocan el SL y el TP, se toma el **SL** (convención conservadora).
- Si el open ya cruzó el SL o el TP (gap), la orden se llena **al open**, no al nivel del SL/TP.

El ATR y los múltiplos se fijan al entrar, con los valores de la barra de señal.

## 5. Sizing

Se arriesga una fracción `rho` del capital en cada operación:

```
unidades = rho · capital / (sl_mult · ATR)
```

Si el precio llega al stop, la pérdida es `rho · capital` (antes de costos).

**Sin apalancamiento:** si el nocional más la comisión de entrada pasa del capital disponible, las
unidades se recortan a lo que alcanza con el capital (`compute_sizing` en `src/backtest.py`).

## 6. Costos de transacción y break-even

- **Comisión:** 0.125% por lado, en cada apertura y en cada cierre, sobre el nocional (precio · unidades).
- **Slippage:** 0 en el escenario base. El efecto de costos más altos se mide con la curva de retorno
  neto contra nivel de costo (`cost_sensitivity` en `src/metrics.py`).
- **Borrow fee:** 0 (no se cobra por los cortos).

Cada trade guarda `entry_commission` y `exit_commission` en dólares (y `entry_slippage`,
`exit_slippage`), con los mismos montos que el motor descontó de la caja.

**Break-even:** con stop a `sl_mult · ATR` y take-profit a `rr` veces el stop, la tasa de acierto que se
necesita para no perder es `1 / (1 + rr)` sin costos. Con costos sube, porque cada operación paga
0.25% de ida y vuelta pase lo que pase.

## 7. Orden de eventos en cada barra

En cada barra t el motor (`backtest` en `src/backtest.py`) hace, en este orden:

1. **Open:** si el open ya cruzó el SL o el TP de la posición abierta (gap), se cierra al open.
2. **Open:** si la señal de t−1 es contraria a la posición, se cierra al open.
3. **Open:** si no hay posición y la señal de t−1 es ±1, se abre al open con el ATR de t−1.
4. **Intrabar:** se revisan SL y TP con el High y el Low de t (si se tocan los dos, gana el SL).
5. **Close:** si se cumplió el holding máximo, se cierra al cierre de t.

Al final de cada barra: `equity = cash + unidades · Close` (unidades negativas en un corto).

## 8. Parámetros iniciales θ0

Valores antes de optimizar (`THETA0` en `src/signals.py`). Las ventanas están en velas de 4h,
salvo `max_holding`, que está en barras de 5 min.

| Parámetro | Valor | Qué es |
|-----------|-------|--------|
| `ema_fast` | 12 | EMA rápida |
| `ema_slow` | 48 | EMA lenta |
| `roc_window` | 12 | ventana del ROC |
| `bb_window` | 20 | ventana de Bollinger |
| `bb_std` | 2 | desviaciones estándar de las bandas |
| `bb_threshold` | 0.7 | umbral u de %B |
| `adx_window` | 14 | ventana del ADX |
| `adx_threshold` | 20 | umbral del filtro ADX |
| `atr_window` | 14 | ventana del ATR |
| `sl_mult` | 2.0 | stop-loss en múltiplos del ATR |
| `rr` | 3.0 | take-profit en múltiplos del stop |
| `max_holding` | 2016 | barras de 5 min (7 días) |
| `rho` | 0.01 | fracción del capital que se arriesga |

## 9. Capital y lados

- Capital inicial: 1,000,000.
- Se opera en los dos lados: posiciones largas y cortas. Nunca hay dos posiciones abiertas a la vez.

## 10. Detección de régimen

**Método: reglas** (`rule_regimes` en `src/regimes.py`). K-means y HMM se calculan solo para comparar.

Variables, en una ventana móvil de 1 semana de barras de 5 min (causales: en t solo usan datos hasta t):

- **volatilidad**: desviación estándar de los log-rendimientos, anualizada;
- **trend_r2**: R² de una recta ajustada al log-precio (qué tan "en línea recta" se mueve);
- **autocorrelación** de los rendimientos a 1 barra.

El régimen se actualiza **cada hora** (en las barras hh:00) y se mantiene el resto de la hora:

1. **crisis** si la volatilidad pasa del percentil 90 de la volatilidad de ajuste;
2. **trend** si no es crisis y trend_r2 > 0.5;
3. **mean_reversion** en otro caso.

El umbral de crisis se ajusta con **ventana expansiva**: en cada ventana del walk-forward se usa toda la
historia desde el inicio de los datos hasta el final del ajuste, nunca datos posteriores.

**Por qué reglas y no K-means/HMM.** Validación medida con ajuste del 2022-06-01 al 2023-05-14 y fuera de
muestra del 2023-05-15 al 2023-12-31 (dentro del archivo de train):

| Método | Periodo | Duración media (h) | Transiciones/mes | Silhouette | % crisis | % trend | % mean_rev |
|--------|---------|-------------------:|-----------------:|-----------:|---------:|--------:|-----------:|
| Reglas | ajuste | 89.8 | 7.9 | 0.31 | 10.0 | 36.9 | 53.1 |
| Reglas | fuera de muestra | 101.0 | 7.0 | 0.37 | 3.0 | 31.4 | 65.6 |
| K-means | ajuste | 88.8 | 8.0 | 0.32 | 28.5 | 35.1 | 36.4 |
| K-means | fuera de muestra | 95.7 | 7.4 | 0.21 | 70.4 | 13.0 | 16.6 |
| HMM | ajuste | 102.5 | 6.9 | 0.31 | 26.3 | 40.9 | 32.8 |
| HMM | fuera de muestra | 175.9 | 4.0 | 0.20 | 82.5 | 9.4 | 8.1 |

- Los tres métodos cumplen la duración mínima (> 12 h) y cambian de régimen pocas veces al mes.
- Fuera de muestra, K-means y HMM clasifican entre el 70% y el 82% del tiempo como "crisis", aunque la
  volatilidad de ese grupo ya es casi igual a la de "trend". Los nombres dejan de tener sentido: es
  sobreajuste a la forma de los datos de ajuste (sobre todo a la autocorrelación, que fuera de muestra
  se volvió negativa).
- Las reglas mantienen el orden de los nombres fuera de muestra (crisis sigue siendo lo más volátil) y su
  silhouette incluso mejora un poco.

**Silhouette < 0.4 en los tres métodos.** No se llega al objetivo de 0.4. La razón es que BTC no salta
de un régimen a otro: cambia de forma **gradual**, así que muchas horas quedan en la frontera entre dos
regímenes y los grupos se traslapan. Lo tomamos en cuenta: el régimen se usa para ajustar parámetros y
para salir en crisis, no como una clasificación perfecta.

## 11. Reglas de transición

Qué pasa con las posiciones cuando cambia el régimen:

- **R1.** Una posición conserva el SL, el TP y el holding máximo con los que entró, aunque después
  cambie el régimen.
- **R2.** Las entradas nuevas usan los parámetros del régimen vigente en la barra de señal (t) y se
  ejecutan en t+1.
- **R3.** Si el régimen cambia a crisis, la posición abierta se cierra en el open de la siguiente barra
  (`force_exit` en `backtest`, motivo `regime_exit`, paga comisión). Igual que la señal: se decide en t
  y se ejecuta en t+1, sin look-ahead.
- **R4.** Entre tendencia y reversión la posición se mantiene hasta su SL, TP, holding máximo o señal
  contraria.
- **R5.** Si en una ventana de entrenamiento del walk-forward un régimen no llega al mínimo de
  operaciones, ese régimen usa los parámetros globales de esa ventana (así no se optimiza con
  muy pocos datos).
