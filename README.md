# Pronóstico de demanda minorista (M5, Walmart): ¿le gana el aprendizaje automático a un modelo simple?

Pronóstico de ventas diarias a 28 días para 600 series de producto y tienda del dataset M5 de Walmart. Se compara una escalera de modelos, desde baselines hasta LightGBM y una red neuronal, con un protocolo de evaluación fijado de antemano, dos métricas (RMSSE y MAE) e intervalos de confianza por bootstrap entre series para saber qué diferencias son reales. Todo el entrenamiento se hizo en CPU.

**Autor:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

## Resumen

- **Resultado principal:** en el test, LightGBM (RMSSE 0,768; MAE 1,011) y un modelo estadístico simple, suavizado exponencial con índice semanal (0,769; 1,008), quedan prácticamente empatados. La red neuronal (0,772; 1,023) y la media móvil de 28 días (0,782; 1,059) les siguen de cerca.
- **El aprendizaje automático no mejora de forma confiable al modelo simple.** Según el bootstrap, la ventaja de LightGBM sobre el modelo estadístico en MAE (validación) desaparece en test, y en RMSSE no hay diferencia distinguible del ruido. La red neuronal rinde peor que LightGBM en validación.
- **Lo único que se sostiene en las dos ventanas:** el modelo estadístico con índice semanal supera a la media móvil en RMSSE, por un 1,2 % y un 1,7 %.
- **La demanda es muy esporádica:** el 68 % de las observaciones diarias son cero y el 91 % de las series tiene demanda intermitente o irregular. Un RMSSE menor a 1 no es un buen resultado por sí mismo: pronosticar siempre cero da 0,94 y 0,99.
- **RMSSE y MAE no siempre ordenan igual los modelos**, por eso se reportan juntos.

## Datos

| | |
|---|---|
| Fuente | *M5 Forecasting - Accuracy*, la competencia organizada por la Universidad de Nicosia; copia en Zenodo (registro 12636070), licencia CC BY 4.0 |
| Series | 30.490: 3.049 productos en 10 tiendas de 3 estados de EE. UU. (CA, TX y WI) |
| Período | 29 de enero de 2011 al 22 de mayo de 2016 (1.941 días) |
| Archivos | ventas diarias (`sales_train_evaluation.csv`), calendario con eventos y días SNAP, y precios semanales |
| Muestra | 600 series: 60 al azar por tienda, semilla 42. Reproduce la composición del total (FOODS 48,0 % contra 47,1 %) y la intermitencia (68,2 % de días sin ventas contra 68,0 %) |

La competencia evaluaba 42.840 series jerárquicas; aquí se usa solo el nivel más desagregado (producto y tienda).

**Qué mostró la exploración** (notebook 01):
- **Estacionalidad semanal fuerte:** sábado (1,21) y domingo (1,20) venden alrededor de un 20 % más que el promedio, y de martes a jueves, un 12 % menos. La estacionalidad anual es mucho menor (de 0,96 a 1,04).
- **Cierre de Navidad:** el 25 de diciembre las tiendas prácticamente no venden. Otros eventos, como Acción de Gracias (0,73) y Año Nuevo (0,82), venden menos que un día normal. Los días SNAP venden más en los tres estados.
- **Intermitencia:** el 72,7 % de las series es intermitente y el 18,3 % irregular (clasificación de Syntetos, Boylan y Croston). Esas series reúnen el 57,6 % de las unidades.
- **Series que empiezan tarde:** el 77,1 % tiene su primera venta después del primer día y el 44,0 % tiene al menos un año de ceros iniciales. Esos ceros no son demanda nula, por eso las métricas cuentan el pasado desde la primera venta.
- **Precios:** cambian en solo el 1,2 % de las semanas.

## Protocolo de evaluación

Se fijó en el notebook 01, antes de entrenar ningún modelo.

- **Horizonte:** 28 días, desde un origen y sin usar ventas posteriores a él. Se asume que el calendario y los precios del período a pronosticar se conocen de antemano, como ocurre en M5.
- **Ventanas:** validación del 28 de marzo al 24 de abril de 2016 (días 1.886 a 1.913) y test del 25 de abril al 22 de mayo (días 1.914 a 1.941). Los hiperparámetros se eligen en validación; el modelo elegido se reentrena con datos hasta el origen del test y se evalúa en el test una sola vez. La validación no tiene eventos; el test tiene cuatro menores. Ninguna incluye los feriados de mayor impacto.
- **RMSSE (error escalado):** error cuadrático medio del pronóstico dividido por el de un pronóstico ingenuo (repetir el día anterior) en el pasado de la misma serie, con raíz cuadrada. Se cuenta desde la primera venta y se promedia sin ponderar entre series.
- **MAE:** error absoluto medio en unidades por serie y día.

## Modelos

| Modelo | Descripción |
|---|---|
| Cero | Pronostica siempre 0 (referencia para entender las métricas) |
| Ingenuo | Repite la última venta |
| Ingenuo estacional | Repite lo vendido el mismo día de la semana anterior |
| Media móvil de 28 días | Promedio de los últimos 28 días |
| Promedio estacional de 4 semanas | Promedio del mismo día de la semana en las últimas 4 semanas |
| **SES + índice semanal** | Divide cada venta por el índice de su día de la semana (estimado por categoría con todas las series juntas), estima el nivel con suavizado exponencial simple (con alpha elegido por serie, solo con datos anteriores al origen) y vuelve a aplicar el índice |
| **LightGBM** | Un modelo global con estrategia directa: cada fila es una serie, un origen y un día del horizonte. 34 variables: ventas recientes, promedios, intermitencia, antigüedad de la serie, mismo día de la semana, calendario, SNAP, precio y variables de la serie. Entrenado con 52 orígenes semanales; se eligió en validación entre seis configuraciones (pérdida de Tweedie con exponente 1,5 y 15 hojas, 509 árboles) |
| **Red neuronal (Keras)** | Perceptrón multicapa de 256, 128 y 64 unidades sobre las mismas filas y variables, con embeddings para las variables categóricas, salida positiva y pérdida de Tweedie. Se promedian 3 redes con semillas distintas |

Una prueba automática comprueba que las variables del modelo no cambian al alterar las ventas posteriores al origen, para descartar fugas de información.

## Resultados

| Modelo | RMSSE validación | RMSSE test | MAE validación | MAE test |
|---|---|---|---|---|
| **LightGBM** | **0,740** | **0,768** | **0,983** | 1,011 |
| Promedio 50/50 de LightGBM y la red | 0,743 | 0,769 | 0,994 | 1,013 |
| **SES + índice semanal** | 0,742 | 0,769 | 1,036 | **1,008** |
| Red neuronal | 0,749 | 0,772 | 1,015 | 1,023 |
| Media móvil de 28 días | 0,751 | 0,782 | 1,020 | 1,059 |
| Promedio estacional de 4 semanas | 0,811 | 0,849 | 1,045 | 1,101 |
| Cero | 0,940 | 0,990 | 1,392 | 1,477 |
| Ingenuo | 0,936 | 1,007 | 1,361 | 1,306 |
| Ingenuo estacional | 0,995 | 1,029 | 1,195 | 1,233 |

El promedio 50/50 es una regla fija definida de antemano, no ajustada a los resultados.

### ¿Las diferencias son reales?

Los cuatro mejores modelos están dentro de un 0,5 % en RMSSE, así que se calculó el intervalo de confianza del 95 % de la diferencia de error entre pares de modelos con un bootstrap pareado entre series (5.000 remuestras). Una diferencia negativa indica que el primer modelo se equivoca menos; con asterisco, los intervalos que excluyen el cero.

| Comparación (A − B) | RMSSE validación | RMSSE test | MAE validación | MAE test |
|---|---|---|---|---|
| LightGBM − SES | −0,0018 (−0,0079 a 0,0046) | −0,0009 (−0,0056 a 0,0039) | −0,0527 (−0,1164 a −0,0051) * | +0,0035 (−0,0120 a 0,0196) |
| Red neuronal − LightGBM | +0,0093 (0,0037 a 0,0158) * | +0,0038 (−0,0004 a 0,0088) | +0,0316 (0,0115 a 0,0554) * | +0,0116 (−0,0076 a 0,0386) |
| Red neuronal − SES | +0,0076 (−0,0012 a 0,0183) | +0,0029 (−0,0019 a 0,0080) | −0,0211 (−0,0823 a 0,0296) | +0,0150 (−0,0042 a 0,0391) |
| Promedio 50/50 − SES | +0,0010 (−0,0058 a 0,0090) | −0,0003 (−0,0044 a 0,0038) | −0,0415 (−0,1050 a 0,0090) | +0,0056 (−0,0077 a 0,0206) |
| SES − media móvil | −0,0093 (−0,0154 a −0,0036) * | −0,0131 (−0,0195 a −0,0078) * | +0,0157 (−0,0197 a 0,0619) | −0,0512 (−0,1056 a −0,0084) * |

- **Solo 6 de las 20 comparaciones excluyen el cero**, y con 20 intervalos cabe esperar aproximadamente uno por azar.
- **SES supera a la media móvil en RMSSE en las dos ventanas.** Es la única diferencia que se sostiene en ambas.
- **LightGBM no supera de forma confiable a SES:** su ventaja de MAE en validación (−0,053) desaparece en test. Además, su configuración se eligió entre seis en esa misma ventana, por lo que el resultado de validación es algo optimista.
- **La red neuronal es peor que LightGBM** en validación y tiene el mismo signo en test, aunque ahí los intervalos incluyen el cero. Tampoco supera a SES. Su RMSSE varía hasta 0,009 según la semilla, más que la diferencia entre LightGBM y SES.
- **Los intervalos del MAE son mucho más anchos que los del RMSSE:** el MAE está en unidades, y unas pocas series de mucho volumen lo dominan.
- **Pronosticar cero es el peor modelo en MAE,** aunque el 60 % de las series tiene mediana cero en los últimos 56 días. Las 45 series erráticas y suaves explican alrededor del 31 % de su error absoluto.

**Lectura general (hipótesis sin verificar):** a nivel de producto, tienda y día, la demanda es tan esporádica y ruidosa que casi todo lo predecible está en el nivel reciente de ventas. Las variables adicionales y los modelos más flexibles no producen una mejora confiable en estas ventanas.

Estos resultados no son comparables con los de la competencia: aquí se evalúan 600 series del nivel más desagregado, en una sola ventana, con RMSSE sin ponderar y con mucha menos ingeniería de variables y ajuste. En la competencia, los mejores resultados se obtuvieron con aprendizaje automático (el ganador combinó varios modelos LightGBM).

## Limitaciones

- Una sola ventana por etapa: el bootstrap mide la variabilidad entre series, pero no entre ventanas de tiempo. No hay corrección por comparaciones múltiples.
- Solo 600 de las 30.490 series; una validación sin eventos y un test con eventos menores.
- LightGBM y la red se ajustaron de forma limitada (de la red no se ajustaron la arquitectura, la tasa de aprendizaje ni la regularización; se probaron solo dos pérdidas y 3 semillas). La red es un perceptrón multicapa, por lo que los resultados no se extienden a arquitecturas secuenciales como LSTM o Transformers.
- RMSSE sin ponderar por ingresos, a diferencia del WRMSSE de la competencia.
- Las estadísticas de la exploración usan todos los días, incluidas las ventanas de evaluación; no se usan para ajustar ningún modelo, que solo ven datos anteriores a cada origen.
- Las tablas de eventos, SNAP y precios son descriptivas y no establecen relaciones causales.

## Trabajo futuro

- Evaluar con varios orígenes consecutivos para estimar la variabilidad entre ventanas.
- Usar las 30.490 series y evaluar el WRMSSE.
- Ajustar mejor los hiperparámetros, probar arquitecturas secuenciales y agregar variables agregadas por tienda o producto.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_baselines_estadistico.ipynb
│   ├── 03_lightgbm.ipynb
│   └── 04_red_neuronal_comparacion.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py            # carga de los datos M5 y muestra de series
│   ├── metrics.py         # RMSSE y MAE
│   ├── forecasting.py     # baselines, modelo estadístico y clasificación de la demanda
│   ├── gbm.py             # variables y entrenamiento de LightGBM
│   ├── nn.py              # preprocesamiento y red neuronal (Keras)
│   └── comparacion.py     # métricas por serie y bootstrap pareado
├── reports/               # tablas de resultados (.csv) y predicciones (.npz)
├── models/                # modelos entrenados (no se versionan)
└── data/                  # datos descargados (no se versionan)
```

## Cómo reproducirlo

1. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Obtener los datos de M5 desde la [página de Zenodo](https://zenodo.org/records/12636070) y dejar el zip (o los CSV) en `data/raw/`. `src/data.py` lo descomprime e intenta descargarlo si no está; si la descarga automática falla, basta con guardar el zip a mano en esa carpeta. Se necesitan `calendar.csv`, `sell_prices.csv` y `sales_train_evaluation.csv`.
3. Ejecutar los notebooks en orden desde `notebooks/`. El 03 usa los resultados del 02, y el 04 usa las predicciones que guarda el 03. En una CPU de escritorio, los modelos de los notebooks 02 y 03 tardan segundos por configuración y el entrenamiento de las redes del 04, unos pocos minutos.

Desarrollado con Python 3.13. Las semillas están fijadas, pero el entrenamiento de las redes puede variar levemente entre ejecuciones y equipos.

## Referencias y licencias

- Makridakis, S., Spiliotis, E. y Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
- Syntetos, A. A., Boylan, J. E. y Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495-503. https://doi.org/10.1057/palgrave.jors.2601841
- Hyndman, R. J. y Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Datos: M5 Forecasting - Accuracy (copia en Zenodo, registro 12636070, licencia CC BY 4.0). Los datos no se incluyen en este repositorio.
