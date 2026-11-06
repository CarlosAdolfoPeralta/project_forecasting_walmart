# Pronóstico de demanda en Walmart con M5: baselines, LightGBM y red neuronal

Proyecto de Data Science sobre el dataset de la competencia **M5 Forecasting – Accuracy** (ventas diarias de Walmart). Se pronostica la demanda a 28 días de una muestra de 600 series y se comparan, con las mismas reglas de evaluación, modelos de complejidad creciente: baselines simples, un modelo estadístico (suavizado exponencial con índice semanal), LightGBM y una red neuronal en Keras.

**Autor:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

La pregunta del proyecto no es "¿gana el deep learning?", sino **¿qué tanto mejora cada modelo más complejo y esa mejora es mayor que el ruido?** La respuesta, con intervalos de confianza, está en el [notebook 04](notebooks/04_red_neuronal_comparacion.ipynb).

## Resultado principal

Con la muestra y las ventanas de este proyecto, **LightGBM y el modelo estadístico simple (SES + índice semanal) empatan**, y la red neuronal queda levemente por detrás. Una combinación 50/50 de LightGBM y la red tampoco mejora.

Error de cada modelo en las dos ventanas de 28 días (menor es mejor). El test se evaluó una sola vez, con el modelo ya elegido en validación:

| Modelo | RMSSE validación | RMSSE test | MAE validación | MAE test |
|---|---:|---:|---:|---:|
| LightGBM | 0,740 | 0,768 | 0,983 | 1,011 |
| Promedio LightGBM + red neuronal | 0,743 | 0,769 | 0,994 | 1,013 |
| SES + índice semanal | 0,742 | 0,769 | 1,036 | 1,008 |
| Red neuronal | 0,749 | 0,772 | 1,015 | 1,023 |
| Media móvil 28 días | 0,751 | 0,782 | 1,020 | 1,059 |
| Promedio estacional (4 semanas) | 0,811 | 0,849 | 1,045 | 1,101 |
| Cero | 0,940 | 0,990 | 1,392 | 1,477 |
| Ingenuo | 0,936 | 1,007 | 1,361 | 1,306 |
| Ingenuo estacional | 0,995 | 1,029 | 1,195 | 1,233 |

**Qué diferencias son reales** (bootstrap pareado entre las 600 series, IC del 95 %):

- De 20 comparaciones, solo 6 tienen un intervalo que excluye el cero. Con 20 intervalos cabe esperar aproximadamente uno por azar.
- SES supera a la media móvil en RMSSE en las dos ventanas (1,2 % y 1,7 %): es la única diferencia que se sostiene en ambas.
- La red neuronal rinde peor que LightGBM en validación; en test la diferencia es menor y los intervalos incluyen el cero.
- LightGBM no supera de forma confiable a SES: su ventaja de MAE en validación desaparece en test.
- Diferencias de menos de alrededor de 1 % no se distinguen del ruido con 600 series y una ventana de 28 días.

Con series tan esporádicas, casi todo lo predecible parece estar en el nivel reciente de ventas: las tres variables más usadas por LightGBM (promedios de las últimas semanas) suman el 78 % de la ganancia, y el día de la semana pesa menos del 1 %.

## Datos

El dataset M5 tiene 30.490 series (3.049 productos en 10 tiendas de California, Texas y Wisconsin), con ventas diarias del 29 de enero de 2011 al 22 de mayo de 2016 (1.941 días), un calendario con eventos y días SNAP, y precios semanales. Se usa la versión `evaluation`.

Algunos hallazgos de la exploración ([notebook 01](notebooks/01_exploracion.ipynb)):

- **Demanda muy esporádica:** el 68 % de las observaciones diarias son cero, y el 91 % de las series se clasifica como intermitente o irregular.
- **Series que empiezan tarde:** el 77 % de las series tiene su primera venta después del primer día, y el 44 % tiene al menos un año de ceros iniciales.
- **Estacionalidad semanal fuerte en el total:** sábado y domingo venden alrededor de un 20 % más que el promedio, y de martes a jueves alrededor de un 12 % menos.
- **Cierre de Navidad:** el 25 de diciembre las tiendas prácticamente no venden.
- **Días SNAP:** se vende más en esos días (1,08 veces en California, 1,11 en Texas y 1,21 en Wisconsin, como diferencia descriptiva y no causal).

**Los datos no se incluyen en el repositorio** (pesan unos 430 MB). Se descargan de Kaggle (competencia *M5 Forecasting – Accuracy*, sujeta a sus términos). También existe una copia en Zenodo (registro 12636070, licencia CC BY 4.0).

## Metodología

- **Muestra:** 600 series elegidas al azar (60 por tienda, semilla 42).
- **Horizonte:** 28 días, sin usar información posterior al origen del pronóstico.
- **Ventanas:** validación del 28 de marzo al 24 de abril de 2016 (días 1.886 a 1.913) y test del 25 de abril al 22 de mayo de 2016 (días 1.914 a 1.941). Las variantes de modelos se comparan y eligen en validación; el test se usa una sola vez, con el modelo reentrenado con los datos hasta su origen.
- **Métricas:** RMSSE (error escalado, que cuenta cada serie desde su primera venta) y MAE. Se leen juntas porque con demanda esporádica pueden ordenar los modelos de forma distinta.
- **Modelos:**
  1. Baselines: cero, ingenuo, ingenuo estacional, media móvil de 28 días y promedio estacional de 4 semanas.
  2. SES (suavizado exponencial simple) con índice semanal por categoría.
  3. LightGBM global con estrategia directa (una fila por serie, origen y día del horizonte), con promedios móviles, rezagos, antigüedad de la serie, calendario, eventos, SNAP y precio. Se probaron exponentes de Tweedie y tamaños de árbol; el mejor fue Tweedie 1,5 con 15 hojas.
  4. Red neuronal (perceptrón multicapa de 256, 128 y 64 unidades, con embeddings para las variables categóricas), con las mismas filas y variables que LightGBM. Se promedian 3 redes con semillas distintas.
- **Comparación:** bootstrap pareado entre series (5.000 remuestras) para decidir si las diferencias superan el ruido.

## Estructura del repositorio

```
.
├── data/                  # se genera al correr el proyecto; ignorada por Git
│   ├── raw/               # archivos originales de M5 (se colocan a mano)
│   └── processed/         # datos procesados
├── models/                # modelos entrenados de LightGBM; ignorada por Git
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_baselines_estadistico.ipynb
│   ├── 03_lightgbm.ipynb
│   └── 04_red_neuronal_comparacion.ipynb
├── reports/               # resultados generados por los notebooks (CSV y predicciones)
│   └── figures/
├── src/
│   ├── data.py            # carga y preparación de los datos M5
│   ├── metrics.py         # RMSSE y MAE
│   ├── forecasting.py     # baselines y modelo estadístico (SES + índice semanal)
│   ├── gbm.py             # variables y entrenamiento de LightGBM
│   ├── nn.py              # red neuronal en Keras
│   └── comparacion.py     # comparación por serie y bootstrap pareado
├── requirements.txt
└── .gitignore
```

Los notebooks se apoyan unos en otros a través de `reports/`: el 04 lee las predicciones de LightGBM que guarda el 03 y acumula sus resultados en `reports/resultados_modelos.csv`.

## Cómo reproducirlo

1. Crear un entorno e instalar las dependencias (las versiones de `requirements.txt` son mínimas):

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    |    macOS / Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Descargar los datos de M5 desde Kaggle y colocar los archivos en `data/raw/`.
3. Abrir Jupyter y ejecutar los notebooks **en orden**: `01`, `02`, `03` y `04`. Cada notebook debe ejecutarse con la carpeta `notebooks/` como directorio de trabajo, porque calcula la raíz del proyecto como `..`.

El proyecto se desarrolló con Python 3.13 en Windows, solo con CPU. La ejecución completa tarda alrededor de 10 minutos, de los cuales el notebook 04 (la red neuronal) lleva unos 3. Las semillas están fijadas, pero el entrenamiento de las redes puede variar levemente entre entornos y versiones.

## Limitaciones

- Una sola ventana de validación y una de test: el bootstrap mide la variabilidad entre series, pero no entre ventanas de tiempo.
- 600 de las 30.490 series, y RMSSE sin ponderar por ingresos (a diferencia del WRMSSE de la competencia).
- La red neuronal es un perceptrón multicapa sobre variables construidas a mano; los resultados no se extienden a arquitecturas secuenciales como LSTM o Transformers.
- Hiperparámetros explorados de forma limitada, y sin corrección por comparaciones múltiples.
- Las ventanas de evaluación no incluyen los feriados de mayor impacto (Navidad, Acción de Gracias, Año Nuevo).

## Fuente de los datos y licencia

- **Datos:** competencia M5, organizada por la Universidad de Nicosia. Referencia: Makridakis, S., Spiliotis, E. y Assimakopoulos, V. (2022). *M5 accuracy competition: Results, findings, and conclusions*. International Journal of Forecasting, 38(4).
- **Licencia:** este repositorio no incluye una licencia para el código, y los datos no forman parte de él.
