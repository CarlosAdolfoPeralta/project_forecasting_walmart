"""Red neuronal (Keras) para el pronóstico directo a 28 días: preprocesamiento, modelo y entrenamiento.

Usa las mismas filas y variables que LightGBM (src/gbm.py), para que la comparación sea justa.
"""
import numpy as np
import pandas as pd

# Variables que entran como embeddings (una representación numérica aprendida por categoría)
CATEGORICAS_NN = ["tienda", "departamento", "categoria", "tipo_evento", "dia_semana"]
# Variables sesgadas a la derecha: se les aplica log(1 + x) antes de estandarizar
LOG1P = (
    [f"lag_{j}" for j in range(7)] + [f"media_{w}" for w in (7, 14, 28, 56, 91)]
    + ["desvio_28", "media_con_venta_91", "dias_desde_venta", "edad",
       "lag_mismo_dia", "prom_mismo_dia_4s", "tendencia", "precio_obj"]
)
# Variables con valores faltantes: se completan y se agrega un indicador de faltante
CON_FALTANTES = ["media_con_venta_91", "precio_obj", "precio_rel_origen", "precio_rel_mediana"]


class Preprocesador:
    """Convierte el conjunto de variables en las entradas de la red.

    Las numéricas se transforman (log1p en las sesgadas), se estandarizan con la media y el desvío
    del conjunto de entrenamiento y los faltantes se completan con la media (0 tras estandarizar).
    Las categóricas pasan a códigos enteros para los embeddings.
    """

    def _numericas(self, X):
        A = X[self.numericas].astype("float32").copy()
        for c in CON_FALTANTES:
            A[f"falta_{c}"] = A[c].isna().astype("float32")
        for c in LOG1P:
            A[c] = np.log1p(A[c].clip(lower=0))
        return A

    def fit(self, X):
        self.numericas = [c for c in X.columns if c not in CATEGORICAS_NN]
        A = self._numericas(X)
        self.columnas = list(A.columns)
        self.media = A.mean()
        self.desvio = A.std().replace(0, 1).fillna(1)
        self.n_cat = {}
        for c in CATEGORICAS_NN:
            self.n_cat[c] = len(X[c].cat.categories) if str(X[c].dtype) == "category" else 7  # 7 días de la semana
        return self

    def transform(self, X):
        A = (self._numericas(X)[self.columnas] - self.media) / self.desvio
        entradas = {"num": A.fillna(0).to_numpy(dtype="float32")}
        for c in CATEGORICAS_NN:
            codigos = X[c].cat.codes.to_numpy() if str(X[c].dtype) == "category" else X[c].to_numpy()
            entradas[c] = codigos.astype("int32").reshape(-1, 1)
        return entradas


def perdida_tweedie(p):
    """Pérdida de Tweedie (log-verosimilitud negativa sin constantes) para un exponente p en [1, 2).

    Con p = 1 es la pérdida de Poisson. Exige predicciones positivas.
    """
    import tensorflow as tf

    def perdida(y_true, y_pred):
        mu = tf.maximum(y_pred, 1e-3)
        y = tf.reshape(tf.cast(y_true, mu.dtype), tf.shape(mu))
        if p == 1.0:
            valor = mu - y * tf.math.log(mu)
        else:
            valor = -y * tf.pow(mu, 1.0 - p) / (1.0 - p) + tf.pow(mu, 2.0 - p) / (2.0 - p)
        return tf.reduce_mean(valor, axis=-1)

    return perdida


def construir_red(prep, capas=(256, 128, 64), dropout=0.1):
    """Perceptrón multicapa con embeddings para las variables categóricas y salida positiva."""
    from tensorflow import keras

    L = keras.layers
    entradas = {"num": L.Input(shape=(len(prep.columnas),), name="num")}
    partes = [entradas["num"]]
    for nombre, n in prep.n_cat.items():
        e = L.Input(shape=(1,), name=nombre, dtype="int32")
        entradas[nombre] = e
        dim = min(8, n // 2 + 1)
        partes.append(L.Flatten()(L.Embedding(n, dim)(e)))
    x = L.Concatenate()(partes)
    for unidades in capas:
        x = L.Dense(unidades, activation="relu")(x)
        if dropout:
            x = L.Dropout(dropout)(x)
    salida = L.Dense(1, activation="softplus")(x)
    return keras.Model(entradas, salida)


def entrenar_red(prep, ent_tr, y_tr, ent_ctrl, y_ctrl, p, semilla=42, epocas=30, batch=1024, paciencia=3, lr=1e-3):
    """Entrena con parada temprana sobre un conjunto de control. Devuelve (modelo, mejor época, historial)."""
    from tensorflow import keras

    keras.utils.set_random_seed(semilla)
    modelo = construir_red(prep)
    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss=perdida_tweedie(p))
    hist = modelo.fit(
        ent_tr, np.asarray(y_tr, dtype="float32").reshape(-1, 1),
        validation_data=(ent_ctrl, np.asarray(y_ctrl, dtype="float32").reshape(-1, 1)),
        epochs=epocas, batch_size=batch, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)],
    )
    mejor = int(np.argmin(hist.history["val_loss"])) + 1
    return modelo, mejor, hist.history


def reentrenar_red(prep, ent, y, p, epocas, semilla=42, batch=1024, lr=1e-3):
    """Entrena con un número fijo de épocas, sin conjunto de control."""
    from tensorflow import keras

    keras.utils.set_random_seed(semilla)
    modelo = construir_red(prep)
    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss=perdida_tweedie(p))
    modelo.fit(ent, np.asarray(y, dtype="float32").reshape(-1, 1), epochs=epocas, batch_size=batch, verbose=0)
    return modelo
