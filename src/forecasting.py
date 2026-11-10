"""Baselines y modelo estadístico simple para pronosticar demanda diaria (M5).

Todas las funciones reciben el historial hasta el origen, `Y` con forma (n_series, T),
y devuelven el pronóstico de los próximos `h` días con forma (n_series, h).
"""
import os

import numpy as np
import pandas as pd

from .metrics import first_sale_index


def ventanas(T, horizonte=28):
    """Las dos últimas ventanas de `horizonte` días: (primer día, último día), con días desde 1."""
    return {
        "validation": (T - 2 * horizonte + 1, T - horizonte),
        "test": (T - horizonte + 1, T),
    }


def clasificar_demanda(Y):
    """Clasifica cada serie según Syntetos y Boylan (2005), desde su primera venta.

    ADI: intervalo medio entre demandas. CV²: variabilidad de las cantidades vendidas.
    Umbrales: ADI = 1,32 y CV² = 0,49.
    """
    Y = np.asarray(Y, dtype=float)
    n, T = Y.shape
    inicio = first_sale_index(Y)
    adi = np.full(n, np.nan)
    cv2 = np.full(n, np.nan)
    for i in range(n):
        y = Y[i, inicio[i]:]
        nz = y[y > 0]
        if len(nz) > 0:
            adi[i] = len(y) / len(nz)
            cv2[i] = (nz.std() / nz.mean()) ** 2
    tipo = np.select(
        [np.isnan(adi), (adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
        ["no sales", "smooth", "erratic", "intermittent"],
        default="irregular",
    )
    return pd.DataFrame({"start": inicio, "adi": adi, "cv2": cv2, "type": tipo})


# ----------------------------------------------------------------------------- baselines

def pronostico_cero(Y, h):
    """Siempre cero. Sirve de referencia para ver cómo se comportan las métricas."""
    return np.zeros((Y.shape[0], h))


def pronostico_ingenuo(Y, h):
    """Repite la última venta observada."""
    return np.repeat(np.asarray(Y, float)[:, -1:], h, axis=1)


def pronostico_ingenuo_estacional(Y, h, periodo=7):
    """Repite lo vendido el mismo día de la semana anterior."""
    Y = np.asarray(Y, float)
    T = Y.shape[1]
    cols = [T + k - periodo * int(np.ceil(k / periodo)) - 1 for k in range(1, h + 1)]
    return Y[:, cols]


def pronostico_media_movil(Y, h, ventana=28):
    """Promedio de los últimos `ventana` días, constante en todo el horizonte."""
    return np.repeat(np.asarray(Y, float)[:, -ventana:].mean(axis=1, keepdims=True), h, axis=1)


def pronostico_promedio_estacional(Y, h, semanas=4, periodo=7):
    """Para cada día del horizonte, promedio de ese mismo día de la semana en las últimas `semanas`."""
    Y = np.asarray(Y, float)
    T = Y.shape[1]
    out = np.empty((Y.shape[0], h))
    for k in range(1, h + 1):
        m = int(np.ceil(k / periodo))
        cols = [T + k - periodo * (m + j) - 1 for j in range(semanas)]
        out[:, k - 1] = Y[:, cols].mean(axis=1)
    return out


# ------------------------------------------------------------- modelo estadístico simple

def indices_semanales(Y, wday_train, grupos, dias=364):
    """Índice de cada día de la semana por grupo (por ejemplo, categoría).

    Es la venta media de ese día de la semana dividida por la venta media de todos los días,
    en los últimos `dias` días y contando cada serie solo desde su primera venta. Se estima
    con todas las series del grupo juntas, para que no dependa del ruido de una serie con
    pocas ventas. Cada índice promedia 1.
    """
    Y = np.asarray(Y, float)
    n, T = Y.shape
    inicio = first_sale_index(Y)
    factores = np.ones((n, 7))
    por_grupo = {}
    for g in np.unique(grupos):
        filas = grupos == g
        Yw = Y[filas][:, -dias:]
        wd = np.asarray(wday_train)[-dias:]
        dias_abs = np.arange(T - Yw.shape[1], T)
        valido = dias_abs[None, :] >= inicio[filas][:, None]
        medias = np.full(7, np.nan)
        for w in range(7):
            c = wd == w
            den = valido[:, c].sum()
            if den > 0:
                medias[w] = (Yw[:, c] * valido[:, c]).sum() / den
        medias = np.where(np.isnan(medias), np.nanmean(medias), medias)
        f = np.maximum(medias / medias.mean(), 1e-3)
        f = f / f.mean()
        factores[filas] = f
        por_grupo[g] = f
    return factores, por_grupo


def ses_semanal(Y, h, wday_train, wday_futuro, grupos, alphas=(0.02, 0.05, 0.1, 0.2, 0.4),
                dias_seleccion=364, dias_indice=364):
    """Suavizado exponencial simple sobre la serie sin estacionalidad semanal.

    1. Se divide cada venta por el índice semanal de su día (índices compartidos por grupo).
    2. Se estima el nivel de la serie con suavizado exponencial simple: cada día el nivel
       se corrige una fracción `alpha` del error. Alpha se elige por serie, entre `alphas`,
       como el que minimiza el error cuadrático de predecir un día con el nivel del día
       anterior en los últimos `dias_seleccion` días.
    3. El pronóstico de cada día es el nivel final multiplicado por el índice de ese día.

    Devuelve (pronóstico, info) con info = {"alphas": alpha elegido por serie, "indices": índice por grupo}.
    """
    Y = np.asarray(Y, float)
    n, T = Y.shape
    wday_train = np.asarray(wday_train)
    wday_futuro = np.asarray(wday_futuro)
    inicio = first_sale_index(Y)

    factores, por_grupo = indices_semanales(Y, wday_train, grupos, dias_indice)
    F = factores[np.arange(n)[:, None], wday_train[None, :]]
    Z = Y / F

    alphas_arr = np.asarray(alphas, dtype=float)[:, None]
    nivel = np.zeros((len(alphas), n))
    sse = np.zeros((len(alphas), n))
    for t in range(T):
        activa = t > inicio
        arrancan = np.where(inicio == t)[0]
        for i in arrancan:
            nivel[:, i] = Z[i, t:min(t + 28, T)].mean()
        if activa.any():
            err = Z[:, t][None, :] - nivel
            if t >= T - dias_seleccion:
                sse += np.where(activa[None, :], err ** 2, 0.0)
            nivel = np.where(activa[None, :], nivel + alphas_arr * err, nivel)

    mejor = sse.argmin(axis=0)
    nivel_final = nivel[mejor, np.arange(n)]
    Ff = factores[np.arange(n)[:, None], wday_futuro[None, :]]
    pron = np.maximum(nivel_final[:, None] * Ff, 0.0)
    pron[inicio >= T] = 0.0
    return pron, {"alphas": np.asarray(alphas)[mejor], "indices": por_grupo}


# ---------------------------------------------------------------------------- resultados

def guardar_resultados(df, ruta):
    """Agrega los resultados a un CSV acumulado (reemplaza las filas del mismo modelo y ventana)."""
    if os.path.exists(ruta):
        previo = pd.read_csv(ruta)
        df = pd.concat([previo, df], ignore_index=True)
    df = df.drop_duplicates(subset=["model", "window"], keep="last")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False)
    return df
