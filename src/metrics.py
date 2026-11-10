"""Métricas de pronóstico: error escalado (RMSSE) y MAE."""
import numpy as np


def first_sale_index(Y):
    """Posición de la primera venta de cada serie (o la longitud total si nunca vendió)."""
    Y = np.asarray(Y)
    hay = (Y > 0).any(axis=1)
    return np.where(hay, (Y > 0).argmax(axis=1), Y.shape[1])


def rmsse(y_train, y_true, y_pred):
    """RMSSE de cada serie.

    Error cuadrático medio del pronóstico, dividido por el error cuadrático medio de un
    pronóstico ingenuo (repetir el día anterior) en el pasado de la misma serie, y con raíz
    cuadrada. El pasado se cuenta desde la primera venta, porque los ceros previos
    corresponden a productos que todavía no se vendían.

    Devuelve NaN en las series sin ventas en el entrenamiento o sin variación.
    y_train: (n_series, T); y_true, y_pred: (n_series, horizonte).
    """
    y_train, y_true, y_pred = (np.asarray(a, dtype=float) for a in (y_train, y_true, y_pred))
    n, T = y_train.shape
    inicio = first_sale_index(y_train)
    escala = np.full(n, np.nan)
    for i in range(n):
        if T - inicio[i] >= 2:
            m = np.mean(np.diff(y_train[i, inicio[i]:]) ** 2)
            if m > 0:
                escala[i] = m
    ecm = np.mean((y_true - y_pred) ** 2, axis=1)
    return np.sqrt(ecm / escala)


def mae(y_true, y_pred):
    """Error absoluto medio, en unidades vendidas (promedio sobre todas las series y días)."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def evaluar(y_train, y_true, y_pred):
    """RMSSE promedio (sobre las series válidas) y MAE."""
    r = rmsse(y_train, y_true, y_pred)
    return {
        "RMSSE": float(np.nanmean(r)),
        "MAE": mae(y_true, y_pred),
        "valid_series": int(np.isfinite(r).sum()),
        "total_series": int(len(r)),
    }
