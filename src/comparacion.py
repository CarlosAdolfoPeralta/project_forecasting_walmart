"""Comparación de modelos por serie y bootstrap pareado entre series."""
import numpy as np

from .metrics import rmsse


def por_serie(y_train, y_true, pred):
    """RMSSE y MAE de cada serie (el RMSSE es NaN en las series sin escala)."""
    r = rmsse(y_train, y_true, pred)
    a = np.abs(np.asarray(y_true, float) - np.asarray(pred, float)).mean(axis=1)
    return r, a


def bootstrap_pareado(a, b, n_boot=5000, rng=None):
    """Diferencia media de error (a − b) entre dos modelos, con intervalo de confianza por bootstrap.

    `a` y `b` son el error de cada serie con cada modelo. Se remuestrean las series con reemplazo,
    conservando el emparejamiento. Una diferencia negativa significa que `a` se equivoca menos.
    `prob_a_mejor` es la proporción de remuestras en que `a` tiene menor error medio.
    """
    rng = rng if rng is not None else np.random.default_rng()
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    medias = d[idx].mean(axis=1)
    return {
        "dif": float(d.mean()),
        "ic_bajo": float(np.percentile(medias, 2.5)),
        "ic_alto": float(np.percentile(medias, 97.5)),
        "prob_a_mejor": float((medias < 0).mean()),
        "series": int(ok.sum()),
    }
