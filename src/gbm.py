"""Variables y entrenamiento de LightGBM para pronosticar 28 días de ventas (estrategia directa).

Cada fila del conjunto de entrenamiento es un trío (serie, origen, horizonte): las variables
describen lo que se sabe en el origen (historial hasta ese día) más lo que se conoce de antemano
del día a pronosticar (calendario, SNAP y precio), y el objetivo es la venta de ese día.
"""
import warnings

import numpy as np
import pandas as pd

from .metrics import first_sale_index

HORIZONTE = 28
TIPOS_EVENTO = ["ninguno", "Cultural", "National", "Religious", "Sporting"]
CATEGORICAS = ["tienda", "departamento", "categoria", "tipo_evento"]

PARAMS_BASE = dict(
    learning_rate=0.05, min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, importance_type="gain", n_jobs=-1, verbose=-1,
)


def preparar_contexto(muestra, calendario, precios, dias):
    """Matrices (series × días) de ventas, precios, SNAP y calendario, y las variables estáticas."""
    n, T, ND = len(muestra), len(dias), len(calendario)
    Y = muestra[dias].to_numpy().astype(np.float32)

    # Precio de cada serie en cada día del calendario (NaN si todavía no tenía precio)
    pares = pd.MultiIndex.from_arrays([muestra["store_id"].astype(str), muestra["item_id"].astype(str)])
    items = muestra["item_id"].astype(str).unique()
    tiendas = muestra["store_id"].astype(str).unique()
    pr = precios[precios["item_id"].isin(items) & precios["store_id"].isin(tiendas)]  # filtrar antes de convertir
    pr = pr.assign(store_id=pr["store_id"].astype(str), item_id=pr["item_id"].astype(str))
    pr = pr.merge(muestra[["store_id", "item_id"]].astype(str).drop_duplicates(), on=["store_id", "item_id"])
    piv = pr.pivot(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price").reindex(pares)
    P = piv.reindex(columns=calendario["wm_yr_wk"].to_numpy()).to_numpy(dtype=np.float32)

    cal = calendario.reset_index(drop=True)
    fechas = pd.to_datetime(cal["date"])
    calendario_arr = {
        "dia_semana": (cal["wday"].to_numpy() - 1).astype(np.int8),
        "dia_mes": fechas.dt.day.to_numpy().astype(np.int8),
        "mes": fechas.dt.month.to_numpy().astype(np.int8),
        "es_navidad": (cal["event_name_1"] == "Christmas").to_numpy().astype(np.int8),
        "tipo_evento": pd.Categorical(cal["event_type_1"].fillna("ninguno"), categories=TIPOS_EVENTO).codes.astype(np.int8),
    }

    estados = muestra["state_id"].astype(str).to_numpy()
    snap = np.zeros((n, ND), dtype=np.int8)
    for e in np.unique(estados):
        snap[estados == e] = cal[f"snap_{e}"].to_numpy().astype(np.int8)

    estaticas = {}
    for nombre, col in (("tienda", "store_id"), ("departamento", "dept_id"), ("categoria", "cat_id")):
        vals = muestra[col].astype(str)
        cats = sorted(vals.unique())
        estaticas[nombre] = (pd.Categorical(vals, categories=cats).codes.astype(np.int8), cats)

    return {"Y": Y, "P": P, "snap": snap, "cal": calendario_arr, "estaticas": estaticas, "n": n, "T": T}


def variables_origen(Y, o):
    """Variables que dependen solo del historial hasta el día `o` (Y[:, :o]).

    Devuelve (diccionario de variables, vector de series con alguna venta hasta el origen).
    """
    H = Y[:, :o]
    f = {}
    for j in range(7):
        f[f"lag_{j}"] = H[:, o - 1 - j]
    for w in (7, 14, 28, 56, 91):
        f[f"media_{w}"] = H[:, -w:].mean(axis=1)
    f["desvio_28"] = H[:, -28:].std(axis=1)
    f["ceros_28"] = (H[:, -28:] == 0).mean(axis=1)
    f["ceros_91"] = (H[:, -91:] == 0).mean(axis=1)

    pos = H > 0
    hay = pos.any(axis=1)
    f["dias_desde_venta"] = np.where(hay, pos[:, ::-1].argmax(axis=1), o).astype(np.float32)
    ult = H[:, -91:]
    nz = (ult > 0).sum(axis=1)
    f["media_con_venta_91"] = np.where(nz > 0, ult.sum(axis=1) / np.maximum(nz, 1), np.nan).astype(np.float32)
    f["tendencia"] = f["media_28"] / (f["media_91"] + 0.01)
    inicio = np.where(hay, pos.argmax(axis=1), o)
    f["edad"] = (o - inicio).astype(np.float32)
    return f, hay


def _mismos_dias(H, o, h=HORIZONTE, periodo=7, semanas=4):
    """Para cada día del horizonte: venta del mismo día de la semana anterior y promedio de las últimas 4."""
    ks = np.arange(1, h + 1)
    m = np.ceil(ks / periodo).astype(int)
    lag = H[:, o + ks - periodo * m - 1]
    prom = np.mean([H[:, o + ks - periodo * (m + j) - 1] for j in range(semanas)], axis=0)
    return lag, prom


def construir_dataset(ctx, o, con_objetivo=True, h=HORIZONTE):
    """Filas (serie × día del horizonte) para el origen `o`.

    Solo se incluyen las series con alguna venta hasta el origen. Devuelve (X, y, idx), donde
    idx son las posiciones de esas series; las filas están ordenadas por serie y luego por horizonte.
    """
    Y = ctx["Y"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f, hay = variables_origen(Y, o)
        idx = np.where(hay)[0]
        m = len(idx)
        ks = np.arange(1, h + 1)
        dias_obj = o + ks - 1  # posición (desde 0) de los días a pronosticar en calendario y precios

        cols = {nombre: np.repeat(v[idx], h) for nombre, v in f.items()}
        lag, prom = _mismos_dias(Y[:, :o], o, h)
        cols["lag_mismo_dia"] = lag[idx].ravel()
        cols["prom_mismo_dia_4s"] = prom[idx].ravel()
        cols["horizonte"] = np.tile(ks, m).astype(np.int8)

        cal = ctx["cal"]
        for nombre in ("dia_semana", "dia_mes", "mes", "es_navidad"):
            cols[nombre] = np.tile(cal[nombre][dias_obj], m)
        cols["snap"] = ctx["snap"][idx][:, dias_obj].ravel()

        P = ctx["P"]
        p_obj = P[idx][:, dias_obj]
        p_ori = P[idx, o - 1][:, None]
        p_med = np.nanmedian(P[idx][:, :o], axis=1)[:, None]
        cols["precio_obj"] = p_obj.ravel()
        cols["precio_rel_origen"] = (p_obj / p_ori).ravel()
        cols["precio_rel_mediana"] = (p_obj / p_med).ravel()

    X = pd.DataFrame(cols)
    for nombre, (codigos, cats) in ctx["estaticas"].items():
        X[nombre] = pd.Categorical.from_codes(np.repeat(codigos[idx], h), categories=cats)
    X["tipo_evento"] = pd.Categorical.from_codes(np.tile(cal["tipo_evento"][dias_obj], m), categories=TIPOS_EVENTO)
    X = X[COLUMNAS]

    y = Y[idx][:, o:o + h].ravel() if con_objetivo else None
    return X, y, idx


def construir_varios(ctx, origenes):
    """Concatena los conjuntos de varios orígenes. Devuelve (X, y)."""
    Xs, ys = [], []
    for o in origenes:
        X, y, _ = construir_dataset(ctx, o)
        Xs.append(X)
        ys.append(y)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def a_matriz(pred, idx, n, h=HORIZONTE):
    """Devuelve el pronóstico como matriz (n_series, h); las series sin ventas hasta el origen reciben 0."""
    out = np.zeros((n, h))
    out[idx] = np.asarray(pred).reshape(len(idx), h)
    return np.maximum(out, 0.0)


def verificar_sin_fuga(ctx, o, semilla=0):
    """Comprueba que las variables no cambian si se altera lo vendido después del origen
    (más allá de diferencias de redondeo en punto flotante)."""
    rng = np.random.default_rng(semilla)
    Y2 = ctx["Y"].copy()
    Y2[:, o:] = rng.integers(0, 50, size=Y2[:, o:].shape)
    X1, _, i1 = construir_dataset(ctx, o, con_objetivo=False)
    X2, _, i2 = construir_dataset({**ctx, "Y": Y2}, o, con_objetivo=False)
    if not np.array_equal(i1, i2):
        return False
    for c in X1.columns:
        if str(X1[c].dtype) == "category":
            iguales = X1[c].equals(X2[c])
        else:  # tolerancia mínima: la suma en punto flotante puede variar en el último decimal
            iguales = np.allclose(X1[c].to_numpy(dtype=float), X2[c].to_numpy(dtype=float), rtol=1e-5, atol=1e-6, equal_nan=True)
        if not iguales:
            return False
    return True


COLUMNAS = (
    [f"lag_{j}" for j in range(7)]
    + [f"media_{w}" for w in (7, 14, 28, 56, 91)]
    + ["desvio_28", "ceros_28", "ceros_91", "dias_desde_venta", "media_con_venta_91", "tendencia", "edad",
       "lag_mismo_dia", "prom_mismo_dia_4s", "horizonte",
       "dia_semana", "dia_mes", "mes", "es_navidad", "tipo_evento", "snap",
       "precio_obj", "precio_rel_origen", "precio_rel_mediana",
       "tienda", "departamento", "categoria"]
)


def entrenar(X, y, X_es, y_es, config, semilla=42, n_estimators=1500, paciencia=50):
    """Entrena LightGBM con parada temprana sobre un conjunto de control (X_es, y_es)."""
    import lightgbm as lgb

    modelo = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=semilla, **{**PARAMS_BASE, **config})
    modelo.fit(X, y, eval_set=[(X_es, y_es)],
               callbacks=[lgb.early_stopping(paciencia, verbose=False), lgb.log_evaluation(0)])
    return modelo


def reentrenar(X, y, config, n_estimators, semilla=42):
    """Entrena LightGBM con un número fijo de árboles, sin conjunto de control."""
    import lightgbm as lgb

    modelo = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=semilla, **{**PARAMS_BASE, **config})
    modelo.fit(X, y)
    return modelo
