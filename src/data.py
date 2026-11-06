"""Carga y preparación de los datos M5 (ventas diarias de Walmart)."""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

ZENODO_PAGINA = "https://zenodo.org/records/12636070"
ZENODO_ZIP = ZENODO_PAGINA + "/files/m5-forecasting-accuracy.zip?download=1"

COLS_ID = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
COLS_CALENDARIO = [
    "d", "date", "wm_yr_wk", "wday", "month", "year",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]
SEED = 42


def _buscar_dir(raw_dir):
    encontrados = sorted(Path(raw_dir).rglob("calendar.csv"))
    return encontrados[0].parent if encontrados else None


def ensure_m5(raw_dir=RAW_DIR, descargar=True):
    """Devuelve la carpeta con los CSV de M5, descomprimiendo o descargando si hace falta."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    carpeta = _buscar_dir(raw_dir)
    if carpeta is not None:
        return carpeta

    zips = sorted(raw_dir.glob("*.zip"))
    if not zips and descargar:
        destino = raw_dir / "m5-forecasting-accuracy.zip"
        try:
            urllib.request.urlretrieve(ZENODO_ZIP, destino)
        except Exception as e:
            raise RuntimeError(
                f"No se pudo descargar M5 desde {ZENODO_ZIP}. "
                f"Se puede descargar el zip desde {ZENODO_PAGINA} y guardarlo en {raw_dir}."
            ) from e
        zips = [destino]
    if not zips:
        raise RuntimeError(f"No hay datos de M5 en {raw_dir}. Descargarlos desde {ZENODO_PAGINA}.")

    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(raw_dir / "m5")
    carpeta = _buscar_dir(raw_dir)
    if carpeta is None:
        raise RuntimeError("El zip descomprimido no contiene calendar.csv")
    return carpeta


def load_calendar(carpeta=None):
    carpeta = carpeta or ensure_m5()
    return pd.read_csv(Path(carpeta) / "calendar.csv", parse_dates=["date"])


def load_prices(carpeta=None):
    carpeta = carpeta or ensure_m5()
    return pd.read_csv(
        Path(carpeta) / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )


def load_sales(carpeta=None, version="evaluation"):
    """Ventas diarias en formato ancho (una columna por día: d_1, d_2, ...).

    La versión `evaluation` llega hasta d_1941 y la `validation` hasta d_1913.
    Las columnas de días se leen como int16 para ahorrar memoria.
    """
    carpeta = Path(carpeta or ensure_m5())
    archivo = carpeta / f"sales_train_{version}.csv"
    if not archivo.exists():
        otra = "validation" if version == "evaluation" else "evaluation"
        archivo = carpeta / f"sales_train_{otra}.csv"
        if not archivo.exists():
            raise FileNotFoundError(f"No se encontró sales_train_{version}.csv en {carpeta}")
    columnas = pd.read_csv(archivo, nrows=0).columns
    dias = [c for c in columnas if c.startswith("d_")]
    tipos = {c: "category" for c in COLS_ID}
    tipos.update({c: np.int16 for c in dias})
    return pd.read_csv(archivo, dtype=tipos)


def day_columns(sales):
    """Nombres de las columnas de días, ordenadas (d_1, d_2, ...)."""
    return sorted((c for c in sales.columns if c.startswith("d_")), key=lambda c: int(c[2:]))


def sample_series(sales, n_per_store=60, seed=SEED):
    """Muestra reproducible: n series al azar de cada tienda."""
    rng = np.random.default_rng(seed)
    elegidos = []
    for _, g in sales.groupby("store_id", observed=True):
        n = min(n_per_store, len(g))
        elegidos.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))
    return sales.loc[sorted(elegidos)].reset_index(drop=True)


def to_long(sales, calendar, prices):
    """Pasa las ventas a formato largo y agrega calendario y precio (un registro por serie y día)."""
    dias = day_columns(sales)
    largo = sales.melt(id_vars=COLS_ID, value_vars=dias, var_name="d", value_name="ventas")
    largo["dia"] = largo["d"].str[2:].astype("int32")
    largo = largo.merge(calendar[COLS_CALENDARIO], on="d", how="left")

    pares = sales[["item_id", "store_id"]].drop_duplicates()
    precios = prices.merge(pares, on=["item_id", "store_id"], how="inner")
    for c in ("item_id", "store_id"):
        precios[c] = precios[c].astype(str)
        largo[c] = largo[c].astype(str)
    largo = largo.merge(precios, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return largo
