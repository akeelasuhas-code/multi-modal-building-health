"""bhas_gee.py -- Google Earth Engine access. The ONLY module that imports `ee`.

Annual dry-season composites of NDVI and NDBI from Landsat 5/7/8/9 Collection 2 Level-2 surface
reflectance, averaged over a small buffer around the building. The construction-year detector itself
(pure numpy) lives in bhas_core.estimate_construction_year.

NOT TESTED AGAINST LIVE EARTH ENGINE in the development sandbox (no credentials there).
Run `python -c "import bhas_gee; print(bhas_gee.init_ee(project='suhas-proj1'))"` in Colab first.
"""
from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd

# SR band names per sensor: (RED, NIR, SWIR1)
_SENSORS = {
    "LANDSAT/LT05/C02/T1_L2": ("SR_B3", "SR_B4", "SR_B5"),
    "LANDSAT/LE07/C02/T1_L2": ("SR_B3", "SR_B4", "SR_B5"),
    "LANDSAT/LC08/C02/T1_L2": ("SR_B4", "SR_B5", "SR_B6"),
    "LANDSAT/LC09/C02/T1_L2": ("SR_B4", "SR_B5", "SR_B6"),
}


def init_ee(project: str | None = None, secrets=None):
    """Try, in order: (1) service account in Streamlit secrets [gee_service_account]; (2) existing user
    credentials (Colab / local after ee.Authenticate()). Returns (ok, message)."""
    try:
        import ee
    except ImportError:
        return False, "earthengine-api is not installed."
    sa = None
    try:
        if secrets is not None and "gee_service_account" in secrets:
            sa = dict(secrets["gee_service_account"])
    except Exception:  # noqa: BLE001  (no secrets file configured)
        sa = None
    try:
        if sa:
            creds = ee.ServiceAccountCredentials(sa["client_email"], key_data=json.dumps(sa))
            ee.Initialize(creds, project=sa.get("project_id") or project)
            return True, "Connected with service account."
        ee.Initialize(project=project)
        return True, "Connected with user credentials."
    except Exception as e:  # noqa: BLE001
        return False, f"Earth Engine not connected: {e}"


def fetch_annual_indices(lat: float, lon: float, radius_m: float = 45.0, y0: int = 1985, y1: int | None = None,
                         season: tuple = (11, 3)) -> pd.DataFrame:
    """One row per year: year, n_images, ndvi, ndbi. Year y uses the season that ENDS in y
    (default Nov(y-1)..Mar(y): Hyderabad's dry season, avoids monsoon cloud and crop phenology swings)."""
    import ee

    y1 = y1 or (datetime.date.today().year - 1)
    sm, em = season
    wrap = sm > em
    pt = ee.Geometry.Point([lon, lat])
    region = pt.buffer(radius_m)

    def prep(red, nir, swir):
        def f(img):
            qa = img.select("QA_PIXEL")
            clear = (qa.bitwiseAnd(1 << 1).eq(0)      # dilated cloud
                     .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
                     .And(qa.bitwiseAnd(1 << 4).eq(0)))  # cloud shadow
            sr = img.select([red, nir, swir]).multiply(0.0000275).add(-0.2).rename(["RED", "NIR", "SWIR1"])
            ndvi = sr.normalizedDifference(["NIR", "RED"]).rename("NDVI")
            ndbi = sr.normalizedDifference(["SWIR1", "NIR"]).rename("NDBI")
            return ndvi.addBands(ndbi).updateMask(clear).copyProperties(img, ["system:time_start"])
        return f

    merged = None
    for name, (r, n, s) in _SENSORS.items():
        c = ee.ImageCollection(name).filterBounds(pt).map(prep(r, n, s))
        merged = c if merged is None else merged.merge(c)

    empty = ee.Image.constant([0, 0]).rename(["NDVI", "NDBI"]).updateMask(ee.Image.constant(0))

    def per_year(y):
        y = ee.Number(y)
        start = ee.Date.fromYMD(y.subtract(1) if wrap else y, sm, 1)
        end = ee.Date.fromYMD(y, em, 1).advance(1, "month")
        sub = merged.filterDate(start, end)
        n = sub.size()
        comp = ee.Image(ee.Algorithms.If(n.gt(0), sub.median(), empty))
        vals = comp.reduceRegion(ee.Reducer.mean(), region, 30, maxPixels=1_000_000)
        return ee.Feature(None, {"year": y, "n": n, "ndvi": vals.get("NDVI"), "ndbi": vals.get("NDBI")})

    feats = ee.FeatureCollection(ee.List.sequence(y0, y1).map(per_year)).getInfo()["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({"year": int(p["year"]), "n_images": int(p.get("n") or 0),
                     "ndvi": np.nan if p.get("ndvi") is None else float(p["ndvi"]),
                     "ndbi": np.nan if p.get("ndbi") is None else float(p["ndbi"])})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
