"""Zone map from static features, so places without a well (and held-out wells) get a zone (PLAN.md §2.3).

1. A random-forest classifier learns each training well's behaviour zone from features known everywhere:
   location, elevation, land cover and canal proximity. Features are read from the 1 km grid cell of the well,
   exactly as they are for every other cell, so training and mapping use the same inputs.
2. The classifier labels every cell of the study area (district outline + buffer).
3. A majority filter (each cell takes the most common zone within a square window) removes speckle, then patches
   smaller than `min_patch_km2` are absorbed by the zone that surrounds them most.
4. Wells, including held-out ones, take the zone of their map cell. This is the only rule used for prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

from src.utils.geo import Grid

STATIC = ["x_km", "y_km", "elev_m", "lc_cropland_frac", "lc_builtup_frac", "lc_tree_frac", "canal_dist_km",
          "canal_density_km_per_km2"]


def grid_of(ds: xr.Dataset) -> Grid:
    res = float(ds.attrs.get("res_m", ds.x[1] - ds.x[0]))
    return Grid(float(ds.x[0]) - res / 2, float(ds.y[0]) - res / 2, res, ds.sizes["x"], ds.sizes["y"])


def study_mask(ds: xr.Dataset, buffer_km: float) -> np.ndarray:
    """Cells inside a district or within `buffer_km` of one."""
    inside = ds.district_id.values > 0
    dist = ndimage.distance_transform_edt(~inside) * float(ds.attrs.get("res_m", 1000.0)) / 1000.0
    return dist <= buffer_km


def grid_features(ds: xr.Dataset, names=STATIC) -> np.ndarray:
    """(ny·nx, n_features) static features for every cell, NaNs filled from the nearest valid cell."""
    yy, xx = np.meshgrid(ds.y.values / 1000.0, ds.x.values / 1000.0, indexing="ij")
    cols = []
    for n in names:
        a = xx if n == "x_km" else yy if n == "y_km" else ds[n].values.astype(float)
        bad = ~np.isfinite(a)
        if bad.any():
            idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            a = a[tuple(idx)]
        cols.append(a.ravel())
    return np.column_stack(cols)


def nearest_in_mask(grid: Grid, mask: np.ndarray, x, y) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of each point's cell, moved to the nearest cell inside `mask` if it falls outside."""
    r, c = grid.index(x, y)
    r, c = np.clip(r, 0, grid.ny - 1), np.clip(c, 0, grid.nx - 1)
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return idx[0][r, c], idx[1][r, c]


def well_features(ds: xr.Dataset, mask: np.ndarray, x, y, names=STATIC) -> np.ndarray:
    F = grid_features(ds, names).reshape(ds.sizes["y"], ds.sizes["x"], -1)
    r, c = nearest_in_mask(grid_of(ds), mask, x, y)
    return F[r, c]


def fit_zone_classifier(F: np.ndarray, labels: np.ndarray, seed: int = 0, n_estimators: int = 500,
                        min_samples_leaf: int = 3) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                 class_weight="balanced", random_state=seed, n_jobs=-1)
    return clf.fit(F, labels)


def majority_filter(zone: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """Each masked cell takes the most common zone among masked cells in a window × window square."""
    ids = np.unique(zone[mask])
    votes = np.stack([ndimage.uniform_filter(((zone == i) & mask).astype(float), size=window, mode="constant")
                      for i in ids])
    out = np.where(mask, ids[np.argmax(votes, axis=0)], 0)
    return out


def absorb_small_patches(zone: np.ndarray, mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Patches (connected areas of one zone) smaller than `min_cells` take the most common neighbouring zone."""
    z = zone.copy()
    for _ in range(50):
        changed = False
        patches = []
        for i in np.unique(z[mask]):
            lab, n = ndimage.label((z == i) & mask)
            sizes = np.bincount(lab.ravel())
            patches += [(sizes[j], lab == j) for j in range(1, n + 1) if sizes[j] < min_cells]
        for _, cells in sorted(patches, key=lambda p: p[0]):
            ring = ndimage.binary_dilation(cells) & ~cells & mask
            nb = z[ring]
            if nb.size:
                vals, cnt = np.unique(nb, return_counts=True)
                z[cells] = vals[np.argmax(cnt)]
                changed = True
        if not changed:
            break
    return z


def build_zone_map(clf, ds: xr.Dataset, mask: np.ndarray, window: int = 7, min_patch_km2: float = 50.0,
                   names=STATIC) -> np.ndarray:
    """(ny, nx) zone ids on the model grid (0 outside the mask)."""
    F = grid_features(ds, names)
    raw = np.zeros(mask.size, int)
    raw[mask.ravel()] = clf.predict(F[mask.ravel()])
    raw = raw.reshape(mask.shape)
    smooth = majority_filter(raw, mask, window)
    res_km = float(ds.attrs.get("res_m", 1000.0)) / 1000.0
    return absorb_small_patches(smooth, mask, int(np.ceil(min_patch_km2 / res_km ** 2)))


def assign_zone(zone_map: np.ndarray, ds: xr.Dataset, mask: np.ndarray, x, y) -> np.ndarray:
    r, c = nearest_in_mask(grid_of(ds), mask, x, y)
    return zone_map[r, c]


def count_patches(zone_map: np.ndarray, mask: np.ndarray) -> pd.Series:
    """Number of separate patches per zone (1 = one connected region)."""
    return pd.Series({int(i): ndimage.label((zone_map == i) & mask)[1] for i in np.unique(zone_map[mask])})


def map_assignment_cv(ds: xr.Dataset, mask: np.ndarray, x, y, labels: np.ndarray, groups: np.ndarray,
                      n_folds: int = 5, seed: int = 0, **map_kw) -> np.ndarray:
    """Leave-blocks-out check of the whole map rule: for each fold, fit on the other folds' wells, build the map,
    and read the zone at the left-out wells. Returns the predicted zone per well (compare with `labels`)."""
    rng = np.random.default_rng(seed)
    ug = rng.permutation(np.unique(groups))
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    fold = np.array([fold_of[g] for g in groups])
    F = well_features(ds, mask, x, y)
    pred = np.zeros(len(labels), int)
    clf_kw = {k: map_kw.pop(k) for k in ("n_estimators", "min_samples_leaf") if k in map_kw}
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        clf = fit_zone_classifier(F[tr], labels[tr], seed=seed, **clf_kw)
        zm = build_zone_map(clf, ds, mask, **map_kw)
        pred[te] = assign_zone(zm, ds, mask, np.asarray(x)[te], np.asarray(y)[te])
    return pred


def zone_polygons(zone_map: np.ndarray, ds: xr.Dataset, mask: np.ndarray):
    """GeoDataFrame with one (multi)polygon per zone, in UTM 43N, with its area."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    g = grid_of(ds)
    north_up = zone_map[::-1].astype(np.int32)
    geoms = [(shape(s), int(v)) for s, v in shapes(north_up, mask=mask[::-1], transform=g.transform_north_up())]
    gdf = gpd.GeoDataFrame({"zone": [v for _, v in geoms]}, geometry=[s for s, _ in geoms], crs="EPSG:32643")
    gdf = gdf.dissolve("zone").reset_index()
    gdf["area_km2"] = gdf.area / 1e6
    return gdf
