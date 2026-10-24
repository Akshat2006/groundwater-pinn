"""M4: spatial block splits, behaviour features, zones and zone maps.

    python scripts/build_zones.py          -> data/processed/{splits,behaviour_features,zone_assign}.parquet,
                                              zones.nc, zone_map.tif, zones.gpkg, zoning_summary.json
    python scripts/check_m4.py             -> reports/m4_zoning.md + figures

Run after scripts/build_master.py (which resets the zone columns of master.parquet and wells.parquet).

For each block layout (configs/splits.yaml) and zoning method (configs/zoning.yaml) the zones are built from the
wells in *training* blocks only, using training years only. Validation and test wells get their zone from the map,
the same way any unmonitored place would. The main zoning (config `main`, layout 0) is written into
master.parquet and wells.parquet.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.splits import spatial_block_split  # noqa: E402
from src.features.behaviour import behaviour_features  # noqa: E402
from src.utils.config import data_dir, load_config  # noqa: E402
from src.zoning.assign import (LOCATION, STATIC, assign_zone, build_zone_map, count_patches,  # noqa: E402
                               fit_zone_classifier, grid_of, map_assignment_cv, study_mask, well_features,
                               zone_polygons)
from src.zoning.cluster import (FeatureScaler, choose_k, cluster_wells, merge_small_zones,  # noqa: E402
                                neighbour_graph, order_by)


def zones_for(k, X, feats, train_wells, wells, ds, mask, zcfg, mcfg, method, graph, groups):
    """Cluster, merge, map and check one (method, layout, k). Returns behaviour labels, map, well zones, summary."""
    seed = zcfg["seed"]
    map_kw = {k2: v for k2, v in zcfg["map"].items() if k2 != "mask_buffer_km"}
    kind = mcfg["map_from"]
    names = LOCATION if kind == "nearest" else STATIC
    lab = merge_small_zones(cluster_wells(X, k, seed, method, graph), X, zcfg["min_zone_wells"],
                            graph if method == "contiguous" else None)
    lab = order_by(lab, feats.loc[train_wells, "trend_m_per_yr"].to_numpy())
    xy = wells.loc[train_wells, ["x_m", "y_m"]].to_numpy()
    if lab.max() == 1:
        zmap = np.where(mask, 1, 0)
        cv_pred = np.ones(len(lab), int)
    else:
        F = well_features(ds, mask, xy[:, 0], xy[:, 1], names)
        clf = fit_zone_classifier(F, lab, seed=seed, kind=kind, **zcfg["classifier"])
        zmap = build_zone_map(clf, ds, mask, names=names, **map_kw)
        cv_pred = map_assignment_cv(ds, mask, xy[:, 0], xy[:, 1], lab, groups, n_folds=zcfg["cv_folds"], seed=seed,
                                    kind=kind, names=names, **zcfg["classifier"], **map_kw)
    in_map = assign_zone(zmap, ds, mask, xy[:, 0], xy[:, 1])
    all_zone = assign_zone(zmap, ds, mask, wells.x_m.to_numpy(), wells.y_m.to_numpy())
    share = np.bincount(lab)[1:] / len(lab)
    summary = {
        "k_requested": int(k), "k": int(lab.max()),
        "wells_per_zone": {int(i): int((lab == i).sum()) for i in np.unique(lab)},
        "cv_accuracy": float((cv_pred == lab).mean()),
        "chance_accuracy": float(share.max()),
        "in_sample_map_agreement": float((in_map == lab).mean()),
        "patches": {int(z): int(n) for z, n in count_patches(zmap, mask).items()},
        "area_km2": {int(i): int(((zmap == i) & mask).sum() * (grid_of(ds).res / 1000) ** 2)
                     for i in np.unique(zmap[mask])},
    }
    return lab, zmap, all_zone, summary


def main_selection(summary: dict, zcfg: dict, layout: int) -> tuple[str, int]:
    method = zcfg["main"]["method"]
    k = zcfg["main"]["k"]
    if k == "chosen":
        k = summary["layouts"][str(layout)]["methods"][method]["chosen_k"]
    return method, int(k)


def main() -> None:
    t0 = time.time()
    proc = data_dir() / "processed"
    cfg = load_config(ROOT / "configs" / "study_area.yaml")
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    zcfg = load_config(ROOT / "configs" / "zoning.yaml")
    master = pd.read_parquet(proc / "master.parquet")
    wells = pd.read_parquet(proc / "wells.parquet").set_index("well_id")
    ds = xr.open_dataset(proc / "static_1km.nc").load()
    mask = study_mask(ds, zcfg["map"]["mask_buffer_km"])

    shallow = master[master.depth_class == "shallow"]
    eligible = wells.index.isin(shallow.loc[shallow.head_m.notna(), "well_id"].unique())
    feats = behaviour_features(shallow, tuple(cfg["period"]["train"]), zcfg["min_years"])
    print(f"behaviour features for {len(feats)} wells ({len(shallow.well_id.unique())} shallow wells in total)")

    methods = list(zcfg["methods"])
    split_rows, assign_rows, summary = [], [], {"layouts": {}}
    maps = {}
    for layout in scfg["layouts"]:
        sp = spatial_block_split(wells.x_m, wells.y_m, eligible, scfg["block_km"], scfg["test_frac"],
                                 scfg["val_frac"], scfg["tol"], seed=layout)
        sp.index = wells.index
        split_rows.append(sp.assign(layout=layout).reset_index())
        train_wells = feats.index[sp.loc[feats.index, "split_space"].eq("train")]
        groups = sp.loc[train_wells, "block"].to_numpy()
        lay = {"n_cluster_wells": int(len(train_wells)),
               "split_counts": {s: int(((sp.split_space == s) & eligible).sum()) for s in ("train", "val", "test")},
               "tier1_counts": {s: int(((sp.split_space == s) & wells.tier1).sum()) for s in ("train", "val", "test")},
               "methods": {}}
        for method in methods:
            mcfg = zcfg["methods"][method]
            fnames = mcfg["features"]
            scaler = FeatureScaler.fit(feats.loc[train_wells, fnames].to_numpy(), zcfg["winsor_pct"])
            X = scaler.transform(feats.loc[train_wells, fnames].to_numpy())
            graph = (neighbour_graph(wells.loc[train_wells, ["x_m", "y_m"]].to_numpy(), mcfg["neighbours"])
                     if method == "contiguous" else None)
            sel = choose_k(X, range(zcfg["k_range"][0], zcfg["k_range"][1] + 1), zcfg["seed"], method, graph)
            ml = {"features": fnames, "chosen_k": sel.chosen, "elbow_k": sel.elbow,
                  "k_table": sel.table.round(4).to_dict("records"),
                  "scaler": {k2: getattr(scaler, k2).round(5).tolist() for k2 in ("lo", "hi", "mean", "sd")},
                  "zones": {}}
            for k in sorted(set(zcfg["sweep_k"]) | {sel.chosen}):
                lab, zmap, all_zone, s = zones_for(k, X, feats, train_wells, wells, ds, mask, zcfg, mcfg, method,
                                                   graph, groups)
                ml["zones"][str(k)] = s
                maps[(method, layout, k)] = zmap
                behaviour = pd.Series(lab, index=train_wells)
                assign_rows.append(pd.DataFrame({"well_id": wells.index, "method": method, "layout": layout, "k": k,
                                                 "zone_behaviour": behaviour.reindex(wells.index).astype("Int64").values,
                                                 "zone_map": all_zone}))
                print(f"{method:10s} layout {layout} k={k:>2} -> {s['k']} zones {s['wells_per_zone']}  map accuracy "
                      f"{s['cv_accuracy']:.2f} (chance {s['chance_accuracy']:.2f})  patches {sum(s['patches'].values())}"
                      + ("  <- chosen" if k == sel.chosen else ""), flush=True)
            lay["methods"][method] = ml
        summary["layouts"][str(layout)] = lay

    splits = pd.concat(split_rows, ignore_index=True)
    assign = pd.concat(assign_rows, ignore_index=True)
    splits.to_parquet(proc / "splits.parquet", index=False)
    assign.to_parquet(proc / "zone_assign.parquet", index=False)
    feats.reset_index().to_parquet(proc / "behaviour_features.parquet", index=False)

    layouts = list(scfg["layouts"])
    ks = sorted({k for (_, _, k) in maps})
    cube = np.zeros((len(methods), len(layouts), len(ks), ds.sizes["y"], ds.sizes["x"]), np.int8)
    for (m, lo, k), zm in maps.items():
        cube[methods.index(m), layouts.index(lo), ks.index(k)] = zm
    xr.Dataset({"zone": (("method", "layout", "k", "y", "x"), cube), "mask": (("y", "x"), mask)},
               coords={"method": methods, "layout": layouts, "k": ks, "y": ds.y, "x": ds.x},
               attrs={"crs": "EPSG:32643", "note": "0 = outside the mapped area or k not built; zone 1 = steepest "
                                                   "mean decline"}).to_netcdf(proc / "zones.nc")

    main_layout = layouts[0]
    method, k_main = main_selection(summary, zcfg, main_layout)
    summary.update(main_layout=main_layout, main_method=method, main_k=k_main, seconds=round(time.time() - t0))
    zm = maps[(method, main_layout, k_main)]
    import rasterio
    g = grid_of(ds)
    with rasterio.open(proc / "zone_map.tif", "w", driver="GTiff", height=g.ny, width=g.nx, count=1, dtype="int8",
                       crs="EPSG:32643", transform=g.transform_north_up(), nodata=0) as dst:
        dst.write(zm[::-1].astype(np.int8), 1)
    zone_polygons(zm, ds, mask).to_file(proc / "zones.gpkg", driver="GPKG")

    sel_main = assign[(assign.method == method) & (assign.layout == main_layout) & (assign.k == k_main)].set_index("well_id")
    sp_main = splits[splits.layout == main_layout].set_index("well_id").split_space
    for name, table in (("master", master), ("wells", wells.reset_index())):
        table["split_space"] = table.well_id.map(sp_main)
        table["zone_behaviour"] = table.well_id.map(sel_main.zone_behaviour).astype("Int64")
        table["zone_map"] = table.well_id.map(sel_main.zone_map).astype("Int64")
        table.to_parquet(proc / f"{name}.parquet", index=False)
    (proc / "zoning_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"main: {method}, layout {main_layout}, k = {k_main}; wrote zones to {proc} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
