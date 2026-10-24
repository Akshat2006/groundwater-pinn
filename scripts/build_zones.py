"""M4: spatial block splits, behaviour features, zones and zone maps.

    python scripts/build_zones.py          -> data/processed/{splits,behaviour_features,zone_assign}.parquet,
                                              zones.nc, zone_map.tif, zones.gpkg, zoning_summary.json
    python scripts/check_m4.py             -> reports/m4_zoning.md + figures

Run after scripts/build_master.py (which resets the zone columns of master.parquet and wells.parquet).

For each block layout (configs/splits.yaml) the zones are built from the wells in *training* blocks only, using
training years only. Validation and test wells get their zone from the map, the same way any unmonitored place
would. Layout 0 with its chosen k is the main result and is written into master.parquet and wells.parquet.
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
from src.zoning.assign import (assign_zone, build_zone_map, count_patches, fit_zone_classifier, grid_of,  # noqa: E402
                               map_assignment_cv, study_mask, well_features, zone_polygons)
from src.zoning.cluster import FeatureScaler, choose_k, cluster_wells, merge_small_zones, order_by  # noqa: E402


def zones_for(k, X, feats, train_wells, wells, ds, mask, zcfg, groups):
    """Cluster, merge, map and check one (layout, k). Returns behaviour labels, map, well zones and a summary."""
    seed = zcfg["seed"]
    map_kw = {k2: v for k2, v in zcfg["map"].items() if k2 != "mask_buffer_km"}
    lab = merge_small_zones(cluster_wells(X, k, seed), X, zcfg["min_zone_wells"])
    lab = order_by(lab, feats.loc[train_wells, "trend_m_per_yr"].to_numpy())
    xy = wells.loc[train_wells, ["x_m", "y_m"]].to_numpy()
    if lab.max() == 1:
        zmap = np.where(mask, 1, 0)
        cv_pred = np.ones(len(lab), int)
    else:
        F = well_features(ds, mask, xy[:, 0], xy[:, 1])
        clf = fit_zone_classifier(F, lab, seed=seed, **zcfg["classifier"])
        zmap = build_zone_map(clf, ds, mask, **map_kw)
        cv_pred = map_assignment_cv(ds, mask, xy[:, 0], xy[:, 1], lab, groups, n_folds=zcfg["cv_folds"], seed=seed,
                                    **zcfg["classifier"], **map_kw)
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
    return lab, zmap, all_zone, cv_pred, summary


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
    fnames = zcfg["features"]
    print(f"behaviour features for {len(feats)} wells ({len(shallow.well_id.unique())} shallow wells in total)")

    split_rows, assign_rows, summary = [], [], {"layouts": {}}
    ks_all = sorted(set(zcfg["sweep_k"]))
    maps = {}
    for layout in scfg["layouts"]:
        sp = spatial_block_split(wells.x_m, wells.y_m, eligible, scfg["block_km"], scfg["test_frac"],
                                 scfg["val_frac"], scfg["tol"], seed=layout)
        sp.index = wells.index
        split_rows.append(sp.assign(layout=layout).reset_index())
        train_wells = feats.index[sp.loc[feats.index, "split_space"].eq("train")]
        scaler = FeatureScaler.fit(feats.loc[train_wells, fnames].to_numpy(), zcfg["winsor_pct"])
        X = scaler.transform(feats.loc[train_wells, fnames].to_numpy())
        sel = choose_k(X, range(zcfg["k_range"][0], zcfg["k_range"][1] + 1), zcfg["seed"])
        groups = sp.loc[train_wells, "block"].to_numpy()
        lay = {"chosen_k": sel.chosen, "elbow_k": sel.elbow, "k_table": sel.table.round(4).to_dict("records"),
               "n_cluster_wells": int(len(train_wells)),
               "split_counts": {s: int(((sp.split_space == s) & eligible).sum()) for s in ("train", "val", "test")},
               "tier1_counts": {s: int(((sp.split_space == s) & wells.tier1).sum()) for s in ("train", "val", "test")},
               "scaler": {k2: getattr(scaler, k2).round(5).tolist() for k2 in ("lo", "hi", "mean", "sd")},
               "zones": {}}
        for k in sorted(set(ks_all) | {sel.chosen}):
            lab, zmap, all_zone, cv_pred, s = zones_for(k, X, feats, train_wells, wells, ds, mask, zcfg, groups)
            lay["zones"][str(k)] = s
            maps[(layout, k)] = zmap
            behaviour = pd.Series(lab, index=train_wells)
            assign_rows.append(pd.DataFrame({"well_id": wells.index, "layout": layout, "k": k,
                                             "zone_behaviour": behaviour.reindex(wells.index).astype("Int64").values,
                                             "zone_map": all_zone}))
            print(f"layout {layout} k={k:>2} -> {s['k']} zones {s['wells_per_zone']}  map accuracy "
                  f"{s['cv_accuracy']:.2f} (chance {s['chance_accuracy']:.2f})  patches {s['patches']}"
                  + ("  <- chosen" if k == sel.chosen else ""), flush=True)
        summary["layouts"][str(layout)] = lay

    splits = pd.concat(split_rows, ignore_index=True)
    assign = pd.concat(assign_rows, ignore_index=True)
    splits.to_parquet(proc / "splits.parquet", index=False)
    assign.to_parquet(proc / "zone_assign.parquet", index=False)
    feats.reset_index().to_parquet(proc / "behaviour_features.parquet", index=False)

    layouts = list(scfg["layouts"])
    ks = sorted({k for (_, k) in maps})
    cube = np.zeros((len(layouts), len(ks), ds.sizes["y"], ds.sizes["x"]), np.int8)
    for (lo, k), zm in maps.items():
        cube[layouts.index(lo), ks.index(k)] = zm
    xr.Dataset({"zone": (("layout", "k", "y", "x"), cube), "mask": (("y", "x"), mask)},
               coords={"layout": layouts, "k": ks, "y": ds.y, "x": ds.x},
               attrs={"crs": "EPSG:32643", "note": "0 = outside the mapped area; zone 1 = steepest mean decline"}
               ).to_netcdf(proc / "zones.nc")

    main_layout = layouts[0]
    k_main = summary["layouts"][str(main_layout)]["chosen_k"]
    summary.update(main_layout=main_layout, main_k=k_main, features=fnames, seconds=round(time.time() - t0))
    zm = maps[(main_layout, k_main)]
    import rasterio
    g = grid_of(ds)
    with rasterio.open(proc / "zone_map.tif", "w", driver="GTiff", height=g.ny, width=g.nx, count=1, dtype="int8",
                       crs="EPSG:32643", transform=g.transform_north_up(), nodata=0) as dst:
        dst.write(zm[::-1].astype(np.int8), 1)
    zone_polygons(zm, ds, mask).to_file(proc / "zones.gpkg", driver="GPKG")

    sel_main = assign[(assign.layout == main_layout) & (assign.k == k_main)].set_index("well_id")
    sp_main = splits[splits.layout == main_layout].set_index("well_id").split_space
    for name, table in (("master", master), ("wells", wells.reset_index())):
        table["split_space"] = table.well_id.map(sp_main)
        table["zone_behaviour"] = table.well_id.map(sel_main.zone_behaviour).astype("Int64")
        table["zone_map"] = table.well_id.map(sel_main.zone_map).astype("Int64")
        table.to_parquet(proc / f"{name}.parquet", index=False)
    (proc / "zoning_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"main: layout {main_layout}, k = {k_main}; wrote zones to {proc} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
