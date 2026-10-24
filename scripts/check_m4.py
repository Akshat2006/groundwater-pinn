"""M4 report: splits, behaviour features, k selection, zone map and how well the map assigns zones.

    python scripts/check_m4.py      -> reports/m4_zoning.md, reports/figures/m4_*.png, reports/tables/m4_zones.csv

Reads the outputs of scripts/build_zones.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.behaviour import FEATURES  # noqa: E402
from src.utils.config import data_dir, load_config  # noqa: E402
from src.zoning.assign import study_mask, well_features  # noqa: E402
from src.zoning.report import block_cv_r2, correlation_by_distance, neighbour_correlation  # noqa: E402

FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated, fixed order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"amplitude_m": "Seasonal swing (m)", "trend_m_per_yr": "Long-term trend (m/yr)",
         "monsoon_resp_m_per_100mm": "Monsoon response (m per 100 mm)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


def plot_k(table: pd.DataFrame, elbow: int, chosen: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), dpi=150)
    for ax, col, title in ((axes[0], "inertia", "Within-zone spread (lower = tighter zones)"),
                           (axes[1], "silhouette", "Silhouette (higher = better separated)")):
        ax.plot(table.k, table[col], color=ZONE_COLORS[0], lw=2, marker="o", ms=4)
        for k, name in ((elbow, "elbow"), (chosen, "chosen")):
            v = table.loc[table.k == k, col].iloc[0]
            ax.plot([k], [v], "o", ms=9, mfc="none", mec=INK, mew=1.2)
            ax.annotate(name, (k, v), xytext=(6, 6), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("number of zones k (before merging small zones)", fontsize=8, color=MUTED)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "m4_k_selection.png")
    plt.close(fig)


def plot_map(zmap: np.ndarray, mask: np.ndarray, ds: xr.Dataset, wells: pd.DataFrame, k: int) -> None:
    ext = [ds.x.values[0] / 1e3 - 0.5, ds.x.values[-1] / 1e3 + 0.5, ds.y.values[0] / 1e3 - 0.5, ds.y.values[-1] / 1e3 + 0.5]
    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=150)
    shown = np.where(mask, zmap, np.nan).astype(float)
    cmap = ListedColormap(ZONE_COLORS[:k])
    ax.imshow(shown, origin="lower", extent=ext, cmap=cmap, vmin=0.5, vmax=k + 0.5, alpha=0.35, interpolation="nearest")
    dist = ds.district_id.values
    ax.contour(ds.x.values / 1e3, ds.y.values / 1e3, (dist > 0).astype(float), levels=[0.5], colors=MUTED, linewidths=0.8)
    edges = (ndimage.maximum_filter(dist, 2) != ndimage.minimum_filter(dist, 2)) & (dist > 0)
    ax.imshow(np.where(edges, 1.0, np.nan), origin="lower", extent=ext, cmap=ListedColormap([GRID]), interpolation="nearest")
    tr = wells[wells.zone_behaviour.notna()]
    for z in range(1, k + 1):
        s = tr[tr.zone_behaviour == z]
        ax.scatter(s.x_m / 1e3, s.y_m / 1e3, s=22, color=ZONE_COLORS[z - 1], edgecolor=SURFACE, lw=0.8, zorder=3,
                   label=f"zone {z}")
        lab, n = ndimage.label((zmap == z) & mask)
        if n:
            big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            r, c = ndimage.center_of_mass(lab == big)
            ax.text(ds.x.values[int(c)] / 1e3, ds.y.values[int(r)] / 1e3, str(z), fontsize=13, weight="bold", color=INK,
                    ha="center", va="center", zorder=4)
    held = wells[wells.split_space.isin(["val", "test"]) & wells.has_head]
    ax.scatter(held.x_m / 1e3, held.y_m / 1e3, s=12, marker="x", color=INK, lw=0.8, zorder=3,
               label="held-out well (val/test block)")
    ax.set_aspect("equal")
    ax.set_xlabel("UTM 43N easting (km)", fontsize=8, color=MUTED)
    ax.set_ylabel("northing (km)", fontsize=8, color=MUTED)
    ax.set_title("Zone map (shading) and the wells that defined the zones (dots)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "m4_zone_map.png")
    plt.close(fig)


def plot_features(feats: pd.DataFrame, k: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for z in range(1, k + 1):
        s = feats[feats.zone == z]
        ax.scatter(s.trend_m_per_yr, s.amplitude_m, s=24, color=ZONE_COLORS[z - 1], edgecolor=SURFACE, lw=0.8,
                   label=f"zone {z}")
        ax.annotate(f"zone {z}", (s.trend_m_per_yr.median(), s.amplitude_m.median()), fontsize=8, color=INK,
                    weight="bold", ha="center", xytext=(0, 9), textcoords="offset points")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(LABEL["trend_m_per_yr"] + "  (left = falling faster)", fontsize=8, color=INK)
    ax.set_ylabel(LABEL["amplitude_m"] + "\n(below 0 = falls through the monsoon)", fontsize=8, color=INK)
    ax.set_title("Training wells by behaviour zone", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "m4_zone_features.png")
    plt.close(fig)


def main() -> None:
    proc = data_dir() / "processed"
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    zcfg = load_config(ROOT / "configs" / "zoning.yaml")
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    summ = json.loads((proc / "zoning_summary.json").read_text(encoding="utf-8"))
    L, K = summ["main_layout"], summ["main_k"]
    lay = summ["layouts"][str(L)]
    main = lay["zones"][str(K)]
    k_eff = main["k"]
    master = pd.read_parquet(proc / "master.parquet")
    wells = pd.read_parquet(proc / "wells.parquet")
    feats = pd.read_parquet(proc / "behaviour_features.parquet").set_index("well_id")
    ds = xr.open_dataset(proc / "static_1km.nc").load()
    zones = xr.open_dataset(proc / "zones.nc").load()
    mask = zones["mask"].values.astype(bool)
    zmap = zones.zone.sel(layout=L, k=K).values

    shallow_head = master[(master.depth_class == "shallow") & master.head_m.notna()]
    wells["has_head"] = wells.well_id.isin(shallow_head.well_id) & (wells.depth_class == "shallow")
    w = wells.set_index("well_id")
    feats["zone"] = w.loc[feats.index, "zone_behaviour"]
    feats["split_space"] = w.loc[feats.index, "split_space"]

    plot_k(pd.DataFrame(lay["k_table"]), lay["elbow_k"], lay["chosen_k"])
    plot_map(zmap, mask, ds, wells, k_eff)
    plot_features(feats[feats.zone.notna()].astype({"zone": int}), k_eff)

    # spatial predictability of each feature (all wells with features, layout-0 blocks)
    xy = w.loc[feats.index, ["x_m", "y_m"]].to_numpy() / 1000
    blocks = pd.read_parquet(proc / "splits.parquet").query("layout == @L").set_index("well_id").loc[feats.index, "block"].to_numpy()
    static = well_features(ds, mask, xy[:, 0] * 1000, xy[:, 1] * 1000)
    pred_rows = []
    for f in FEATURES:
        v = feats[f].to_numpy()
        cv = block_cv_r2(v, static, xy, blocks, zcfg["cv_folds"], zcfg["seed"])
        band = correlation_by_distance(v, xy)
        pred_rows.append({"Feature": LABEL[f], "Nearest well alike": f"{neighbour_correlation(v, xy):+.2f}",
                          **{f"Alike at {b}": f"{c:+.2f}" for b, c in band.items()},
                          "Predicted in unseen block (static features)": f"{cv['static features']:+.2f}",
                          "Predicted in unseen block (5 nearest wells)": f"{cv['5 nearest wells']:+.2f}"})
    pred = pd.DataFrame(pred_rows)

    # zone table (main layout and k)
    tr_wells = shallow_head[shallow_head.split_time == "train"].well_id.unique()
    ztab = []
    for z in range(1, k_eff + 1):
        s = feats[feats.zone == z]
        in_map = w[w.zone_map == z]
        ztab.append({"Zone": z, "Wells that defined it": len(s),
                     "Training wells in map zone": int((in_map.split_space.eq("train") & in_map.index.isin(tr_wells)).sum()),
                     "Val wells": int((in_map.split_space.eq("val") & in_map.has_head).sum()),
                     "Test wells": int((in_map.split_space.eq("test") & in_map.has_head).sum()),
                     "Area (km²)": main["area_km2"].get(str(z), 0), "Patches": main["patches"].get(str(z), 0),
                     "Trend (m/yr)": f"{s.trend_m_per_yr.median():+.2f}", "Seasonal swing (m)": f"{s.amplitude_m.median():+.2f}",
                     "Monsoon response (m/100 mm)": f"{s.monsoon_resp_m_per_100mm.median():+.2f}"})
    ztab = pd.DataFrame(ztab)
    ztab.to_csv(TAB / "m4_zones.csv", index=False)

    sweep = []
    for lo, l in summ["layouts"].items():
        for kk, s in l["zones"].items():
            if int(kk) == 1:
                continue
            sweep.append({"Layout": lo, "k asked": kk + (" (chosen)" if int(kk) == l["chosen_k"] else ""),
                          "Zones after merging": s["k"], "Smallest zone (wells)": min(s["wells_per_zone"].values()),
                          "Map accuracy, unseen blocks": f"{100 * s['cv_accuracy']:.0f}%",
                          "Guessing the biggest zone": f"{100 * s['chance_accuracy']:.0f}%",
                          "Patches (total)": sum(s["patches"].values())})
    sweep = pd.DataFrame(sweep)

    splits = pd.DataFrame([{"Layout": lo, **{f"{s} wells": l["split_counts"][s] for s in ("train", "val", "test")},
                            **{f"{s} Tier 1": l["tier1_counts"][s] for s in ("train", "val", "test")},
                            "Wells clustered": l["n_cluster_wells"], "Chosen k": l["chosen_k"]}
                           for lo, l in summ["layouts"].items()])
    n_elig = sum(lay["split_counts"].values())
    min_zone = min(main["wells_per_zone"].values())
    ok_size = min_zone >= zcfg["min_zone_wells"]

    md = [
        "# M4 report: zoning",
        "",
        "*Generated by `scripts/check_m4.py` from the outputs of `scripts/build_zones.py`.*",
        "",
        "## Summary",
        "",
        f"- **Zones:** {k_eff} zones on the main layout (k = {K} asked, small zones merged). Smallest zone: {min_zone} wells. Every zone meets the minimum of {zcfg['min_zone_wells']}.",
        f"- **Zone map:** saved as `data/processed/zone_map.tif` and `zones.gpkg`. Zones for every layout and every k of the sweep are in `zones.nc`.",
        f"- **Assignment accuracy:** {100 * main['cv_accuracy']:.0f}%, against {100 * main['chance_accuracy']:.0f}% for always guessing the biggest zone. **The map rule is barely better than guessing.**",
        "- **Why:** well behaviour is only partly a regional pattern. The long-term trend is the most mappable feature, but even it is only about one-third predictable in an unseen 20 km block. The monsoon response is not predictable at all (see *Is behaviour a map?* below).",
        "- **What this means:** zones describe real differences between the wells that have history, but a held-out well or an unmonitored place often lands in the wrong zone. This is weakness 2 in the brief (§6.4), now measured. The early zoning preview (after the M5 baselines) will show whether zoning still helps prediction.",
        "",
        "## 1. Spatial block splits",
        "",
        f"The area is cut into {scfg['block_km']} km blocks. Whole blocks are held out: about {100 * scfg['test_frac']:.0f}% of wells for testing and {100 * scfg['val_frac']:.0f}% for validation (tuning only).",
        "",
        f"Shares are counted over the {n_elig} shallow wells with at least one usable water level. Layout 0 is the main layout, fixed in advance. Layouts 1 and 2 are the repeats for experiment E16.",
        "",
        *md_table(splits),
        "",
        "*Tier 1* = wells with at least 5 training years that have both a pre- and a post-monsoon reading. Only these get behaviour features.",
        "",
        "## 2. Behaviour features (training years 2005–2018 only)",
        "",
        "| Feature | What it measures |",
        "|---|---|",
        "| Seasonal swing | mean rise from the pre-monsoon (Mar–Jun) to the post-monsoon (Oct–Dec) reading. Below 0 means the level falls through the monsoon, usually because paddy pumping outweighs recharge |",
        "| Long-term trend | Theil–Sen slope of the pre-monsoon level (m per year). Theil–Sen takes the median of all pairwise slopes, so a few bad readings cannot swing it |",
        "| Monsoon response | how much more the level rises in a wetter monsoon, per 100 mm of extra June–September rain |",
        "",
        f"{len(feats)} wells have features. The median well falls {abs(feats.trend_m_per_yr.median()):.2f} m a year, and the fastest falls {abs(feats.trend_m_per_yr.min()):.1f} m a year.",
        "",
        "Before clustering, each feature is clipped to its 2nd–98th percentile and standardised, so all three count equally.",
        "",
        "## 3. Choosing the number of zones",
        "",
        "k-means is run for k = 2 to 12 on the training wells of each layout. The rule was fixed before looking at the zones: take the *elbow* (the k after which more zones stop tightening the groups much), then the best *silhouette* among the elbow and its two neighbours.",
        "",
        f"Zones smaller than {zcfg['min_zone_wells']} wells are then merged into the zone with the nearest centre. Zones are numbered from the steepest decline (zone 1) to the slowest.",
        "",
        f"Main layout: elbow k = {lay['elbow_k']}, chosen k = {lay['chosen_k']}, giving {k_eff} zones after merging.",
        "",
        "![k selection](figures/m4_k_selection.png)",
        "",
        "## 4. The zones",
        "",
        *md_table(ztab),
        "",
        "*Training wells in map zone* = every shallow well in a training block that has training-period readings and falls inside the zone on the map. These are the wells the zone's PINN will learn from.",
        "",
        "![zone features](figures/m4_zone_features.png)",
        "",
        "![zone map](figures/m4_zone_map.png)",
        "",
        "## 5. How the map is made, and how often it assigns the right zone",
        "",
        "1. A random forest learns each training well's zone from features known everywhere: easting, northing, elevation, cropland, built-up and tree cover, distance to the nearest canal, and canal density. Values are read from the well's 1 km cell.",
        f"2. It labels every 1 km cell of the 16 districts plus a {zcfg['map']['mask_buffer_km']} km buffer.",
        f"3. A {zcfg['map']['window']} × {zcfg['map']['window']} km majority filter removes speckle, and patches under {zcfg['map']['min_patch_km2']} km² join the zone around them.",
        "4. Every well, including held-out ones, takes the zone of its map cell.",
        "",
        "**Assignment accuracy** is measured the way the map will be used. The training wells are split into 5 groups of whole 20 km blocks. For each group, the map is rebuilt without it, and we check whether its wells land in their own behaviour zone.",
        "",
        "*Guessing the biggest zone* is the score of calling every well the most common zone, so only accuracy above it means the map has learned something.",
        "",
        *md_table(sweep),
        "",
        "## 6. Is behaviour a map? (why accuracy is low)",
        "",
        "For each feature, the table shows two things:",
        "- *Alike* columns: how similar wells are at different distances apart (a correlation: 1 = identical, 0 = unrelated).",
        "- *Predicted* columns: how well the value of a well in an unseen block is predicted (R²: 1 = perfectly, 0 = no better than the average of all wells).",
        "",
        *md_table(pred),
        "",
        "**Reading:**",
        "- **Trend and seasonal swing are regional patterns.** Neighbouring wells are fairly alike, and the likeness fades to nothing by about 40 km. But with wells about 6 km apart and blocks of 20 km, only about a third of the trend can be predicted in an unseen block.",
        "- **The monsoon response is noise at the well level.** It is a slope through only 5–14 points per well, so it differs randomly between neighbours.",
        "- **Both ways of predicting score the same,** whether from static features or from the 5 nearest wells. So a better classifier would not help much. The limit is how patchy the behaviour is relative to the well spacing.",
        "",
        "## 7. Done criteria (PLAN.md §5 M4)",
        "",
        "| Criterion | Status |",
        "|---|---|",
        f"| Every zone has at least 10–15 training wells | {'✅' if ok_size else '❌'} smallest zone: {min_zone} wells |",
        "| Zone map saved | ✅ `zone_map.tif`, `zones.gpkg`, `zones.nc` |",
        f"| Assignment accuracy reported | ✅ {100 * main['cv_accuracy']:.0f}% (guessing: {100 * main['chance_accuracy']:.0f}%) — **below the 60% warning level in the risk register** |",
        "| Early zoning preview | ⏳ needs the M5 baselines (zoned vs unzoned random forest and bucket model on validation blocks) |",
        "",
        "## Plain-language glossary",
        "",
        "- **k-means:** sorts wells into k groups so that wells in a group have similar features.",
        "- **Spatial block hold-out:** hiding whole areas, not single wells, so the test measures prediction in places the model has never seen.",
        "- **Correlation:** from −1 to 1; how strongly two sets of numbers move together.",
        "- **R²:** the share of the differences between wells that a prediction explains (1 = all, 0 = none).",
        "- **Random forest:** many decision trees voting; a standard, robust classifier.",
    ]
    (ROOT / "reports" / "m4_zoning.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"layout {L}: {k_eff} zones, smallest {min_zone} wells, map accuracy {100 * main['cv_accuracy']:.0f}% "
          f"(guessing {100 * main['chance_accuracy']:.0f}%)")
    print(pred.to_string(index=False))
    print("wrote reports/m4_zoning.md")


if __name__ == "__main__":
    main()
