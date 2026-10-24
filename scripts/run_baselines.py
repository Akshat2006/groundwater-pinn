"""M5: baseline models on the three prediction tasks (PLAN.md §5 M5, §7 E1–E3).

    python scripts/run_baselines.py     -> reports/tables/baselines.csv, reports/m5_baselines.md,
                                           data/processed/predictions/baselines_L{layout}_{stage}.parquet

Tasks (src/evaluation/tasks.py): A = monitored wells in later years, B = held-out blocks in the training years,
C = held-out blocks in later years. Every model learns only from training-block wells in 2005–2018.

| Model | Tasks | Needs |
|---|---|---|
| persistence, seasonal mean, seasonal trend | A | the well's own history |
| ordinary kriging per survey | B | other wells measured in the same survey |
| trend + kriging | C | each training well's seasonal trend carried forward, then kriged to the target place |
| random forest, LightGBM | A, B, C | location, land cover, time, rainfall |
| depth kriging, depth trend + kriging | B, C | as above, but on depth to water; the target's ground elevation is added back (how CGWB maps water levels) |
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summary  # noqa: E402
from src.evaluation.tasks import shallow_levels, target_rows, training_rows, with_layout  # noqa: E402
from src.models.baselines import kriging, naive, trees  # noqa: E402
from src.utils.config import data_dir, load_config  # noqa: E402

TASK_MODELS = {
    "A": ["persistence", "seasonal mean", "seasonal trend", "random forest", "LightGBM"],
    "B": ["kriging", "depth kriging", "random forest", "LightGBM"],
    "C": ["trend + kriging", "depth trend + kriging", "random forest", "LightGBM"],
}
TASK_NAME = {"A": "A: forecast at monitored wells", "B": "B: unmonitored places", "C": "C: unmonitored places, later years"}


def trend_then_krige(train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Carry every training well's seasonal trend to each target survey, then krige those values to the targets."""
    keys = kriging.survey_key(target)
    pseudo = []
    for key, tgt in target.groupby(keys):
        wells = train.groupby("well_id").agg(x_m=("x_m", "first"), y_m=("y_m", "first")).reset_index()
        wells["date"] = tgt.date.median()
        wells["season"] = tgt.season.iloc[0]
        pseudo.append(wells)
    pseudo = pd.concat(pseudo, ignore_index=True)
    pseudo["head_m"] = naive.seasonal_trend(train, pseudo).to_numpy()
    pseudo = pseudo[pseudo.head_m.notna()]
    return kriging.ordinary_kriging(pseudo, target)


def via_depth(model: str, train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """The same model on head − ground elevation (= minus depth to water); the target's elevation is added back."""
    return predict(model, train.assign(head_m=train.head_m - train.ground_elev_m), target) + target.ground_elev_m


def predict(model: str, train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    if model.startswith("depth "):
        return via_depth(model.removeprefix("depth "), train, target)
    if model == "persistence":
        return naive.persistence(train, target)
    if model == "seasonal mean":
        return naive.seasonal_mean(train, target)
    if model == "seasonal trend":
        return naive.seasonal_trend(train, target)
    if model == "kriging":
        return kriging.ordinary_kriging(train, target)
    if model == "trend + kriging":
        return trend_then_krige(train, target)
    if model == "random forest":
        return trees.fit_predict(train, target, "forest")
    if model == "LightGBM":
        return trees.fit_predict(train, target, "lightgbm")
    raise ValueError(model)


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


def main() -> None:
    proc = data_dir() / "processed"
    out_dir = proc / "predictions"
    out_dir.mkdir(exist_ok=True)
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    master = pd.read_parquet(proc / "master.parquet")
    splits = pd.read_parquet(proc / "splits.parquet")
    levels = shallow_levels(master)

    rows = []
    for layout in scfg["layouts"]:
        df = with_layout(levels, splits, layout)
        train = training_rows(df)
        for stage in ("val", "test"):
            preds = []
            for task, models in TASK_MODELS.items():
                tgt = target_rows(df, task, stage).copy()
                for model in models:
                    tgt[model] = predict(model, train, tgt).to_numpy()
                    rows.append({"layout": layout, "stage": stage, "task": task, "model": model,
                                 **summary(tgt, model)})
                    print(f"layout {layout} {stage:4s} task {task} {model:15s} RMSE {rows[-1]['rmse_m']:.2f} m "
                          f"(coverage {100 * rows[-1]['coverage_of_targets']:.0f}%)", flush=True)
                preds.append(tgt.assign(task=task)[["well_id", "date", "task", "head_m", *models]])
            pd.concat(preds).to_parquet(out_dir / f"baselines_L{layout}_{stage}.parquet")

    res = pd.DataFrame(rows)
    tab = ROOT / "reports" / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    res.to_csv(tab / "baselines.csv", index=False)

    def fmt_table(sel: pd.DataFrame) -> pd.DataFrame:
        out = []
        for (task, model), g in sel.groupby(["task", "model"], sort=False):
            l0 = g[g.layout == scfg["layouts"][0]].iloc[0]
            out.append({"Task": TASK_NAME[task], "Model": model,
                        "RMSE, layout 0": f"{l0.rmse_m:.2f} m", "MAE, layout 0": f"{l0.mae_m:.2f} m",
                        "Median well RMSE": f"{l0.median_well_rmse_m:.2f} m",
                        "Median well NSE": "" if np.isnan(l0.median_well_nse) else f"{l0.median_well_nse:.2f}",
                        "RMSE, mean of 3 layouts": f"{g.rmse_m.mean():.2f} m (±{g.rmse_m.std():.2f})",
                        "Readings / wells (layout 0)": f"{l0.n_readings} / {l0.n_wells}",
                        "Coverage": f"{100 * l0.coverage_of_targets:.0f}%"})
        return pd.DataFrame(out)

    test = fmt_table(res[res.stage == "test"])
    val = fmt_table(res[res.stage == "val"])
    best = res[res.stage == "test"].groupby("task").apply(lambda g: g.groupby("model").rmse_m.mean().idxmin())
    md = [
        "# M5 report: baselines",
        "",
        "*Generated by `scripts/run_baselines.py`. Every model learns only from wells in training blocks, years 2005–2018.*",
        "",
        "## The three tasks",
        "",
        "| Task | Predict | For |",
        "|---|---|---|",
        "| A | water level at wells the models learned from | 2020–2024 (validation: 2019) |",
        "| B | water level at wells in held-out 20 km blocks (places the models never saw) | 2005–2018 |",
        "| C | water level at wells in held-out blocks | 2020–2024 (validation: 2019) |",
        "",
        "Test results use the observed rainfall of the test years for every model, so they are *hindcasts*: forecasts made with the weather that actually happened.",
        "",
        "## Test results",
        "",
        *md_table(test),
        "",
        "Best baseline per task (mean RMSE over the 3 layouts): " + "; ".join(f"**{t}: {m}**" for t, m in best.items()) + ".",
        "",
        "*Coverage* = share of target readings the model could predict (kriging needs at least 15 training wells measured in the same survey).",
        "",
        "## Validation results (used only for tuning and the zoning preview)",
        "",
        *md_table(val),
        "",
        "## How to read the numbers",
        "",
        "- **RMSE:** typical error in metres; big misses count extra.",
        "- **MAE:** plain average size of the error in metres.",
        "- **Median well RMSE:** the error of the middle well when wells are sorted by error, so a few bad wells cannot dominate.",
        "- **Median well NSE:** 1 = perfect; 0 = no better than always predicting that well's own average level over the period; below 0 = worse. Wells with at least 4 readings.",
        "- **± after the mean:** the spread over the 3 block layouts. It shows how much the score depends on which areas happened to be held out.",
        "",
        "See `reports/metrics_explained.md` for every measure in plain words.",
    ]
    (ROOT / "reports" / "m5_baselines.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote reports/m5_baselines.md")


if __name__ == "__main__":
    main()
