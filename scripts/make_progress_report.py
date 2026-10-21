"""Progress report (PDF) with a milestone track, project flowchart and key results (M0–M2).

    python scripts/make_progress_report.py     -> reports/progress_report.pdf

Needs the outputs of M0 (reports/tables/g0_*.csv), M1 (reports/figures/m1_cases.png) and M2
(reports/figures/m2_gate_*.png from scripts/check_m2.py).
"""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "progress_report.pdf"

# palette (validated reference categorical slots + neutrals)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, LINE, PALE = "#0b0b0b", "#52514e", "#c9c8c3", "#f3f2ef"
DONE_FILL, NEXT_FILL, TODO_FILL = "#d6f2e6", "#fde3d8", "#f3f2ef"

MILESTONES = [("M0", "Setup"), ("M1", "Simulator"), ("M2", "PINN test\n(gate ✓)"), ("M3", "Data\npipeline"),
              ("M4", "Zoning"), ("M5", "Base-\nlines"), ("M6", "Regional\nPINN"), ("M7", "Zoned\nPINN"),
              ("M8", "Experi-\nments"), ("M9", "Scenarios"), ("M10", "Write-up")]
N_DONE = 2


# ------------------------------------------------------------------ diagrams
def milestone_strip(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.0), dpi=300)
    n = len(MILESTONES)
    xs = [i / (n - 1) for i in range(n)]
    ax.plot([0, 1], [0.6, 0.6], color=LINE, lw=3, solid_capstyle="round", zorder=1)
    ax.plot([0, xs[N_DONE]], [0.6, 0.6], color=AQUA, lw=3, solid_capstyle="round", zorder=2)
    for i, (x, (code, label)) in enumerate(zip(xs, MILESTONES)):
        if i < N_DONE:
            ax.scatter(x, 0.6, s=260, color=AQUA, zorder=3, edgecolor="white", linewidth=1.5)
            ax.text(x, 0.6, "✓", ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
        elif i == N_DONE:
            ax.scatter(x, 0.6, s=260, color="white", zorder=3, edgecolor=ORANGE, linewidth=2.2)
            ax.text(x, 0.6, "▶", ha="center", va="center", color=ORANGE, fontsize=6.5, zorder=4)
        else:
            ax.scatter(x, 0.6, s=150, color="white", zorder=3, edgecolor=LINE, linewidth=1.8)
        weight = "bold" if i <= N_DONE else "normal"
        col = INK if i <= N_DONE else MUTED
        ax.text(x, 1.12, code, ha="center", va="bottom", fontsize=7, color=col, fontweight=weight)
        ax.text(x, 0.08, label, ha="center", va="top", fontsize=5.8, color=col, linespacing=1.1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.55, 1.5)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _box(ax, cx, cy, w, h, text, state, fs=5.6):
    fill, edge = {"done": (DONE_FILL, AQUA), "next": (NEXT_FILL, ORANGE), "running": (NEXT_FILL, ORANGE),
                  "todo": (TODO_FILL, LINE)}[state]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=1.3))
    lines = text.split("\n")
    ax.text(cx, cy + h / 2 - 1.5, lines[0], ha="center", va="top", fontsize=fs + 0.3, fontweight="bold", color=INK)
    if len(lines) > 1:
        ax.text(cx, cy + h / 2 - 4.9, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.5,
                color=MUTED, linespacing=1.25)
    tag = {"done": "✓ done", "next": "▶ next", "running": "▶ running", "todo": ""}[state]
    if tag:
        ax.text(cx + w / 2 - 1.0, cy - h / 2 + 1.0, tag, ha="right", va="bottom", fontsize=5.3,
                color=edge, fontweight="bold")


def _arrow(ax, p0, p1, color=MUTED, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7, color=color, lw=1.0,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")
    # lanes
    for y0, label in ((30.5, "1  Prove the method on a simulated aquifer"),
                      (1.5, "2  Apply it to real Punjab–Haryana data")):
        ax.add_patch(FancyBboxPatch((0.3, y0), 99.4, 22.5, boxstyle="round,pad=0,rounding_size=1.5",
                                    facecolor="white", edgecolor="#e4e3df", linewidth=0.8))
        ax.text(1.5, y0 + 21.3, label, fontsize=6.6, color=MUTED, va="top", fontweight="bold")

    w, h = 14.6, 14.5
    cx = [9.0 + i * 16.4 for i in range(6)]
    y_top, y_bot = 39.5, 10.5
    # lane 1: synthetic track
    _box(ax, cx[0], y_top, w, h, "M0 Setup\nPython env, GPU,\n58 tests,\ndata check", "done")
    _box(ax, cx[1] + 3, y_top, w, h, "M1 Simulator\nfake aquifer with\nknown S, T, pumping;\nchecks passed", "done")
    gx, dw, dh = cx[3], 11.0, 8.5
    ax.add_patch(Polygon([(gx, y_top + dh), (gx + dw, y_top), (gx, y_top - dh), (gx - dw, y_top)], closed=True,
                         facecolor=DONE_FILL, edgecolor=AQUA, linewidth=1.3))
    ax.text(gx, y_top + 1.6, "M2 GATE", ha="center", fontsize=6.2, fontweight="bold", color=INK)
    ax.text(gx, y_top - 2.8, "PINN recovers\nS, T, Q within 10%?", ha="center", fontsize=5.0, color=MUTED,
            linespacing=1.2)
    ax.text(gx + dw + 0.3, y_top + 3.3, "✓ passed\n(attempt 9)", fontsize=5.3, color=AQUA, fontweight="bold",
            va="bottom", linespacing=1.15)
    fx = cx[5] - 1
    _box(ax, fx, y_top, w, h, "M2 stress tests\nrealistic wells,\nnoise, wrong\nrecharge (Colab)", "running")
    _arrow(ax, (cx[0] + w / 2, y_top), (cx[1] + 3 - w / 2, y_top))
    _arrow(ax, (cx[1] + 3 + w / 2, y_top), (gx - dw, y_top))
    _arrow(ax, (gx + dw, y_top), (fx - w / 2, y_top))
    ax.text((gx + dw + fx - w / 2) / 2, y_top - 2.4, "then", ha="center", fontsize=5.8, color=MUTED)
    # lane 2: real data
    _box(ax, cx[0], y_bot, w, h, "Data in hand\n1,139 wells,\nrain, elevation,\nland use, canals", "done")
    _box(ax, cx[1], y_bot, w, h, "M3 Pipeline\nclean + merge\ninto one\nmaster table", "next")
    _box(ax, cx[2], y_bot, w, h, "M4–M5 Zoning\ngroup wells;\nearly check:\ndoes it help?", "todo")
    _box(ax, cx[3], y_bot, w, h, "M6–M7 PINNs\nregional vs\nzoned; infer\npumping", "todo")
    _box(ax, cx[4], y_bot, w, h, "M8 Tests\nvs 6+ simpler\nmodels on\nhidden areas", "todo")
    _box(ax, cx[5], y_bot, w, h, "M9–M10 Results\nwhat-if runs,\nmaps, slides,\npaper", "todo")
    for i in range(5):
        _arrow(ax, (cx[i] + w / 2, y_bot), (cx[i + 1] - w / 2, y_bot))
    _arrow(ax, (gx, y_top - dh), (gx, y_bot + h / 2), color=AQUA)
    ax.text(gx + 0.8, (y_top - dh + y_bot + h / 2) / 2, "yes: real data now allowed", fontsize=5.6,
            color=AQUA, va="center", fontweight="bold")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def wells_per_year(path: Path) -> None:
    py = pd.read_csv(ROOT / "reports" / "tables" / "g0_per_year.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.15), dpi=300)
    spans = [(2005, 2018, "train"), (2019, 2019, "val"), (2020, 2024, "test")]
    for a, b, label in spans:
        ax.axvspan(a - 0.5, b + 0.5, color=PALE, zorder=0)
        ax.text((a + b) / 2, 845, label, ha="center", va="bottom", fontsize=5.8, color=MUTED)
    ax.bar(py.year, py.wells, width=0.7, color=BLUE, zorder=2)
    ax.annotate("2012: deep tube\nwells added", xy=(2012, 300), xytext=(1995.5, 560), fontsize=5.6, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=6))
    ax.set_ylim(0, 930)
    ax.set_xlim(1989.3, 2024.7)
    ax.set_title("CGWB wells read per year (16 districts)", fontsize=7, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(labelsize=5.8, colors=MUTED, width=0.6)
    ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ PDF
def register_fonts() -> str:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(ttf / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(ttf / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
    return "DV"


def build() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    milestone_strip(FIG / "progress_milestones.png")
    flowchart(FIG / "progress_flowchart.png")
    wells_per_year(FIG / "progress_wells_per_year.png")

    font = register_fonts()
    ink, muted = colors.HexColor(INK), colors.HexColor(MUTED)
    body = ParagraphStyle("body", fontName=font, fontSize=8.3, leading=11.0, textColor=ink, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=9.4)
    caption = ParagraphStyle("cap", parent=body, fontSize=6.9, leading=8.6, textColor=muted)
    h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=15, leading=18)
    sub = ParagraphStyle("sub", parent=body, fontSize=8.5, textColor=muted)
    h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=9.6, leading=12, spaceBefore=5, spaceAfter=2.5,
                        textColor=colors.HexColor(BLUE))
    cell = ParagraphStyle("cell", parent=small, fontSize=7.1, leading=8.9)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")

    W = A4[0] - 28 * mm

    def img(name, width):
        from PIL import Image as PILImage
        with PILImage.open(FIG / name) as im:
            w, h = im.size
        return Image(str(FIG / name), width=width, height=width * h / w)

    def table(rows, widths, header=True):
        data = [[Paragraph(str(c), cellb if (header and r == 0) else cell) for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=widths)
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e4e3df")),
                 ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                 ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PALE)),
                      ("LINEBELOW", (0, 0), (-1, 0), 0.8, muted)]
        t.setStyle(TableStyle(style))
        return t

    story = []
    # ---------------- page 1
    story += [Paragraph("Zoned PINNs for Groundwater Forecasting", h1),
              Paragraph(f"Progress report · {date.today():%d %B %Y} · Punjab–Haryana belt, 16 districts · "
                        f"<b>M0–M1 done · M2 gate passed</b>", sub),
              Spacer(1, 4), img("progress_milestones.png", W), Spacer(1, 2)]
    story += [Paragraph("The project in one paragraph", h2),
              Paragraph(
                  "We forecast groundwater levels everywhere in the study area, including places with no monitoring "
                  "well. Wells are grouped into <b>zones</b> that behave alike. Each zone gets a <b>physics-informed "
                  "neural network (PINN)</b>: it fits the measured water levels while obeying the groundwater flow "
                  "equation <b>S·∂h/∂t = T∇²h + R − Q</b> (storage × change in level = sideways flow + recharge − "
                  "pumping). Unmetered farm <b>pumping Q</b> is inferred as a hidden quantity. Before touching real "
                  "data, the method must prove on a simulated aquifer that it recovers answers we planted.", body)]
    story += [Paragraph("Project flow and where we are", h2), img("progress_flowchart.png", 0.9 * W)]
    story += [Paragraph("Done so far", h2), table([
        ["Step", "What was done", "Key result"],
        ["Plan", "Brief checked; 12 design problems ranked and fixed (PLAN.md)",
         "Pumping made identifiable by tying it to land use and the crop calendar"],
        ["Data", "Study area chosen by a data check; every dataset fetched by script",
         "1,139 wells (1990–2024) + 248 sensor wells; nothing left to download by hand"],
        ["M0 Setup", "Python environment, project layout, automatic tests, Colab notebook, literature check",
         "GPU (GTX 1050) works; 58 tests pass; no prior zoned or hidden-pumping PINN on field data found"],
        ["M1 Simulator", "Finite-difference aquifer simulator and 4 synthetic aquifers (A–D) with known answers",
         "Matches exact textbook solutions; water balance closes to 1 part in 10 billion"],
        ["M2 PINN test", "The real PINN trained on a simulated aquifer from deliberately wrong starting values; "
         "9 attempts, each failure diagnosed and fixed (page 3)",
         "<b>Gate passed:</b> storage −5%, flow ease +2%, total pumping −1%; stress tests running on Colab"],
    ], [0.13 * W, 0.45 * W, 0.42 * W])]
    story += [Paragraph("Biggest risks found in the plan review, and how they are handled", h2), table([
        ["Risk", "Why it matters", "How the plan handles it"],
        ["Pumping can't be pinned down", "Many mixes of storage, recharge and pumping fit the same water levels",
         "Pumping tied to land use and crop season; tested on simulated data first; reported as ranges"],
        ["Physics may add little", "Wells are 5–10 km apart, but water moves only ~1 km sideways per season",
         "A simple 'bucket' model and a no-sideways-flow version are tested, so we can say so honestly"],
        ["Result could be luck", "A small test set makes a 5% difference look like chance",
         "859 test wells; 3 different held-out area layouts; confidence ranges; main result fixed in advance"],
        ["PINN training fails", "PINNs often fail to train on messy, sparse data",
         "<b>Happened in M2 attempt 1</b> (physics under-weighted); diagnosed and fixed, see page 3"],
    ], [0.19 * W, 0.37 * W, 0.44 * W])]
    story.append(PageBreak())

    # ---------------- page 2
    data_tab = table([
        ["Dataset", "Coverage"],
        ["CGWB water levels (India-WRIS)", "1,139 wells, 1990–2024, with well type and depth"],
        ["Sensor wells (6-hourly)", "248 wells, mostly 2022–2025"],
        ["IMD rainfall (0.25°, daily)", "All 35 years, 1990–2024"],
        ["Elevation (Copernicus 30 m)", "Whole belt"],
        ["Land use (Bhuvan, ESA)", "2005-06, 2011-12, 2015-16, 2021"],
        ["Canals, districts, crop areas", "OSM canals; crop areas 1997–2025"],
        ["CGWB reports (PDF)", "Aquifer maps for all 16 districts; assessments 2020–24"],
    ], [0.27 * W, 0.30 * W])
    story += [Paragraph("Data in hand", h2),
              Table([[data_tab, img("progress_wells_per_year.png", 0.41 * W)]], colWidths=[0.58 * W, 0.42 * W],
                    style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0)]),
              Spacer(1, 3),
              Paragraph("<b>Usable for the main analysis:</b> 1,031 wells (shallow, unconfined aquifer) · "
                        "<b>193</b> with long enough records to define zones · <b>859</b> wells in the 2020–24 test "
                        "years (663 with earlier history, 196 brand new: a fair test of predicting where the model "
                        "has never seen data).", small),
              Paragraph("<b>Watch-outs:</b> the network shifted from shallow dug wells to deeper tube wells in 2012 "
                        "(check that zones are not just well types); Karnal is thin in WRIS (filled from the India "
                        "Data Portal); land use exists as snapshots, not every year.", small)]

    checks = table([
        ["Check against an exact answer", "Required", "Result"],
        ["Grid made 2× finer → error shrinks", "≈ 4×", "3.9× ✓"],
        ["Water conserved every simulated day", "< 1 in 10⁸", "1 in 10¹⁰ ✓"],
        ["Closed box / straight-line cases", "exact", "exact ✓"],
        ["Single pumping well (Theis)", "close", "within 0.4% ✓"],
    ], [0.26 * W, 0.11 * W, 0.12 * W])
    cases_txt = Paragraph(
        "<b>Four test aquifers</b> (60 × 60 km, 10 years):<br/>"
        "<b>A</b> uniform, falls ~0.9 m/yr: main test<br/>"
        "<b>B</b> two zones with different S, T: can zoning find them?<br/>"
        "<b>C</b> canal + 3 heavy-pumping spots (4–7 m dips): <b>the M2 pass/fail test</b><br/>"
        "<b>D</b> deliberately unfair: hidden recharge, patchy pumping, blurry land-use map: "
        "measures how wrong the model gets when its assumptions are wrong", cell)
    story += [Paragraph("M1 result: the simulated aquifer is verified", h2),
              Table([[checks, cases_txt]], colWidths=[0.50 * W, 0.50 * W],
                    style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4)]),
              Spacer(1, 3),
              KeepTogether([img("m1_cases.png", W),
                            Paragraph("Top: 10-year water-level trend in each test aquifer (orange = falling; dots = the "
                                      "sparse 'wells' the PINN will see). Bottom: the true pumping the PINN must "
                                      "recover without being shown it (aqua outline = canal).", caption)])]

    story.append(PageBreak())

    # ---------------- page 3: M2
    m2 = m2_attempts()

    def pct(v):
        return f"{100 * v:+.0f}%"

    def ok(v):
        return " ✓" if abs(v) <= 0.10 else ""

    rows = [["Attempt", "What changed", "Storage S", "Flow ease T", "Pumping per season", "Total pumping"]]
    for label, change, r in m2:
        rows.append([label, change, pct(r["S_rel_err"]) + ok(r["S_rel_err"]), pct(r["T_rel_err"]) + ok(r["T_rel_err"]),
                     f"{100 * r['Q_net_season_mean_abs_err']:.0f}%" + ok(r["Q_net_season_mean_abs_err"]),
                     pct(r["Q_net_total_rel_err"]) + ok(r["Q_net_total_rel_err"])])
    story += [Paragraph("M2 result: the PINN recovers a hidden aquifer (gate passed)", h2),
              Paragraph(
                  "The real PINN was trained on simulated case C (60 wells read monthly). It started from wrong guesses: "
                  "storage 0.20 instead of 0.12, ease of flow 3,000 instead of 1,500 m²/day. It got no hints. "
                  "To pass, storage, ease of flow and pumping per season all had to come out within 10% of the "
                  "hidden truth. ✓ marks a value within 10%.", small),
              Spacer(1, 2),
              table(rows, [0.10 * W, 0.36 * W, 0.12 * W, 0.12 * W, 0.16 * W, 0.14 * W]),
              Spacer(1, 3),
              img("m2_gate_pumping.png", W),
              Paragraph("Attempt 9: pumping per crop season over 10 years, PINN vs truth (left), and the 10-year "
                        "average pumping map (right). The PINN was never shown either.", caption),
              Spacer(1, 2),
              img("m2_gate_trajectory.png", 0.86 * W),
              Paragraph("How storage S and ease of flow T moved during training of attempt 9 "
                        "(green band = within 10% of the truth).", caption)]
    story += [Paragraph("What went wrong on the way, and the fixes (all kept as project defaults)", h2), table([
        ["Problem found", "Fix"],
        ["Automatic loss balancing let the data swamp the physics", "Fixed weight between data and physics"],
        ["Smooth seasonal pumping made storage come out too low", "Month-of-year input for pumping (the crop calendar)"],
        ["Blurred recharge vs sharp pumping let the model trade one for the other",
         "Pumping blurred over the same 31 days inside the physics check"],
        ["Storage jumped too low when it started learning, then crawled back",
         "Gentle start for S and T, twice the training, final L-BFGS polish"],
    ], [0.52 * W, 0.48 * W])]
    story += [Paragraph("Caveats, next steps and honest status", h2), table([
        ["", ""],
        ["<b>Caveats</b>", "Shoulder months (April, May, October) are overestimated by about 22% (paddy and wheat "
         "seasons are within 4%). One training run only; a repeat with a different random start is running. This "
         "was the friendly case: realistic sparse, noisy data will do worse, and will be reported, not gated."],
        ["<b>Now (Colab)</b>", "M2 stress tests: 30 wells read 4×/year, noise, recharge 20% wrong, the unfair case D, "
         "and the brief's original pumping design."],
        ["<b>Next</b>", "M3 data pipeline (real data) → M4–M5 zoning and the early 'does zoning help?' check → "
         "M6 first PINN on real data."],
        ["<b>Honest status</b>", "No accuracy result on real data exists yet. The 5–10% improvement over a single "
         "regional PINN is a <i>target, not a result</i>; first evidence comes from the M4 check. Pumping on real "
         "data will be an inference without ground truth, reported as ranges."],
    ], [0.16 * W, 0.84 * W], header=False)]

    def make_footer(total):
        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont(font, 6.5)
            canvas.setFillColor(muted)
            canvas.drawString(14 * mm, 9 * mm, "Generated from the project repository (scripts/make_progress_report.py)")
            canvas.drawRightString(A4[0] - 14 * mm, 9 * mm, f"Page {doc.page} of {total}" if total else f"Page {doc.page}")
            canvas.restoreState()
        return footer

    def build_pdf(total):
        doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm,
                                bottomMargin=14 * mm, title="Zoned PINNs for Groundwater Forecasting: progress report",
                                author="Groundwater PINN project")
        f = make_footer(total)
        doc.build(list(story), onFirstPage=f, onLaterPages=f)
        return doc.page

    build_pdf(build_pdf(None))      # second pass knows the page count
    print("wrote", OUT)


def m2_attempts() -> list[tuple[str, str, dict]]:
    """Key M2 gate attempts with their metrics, read from reports/runs."""
    import json

    from src.utils.config import runs_dir
    wanted = [("", "1", "Automatic balancing of data vs physics"),
              ("a5_month_fastparams", "5", "Fixed balance; month-of-year pumping input"),
              ("a6_matched_smoothing", "6", "+ pumping blurred like recharge in the physics check"),
              ("a7_warmup_40k", "7", "+ gentle start for S, T; twice the training"),
              ("a9_a7_plus_lbfgs", "9", "+ final L-BFGS polish (passed)")]
    found = {}
    for d in sorted(runs_dir().glob("*_m2_run1*")):
        m = d / "metrics.json"
        if m.exists():
            r = json.loads(m.read_text())
            found.setdefault(r.get("tag", ""), r)
    return [(label, change, found[tag]) for tag, label, change in wanted if tag in found]


if __name__ == "__main__":
    build()
