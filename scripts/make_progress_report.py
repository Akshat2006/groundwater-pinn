"""Two-page progress report (PDF) with a milestone track, project flowchart and key results.

    python scripts/make_progress_report.py     -> reports/progress_report.pdf

Needs the outputs of M0 (reports/tables/g0_*.csv) and M1 (reports/figures/m1_cases.png).
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
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "progress_report.pdf"

# palette (validated reference categorical slots + neutrals)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, LINE, PALE = "#0b0b0b", "#52514e", "#c9c8c3", "#f3f2ef"
DONE_FILL, NEXT_FILL, TODO_FILL = "#d6f2e6", "#fde3d8", "#f3f2ef"

MILESTONES = [("M0", "Setup"), ("M1", "Simulator"), ("M2", "PINN\ntest"), ("M3", "Data\npipeline"),
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
    fill, edge = {"done": (DONE_FILL, AQUA), "next": (NEXT_FILL, ORANGE), "todo": (TODO_FILL, LINE)}[state]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=1.3))
    lines = text.split("\n")
    ax.text(cx, cy + h / 2 - 1.5, lines[0], ha="center", va="top", fontsize=fs + 0.3, fontweight="bold", color=INK)
    if len(lines) > 1:
        ax.text(cx, cy + h / 2 - 4.9, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.5,
                color=MUTED, linespacing=1.25)
    tag = {"done": "✓ done", "next": "▶ next", "todo": ""}[state]
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
    _box(ax, cx[0], y_top, w, h, "M0 Setup\nPython env, GPU,\n46 tests,\ndata check", "done")
    _box(ax, cx[1] + 3, y_top, w, h, "M1 Simulator\nfake aquifer with\nknown S, T, pumping;\nchecks passed", "done")
    gx, dw, dh = cx[3], 11.0, 8.5
    ax.add_patch(Polygon([(gx, y_top + dh), (gx + dw, y_top), (gx, y_top - dh), (gx - dw, y_top)], closed=True,
                         facecolor=NEXT_FILL, edgecolor=ORANGE, linewidth=1.3))
    ax.text(gx, y_top + 1.6, "M2 GATE", ha="center", fontsize=6.2, fontweight="bold", color=INK)
    ax.text(gx, y_top - 2.8, "PINN recovers\nS, T, Q within 10%?", ha="center", fontsize=5.0, color=MUTED,
            linespacing=1.2)
    ax.text(gx + dw + 0.3, y_top + 4.3, "▶ next", fontsize=5.3, color=ORANGE, fontweight="bold")
    fx = cx[5] - 1
    _box(ax, fx, y_top, w, h, "Fix & retry\nadjust training,\nor backup method\n(ask you first)", "todo")
    _arrow(ax, (cx[0] + w / 2, y_top), (cx[1] + 3 - w / 2, y_top))
    _arrow(ax, (cx[1] + 3 + w / 2, y_top), (gx - dw, y_top))
    _arrow(ax, (gx + dw, y_top), (fx - w / 2, y_top))
    ax.text((gx + dw + fx - w / 2) / 2, y_top - 2.4, "no", ha="center", fontsize=5.8, color=MUTED)
    _arrow(ax, (fx, y_top + h / 2), (gx + 1.5, y_top + dh - 0.6), rad=0.35, ls="--")
    # lane 2: real data
    _box(ax, cx[0], y_bot, w, h, "Data in hand\n1,139 wells,\nrain, elevation,\nland use, canals", "done")
    _box(ax, cx[1], y_bot, w, h, "M3 Pipeline\nclean + merge\ninto one\nmaster table", "todo")
    _box(ax, cx[2], y_bot, w, h, "M4–M5 Zoning\ngroup wells;\nearly check:\ndoes it help?", "todo")
    _box(ax, cx[3], y_bot, w, h, "M6–M7 PINNs\nregional vs\nzoned; infer\npumping", "todo")
    _box(ax, cx[4], y_bot, w, h, "M8 Tests\nvs 6+ simpler\nmodels on\nhidden areas", "todo")
    _box(ax, cx[5], y_bot, w, h, "M9–M10 Results\nwhat-if runs,\nmaps, slides,\npaper", "todo")
    for i in range(5):
        _arrow(ax, (cx[i] + w / 2, y_bot), (cx[i + 1] - w / 2, y_bot))
    _arrow(ax, (gx, y_top - dh), (gx, y_bot + h / 2), color=ORANGE)
    ax.text(gx + 0.8, (y_top - dh + y_bot + h / 2) / 2, "yes: only then use real data", fontsize=5.6,
            color=ORANGE, va="center", fontweight="bold")
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
              Paragraph(f"Progress report · {date.today():%d %B %Y} · Punjab–Haryana alluvial belt, 16 districts · "
                        f"<b>2 of 11 milestones complete</b>", sub),
              Spacer(1, 4), img("progress_milestones.png", W), Spacer(1, 2)]
    story += [Paragraph("The project in one paragraph", h2),
              Paragraph(
                  "We forecast groundwater levels everywhere in the study area, including places with no monitoring "
                  "well. Wells are grouped into <b>zones</b> that behave alike. Each zone gets a <b>physics-informed "
                  "neural network (PINN)</b>: it fits the measured water levels while obeying the groundwater flow "
                  "equation <b>S·∂h/∂t = T∇²h + R − Q</b> (storage × change in level = sideways flow + recharge − "
                  "pumping). Unmetered farm <b>pumping Q</b> is inferred as a hidden quantity. Before touching real "
                  "data, the method must prove on a simulated aquifer that it recovers answers we planted.", body)]
    story += [Paragraph("Project flow and where we are", h2), img("progress_flowchart.png", W)]
    story += [Paragraph("Done so far", h2), table([
        ["Step", "What was done", "Key result"],
        ["Plan", "Brief checked; 12 design problems ranked and fixed (PLAN.md)",
         "Pumping made identifiable by tying it to land use and the crop calendar"],
        ["Data", "Study area chosen by a data check; every dataset fetched by script",
         "1,139 wells (1990–2024) + 248 sensor wells; nothing left to download by hand"],
        ["M0 Setup", "Python environment, project layout, automatic tests, Colab notebook, literature check",
         "GPU (GTX 1050) works; 46 tests pass; no prior zoned or hidden-pumping PINN on field data found"],
        ["M1 Simulator", "Finite-difference aquifer simulator and 4 synthetic aquifers (A–D) with known answers",
         "Matches exact textbook solutions; water balance closes to 1 part in 10 billion"],
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
         "Careful scaling, staged training, automatic tests; a backup method if needed (asks first)"],
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

    story += [Paragraph("Next steps and honest status", h2), table([
        ["Next", "What it decides"],
        ["<b>M2</b> PINN on the simulated aquifer", "Pass/fail gate: recover S, T and pumping within ~10% on case C. "
         "If it fails, fix or switch to the backup method (asks first). No real-data modelling before this passes."],
        ["<b>M3</b> data pipeline", "Clean and merge all data into one table; convert depth-to-water into water level."],
        ["<b>M4–M5</b> zoning + early check", "First real signal on whether zoning beats a single regional model."],
    ], [0.27 * W, 0.73 * W]),
        Spacer(1, 3),
        Paragraph("<b>Honest status:</b> no accuracy result on real data exists yet. The 5–10% improvement over a "
                  "single regional PINN is a <i>target, not a result</i>; the first evidence comes from the zoning "
                  "check in M4. Pumping estimates on real data will be inferences with no ground truth and will be "
                  "reported as ranges.", small)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 6.5)
        canvas.setFillColor(muted)
        canvas.drawString(14 * mm, 9 * mm, "Generated from the project repository (scripts/make_progress_report.py)")
        canvas.drawRightString(A4[0] - 14 * mm, 9 * mm, f"Page {doc.page} of 2")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm,
                            bottomMargin=14 * mm, title="Zoned PINNs for Groundwater Forecasting: progress report",
                            author="Groundwater PINN project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
