#!/usr/bin/env python3
"""Rebuild publication figures for notes_zh.tex from frozen DELIVERABLE values.

Sources on DGX:
  paper_v3/DELIVERABLE/01_main_experiment/01_main_tables/publication/{A,B}_MAIN.csv
  paper_v3/DELIVERABLE/03_ablation_extended/01_tables/E3_{LEARNING,TREE}_CURVE_{A,B}.csv

This script only visualizes archived numbers; it does not fit or evaluate models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT_DIR = Path(__file__).resolve().parent

BLUE = "#0F4D92"
GREEN = "#3A7D44"
RED = "#B64342"
ORANGE = "#C77400"
GRAY = "#8A8F98"
DARK = "#272727"
GRID = "#D9DCE1"

plt.rcParams.update(
    {
        # Figure labels are English; the CJK faces stay in the stack so that a
        # reintroduced Chinese label would still render rather than tofu.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "PingFang SC",
            "Heiti SC",
            "Arial Unicode MS",
            "Noto Sans CJK SC",
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


@dataclass(frozen=True)
class Point:
    name: str
    latency: float
    auroc: float


PROTOCOL_A = [
    Point("Ours", 112.54, 0.79330),
    Point("AlignScore", 677.29, 0.78113),
    Point("Qwen30-Fast", 273.35, 0.76893),
    Point("Qwen30-Judge", 642.74, 0.76390),
    Point("FactCG", 429.94, 0.75500),
    Point("Granite-3.1-2B", 177.34, 0.74024),
    Point("WeCheck", 359.32, 0.73337),
    Point("MiniCheck-FT5", 451.01, 0.70117),
    Point("FactKB", 44.46, 0.67432),
    Point("Granite-3.2-8B", 1011.32, 0.67028),
    Point("Granite-3.2-3B", 243.92, 0.66372),
    Point("MiniCheck-DBTA", 473.33, 0.63642),
    Point("HHEM", 149.64, 0.62453),
    Point("Granite-4.1-LoRA", 448.81, 0.61887),
    Point("Lettuce-v2", 73.68, 0.61216),
    Point("FactCC", 44.86, 0.53237),
]

PROTOCOL_B = [
    Point("Qwen30-Fast", 211.65, 0.83384),
    Point("Ours", 106.14, 0.82256),
    Point("Qwen30-Judge", 600.87, 0.81500),
    Point("AlignScore", 561.65, 0.81167),
    Point("Granite-3.1-2B", 131.81, 0.79583),
    Point("FactCG", 427.50, 0.77998),
    Point("WeCheck", 496.01, 0.76912),
    Point("FactKB", 37.01, 0.76282),
    Point("Granite-3.2-3B", 263.87, 0.75501),
    Point("MiniCheck-FT5", 364.03, 0.73907),
    Point("MiniCheck-DBTA", 480.64, 0.67740),
    Point("HHEM", 223.30, 0.66521),
    Point("Granite-3.2-8B", 690.01, 0.63386),
    Point("Lettuce-v2", 61.61, 0.59951),
    Point("FactCC", 37.60, 0.57089),
    Point("Granite-4.1-LoRA", 376.76, 0.47487),
]

POOL = {"FactCC", "Lettuce-v2", "Granite-3.1-2B"}
LABELS = {
    "Ours",
    "AlignScore",
    "Qwen30-Fast",
    "FactKB",
    "FactCC",
    "Lettuce-v2",
    "Granite-3.1-2B",
}

LABEL_OFFSETS = {
    "A": {
        "Ours": (5, 7),
        "AlignScore": (-52, 7),
        "Qwen30-Fast": (5, 6),
        "FactKB": (-2, 8),
        "FactCC": (-2, -13),
        "Lettuce-v2": (5, -13),
        "Granite-3.1-2B": (5, 6),
    },
    "B": {
        "Ours": (-7, -15),
        "AlignScore": (-51, 7),
        "Qwen30-Fast": (5, 7),
        "FactKB": (-2, 9),
        "FactCC": (-2, -14),
        "Lettuce-v2": (5, -12),
        "Granite-3.1-2B": (5, -13),
    },
}


def pareto_front(points: Iterable[Point]) -> list[Point]:
    """Return nondominated points for lower latency and higher AUROC."""
    frontier: list[Point] = []
    best = float("-inf")
    for point in sorted(points, key=lambda item: (item.latency, -item.auroc)):
        if point.auroc > best:
            frontier.append(point)
            best = point.auroc
    return frontier


def draw_quality_cost() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.15), sharey=False)
    for ax, protocol, points in zip(axes, ("A", "B"), (PROTOCOL_A, PROTOCOL_B)):
        frontier = pareto_front(points)
        ax.plot(
            [point.latency for point in frontier],
            [point.auroc for point in frontier],
            color=GREEN,
            linewidth=1.8,
            linestyle="--",
            zorder=1,
        )
        for point in points:
            if point.name == "Ours":
                marker, color, size, edge, zorder = "*", BLUE, 180, DARK, 5
            elif point.name in POOL:
                marker, color, size, edge, zorder = "D", GREEN, 48, "white", 4
            else:
                marker, color, size, edge, zorder = "o", GRAY, 38, "white", 3
            ax.scatter(
                point.latency,
                point.auroc,
                marker=marker,
                s=size,
                color=color,
                edgecolor=edge,
                linewidth=0.8,
                zorder=zorder,
            )
            if point.name in LABELS:
                dx, dy = LABEL_OFFSETS[protocol][point.name]
                ax.annotate(
                    point.name,
                    (point.latency, point.auroc),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.4,
                    fontweight="bold" if point.name == "Ours" else "normal",
                    color=BLUE if point.name == "Ours" else DARK,
                    zorder=6,
                )
        ax.set_xscale("log")
        ax.set_xlabel("end-to-end latency (ms per summary, log scale)")
        ax.set_ylabel("pooled AUROC")
        ax.set_title(f"({chr(96 + (1 if protocol == 'A' else 2))}) Protocol {protocol}", loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
        ax.tick_params(direction="out", length=3)
        y_values = [point.auroc for point in points]
        margin = 0.035
        ax.set_ylim(min(y_values) - margin, max(y_values) + margin)

    legend = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=BLUE, markeredgecolor=DARK, markersize=11, label="Ours"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GREEN, markeredgecolor="white", markersize=7, label="pool member"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", markersize=7, label="other fixed verifier"),
        Line2D([0], [0], color=GREEN, linestyle="--", linewidth=1.8, label="Pareto front"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.22)
    fig.savefig(OUT_DIR / "quality_cost.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / "quality_cost.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


DATA_FRACTIONS = [5, 10, 20, 35, 50, 75, 100]
DATA_CURVES = {
    "A": {
        "loss": [0.017172, 0.016361, 0.015091, 0.015046, 0.014766, 0.014588, 0.014546],
        "auroc": [0.760850, 0.775170, 0.784046, 0.788744, 0.790283, 0.792410, 0.793300],
    },
    "B": {
        "loss": [0.017293, 0.016430, 0.014115, 0.013695, 0.013849, 0.013117, 0.013290],
        "auroc": [0.790242, 0.790091, 0.806848, 0.808914, 0.813204, 0.817730, 0.822556],
    },
}
TREE_CURVES = {
    "A": {
        "x": [1, 5, 25, 50, 100, 800],
        "loss": [0.015378, 0.014677, 0.014508, 0.014487, 0.014478, 0.014478],
        "auroc": [0.767155, 0.775698, 0.778058, 0.773468, 0.775766, 0.775253],
    },
    "B": {
        "x": [1, 5, 25, 50, 100, 200],
        "loss": [0.014154, 0.013502, 0.013341, 0.013307, 0.013302, 0.013290],
        "auroc": [0.806669, 0.816332, 0.822401, 0.825920, 0.825952, 0.826084],
    },
}


def add_dual_curve(ax, x, loss, auroc, title, log_x=False):
    right = ax.twinx()
    ax.plot(x, loss, color=BLUE, marker="o", markersize=4.5, linewidth=2.0, label="head loss")
    right.plot(x, auroc, color=RED, marker="s", markersize=4.2, linewidth=2.0, label="routed AUROC")
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(value) for value in x])
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("validation head loss", color=BLUE)
    right.set_ylabel("routed AUROC", color=RED)
    ax.tick_params(axis="y", colors=BLUE, direction="out", length=3)
    right.tick_params(axis="y", colors=RED, direction="out", length=3)
    ax.tick_params(axis="x", direction="out", length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(RED)
    right.spines["top"].set_visible(False)
    return right


def draw_convergence() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6))
    right_a_data = add_dual_curve(
        axes[0, 0],
        DATA_FRACTIONS,
        DATA_CURVES["A"]["loss"],
        DATA_CURVES["A"]["auroc"],
        "(a) Protocol A: training size",
    )
    right_b_data = add_dual_curve(
        axes[0, 1],
        DATA_FRACTIONS,
        DATA_CURVES["B"]["loss"],
        DATA_CURVES["B"]["auroc"],
        "(b) Protocol B: training size",
    )
    right_a_trees = add_dual_curve(
        axes[1, 0],
        TREE_CURVES["A"]["x"],
        TREE_CURVES["A"]["loss"],
        TREE_CURVES["A"]["auroc"],
        "(c) Protocol A: forest size",
        log_x=True,
    )
    right_b_trees = add_dual_curve(
        axes[1, 1],
        TREE_CURVES["B"]["x"],
        TREE_CURVES["B"]["loss"],
        TREE_CURVES["B"]["auroc"],
        "(d) Protocol B: forest size",
        log_x=True,
    )
    axes[0, 0].set_xlabel("document groups used in the fit partition (%)")
    axes[0, 1].set_xlabel("document groups used in the fit partition (%)")
    axes[1, 0].set_xlabel("number of trees (log scale)")
    axes[1, 1].set_xlabel("number of trees (log scale)")
    # Keep metric labels on the outside edges of the grid. Central labels
    # collide after the vector figure is scaled to the manuscript width.
    right_a_data.set_ylabel("")
    right_a_trees.set_ylabel("")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    legend = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=2.0, label="validation head loss"),
        Line2D([0], [0], color=RED, marker="s", linewidth=2.0, label="routed AUROC"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.09, top=0.90, hspace=0.44, wspace=0.30)
    fig.savefig(OUT_DIR / "convergence.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / "convergence.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)



# ---------------------------------------------------------------- few-shot adaptation
# Source: paper_v3/runs/fewshot_frac_v1/FEWSHOT_FRACTION_CURVE.csv
# Fractions of each corpus's own training pool; ten seeds per point.
FEWSHOT_FRACS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.00]

FEWSHOT = {
    "CoGenSumm": {
        "auroc": [0.6516, 0.6123, 0.6453, 0.6643, 0.6837, 0.6876, 0.6477, 0.6674, 0.6864, 0.6798],
        "sd": [0.0600, 0.0227, 0.0345, 0.0334, 0.0257, 0.0239, 0.0470, 0.0548, 0.0539, 0.0252],
        "pool": 535, "colour": BLUE, "marker": "o", "style": "-",
    },
    "FRANK": {
        "auroc": [0.8333, 0.8358, 0.7907, 0.8392, 0.8194, 0.8311, 0.8285, 0.8247, 0.8268, 0.8252],
        "sd": [0.0089, 0.0057, 0.0945, 0.0031, 0.0250, 0.0152, 0.0144, 0.0143, 0.0103, 0.0082],
        "pool": 669, "colour": RED, "marker": "s", "style": "--",
    },
    "RAGTruth": {
        "auroc": [0.5430, 0.5979, 0.6131, 0.6334, 0.6396, 0.6381, 0.6358, 0.6378, 0.6377, 0.6380],
        "sd": [0.0429, 0.0266, 0.0120, 0.0210, 0.0153, 0.0090, 0.0035, 0.0013, 0.0016, 0.0015],
        "pool": 2983, "colour": GREEN, "marker": "^", "style": "-.",
    },
    "UniSumEval": {
        "auroc": [0.5369, 0.5427, 0.5655, 0.5571, 0.5660, 0.5606, 0.5689, 0.5832, 0.5823, 0.5923],
        "sd": [0.0261, 0.0313, 0.0360, 0.0156, 0.0138, 0.0171, 0.0195, 0.0244, 0.0245, 0.0144],
        "pool": 1089, "colour": ORANGE, "marker": "D", "style": ":",
    },
}


def draw_fewshot() -> None:
    x = [f * 100 for f in FEWSHOT_FRACS]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.15))

    for ax, mode in zip(axes, ("absolute", "delta")):
        for name, d in FEWSHOT.items():
            base = d["auroc"][0]
            y = d["auroc"] if mode == "absolute" else [v - base for v in d["auroc"]]
            sd = d["sd"]
            ax.fill_between(x, [a - s for a, s in zip(y, sd)], [a + s for a, s in zip(y, sd)],
                            color=d["colour"], alpha=0.10, linewidth=0)
            ax.plot(x, y, color=d["colour"], marker=d["marker"], linestyle=d["style"],
                    linewidth=1.8, markersize=5, markeredgewidth=0)

        ax.set_xlabel("in-corpus training rows added (% of that corpus's own pool)")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)

        if mode == "absolute":
            ax.axhline(0.5, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.0)
            ax.text(99, 0.508, "chance", color=GRAY, fontsize=8, va="bottom", ha="right")
            # Two of ten seeds pick beta = 0.2 instead of 0.0 at this point, which discounts
            # Granite-3.1-2B out of contention and collapses FRANK to 0.60. The mean is dragged
            # down and the band widened by that bistability, not by ordinary seed noise.
            ax.annotate("2/10 seeds flip to $\\beta$=0.2\nand drop the only\nverifier that works here",
                        xy=(10, 0.791), xytext=(38, 0.757),
                        fontsize=7.5, color=DARK, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.8,
                                        connectionstyle="arc3,rad=-0.2"))
            ax.set_ylabel("AUROC")
            ax.set_ylim(0.48, 0.90)
            ax.set_title("(a) absolute", fontsize=10, color=DARK, pad=6)
        else:
            ax.axhline(0.0, color=GRAY, linewidth=1.0)
            ax.set_ylabel("AUROC change from 0%")
            ax.set_ylim(-0.075, 0.125)
            ax.set_title("(b) change from leave-one-dataset-out", fontsize=10, color=DARK, pad=6)

    legend = [
        Line2D([0], [0], color=d["colour"], marker=d["marker"], linestyle=d["style"],
               linewidth=1.8, markersize=5,
               label=f"{name} (pool {d['pool']:,})")
        for name, d in FEWSHOT.items()
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.24)
    fig.savefig(OUT_DIR / "fewshot.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / "fewshot.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

def main() -> None:
    draw_quality_cost()
    draw_convergence()
    draw_fewshot()
    for filename in ("quality_cost.pdf", "quality_cost.png", "convergence.pdf", "convergence.png",
                     "fewshot.pdf", "fewshot.png"):
        path = OUT_DIR / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing figure output: {path}")


if __name__ == "__main__":
    main()
