#!/usr/bin/env python3
"""Every figure of `docs/PERFORMANCE.md`, the development notes and the paper, from one script.

    PYTHONPATH=$PWD/python .venv/bin/python benches/figures.py [--only F1 F4 ...]

Writes `docs/figures/F<n>_<name>.{svg,png}`. Diagrams (F1 to F3) are drawn with plotly
shapes and annotations; the measured figures read `benches/results/*.json` and skip
themselves, with a note, when their input is not there yet.

House style: plotly, `plotly_white`, the default colorway, no marker outlines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benches" / "results"
OUT = ROOT / "docs" / "figures"
pio.templates.default = "plotly_white"
COLORS = pio.templates["plotly"].layout.colorway  # the default colorway
# Plain words for the figures: readers need the step, not the function name.
STEP_LABELS = {
    "pp.calculate_qc_metrics": "Quality-control metrics",
    "pp.filter_cells": "Cell filter",
    "pp.filter_genes": "Gene filter",
    "pp.normalize_total": "Normalisation",
    "pp.log1p": "Log transform",
    "pp.highly_variable_genes": "Variable genes",
    "subset": "Subset to variable genes",
    "pp.scale": "Scaling",
    "pp.pca": "PCA",
    "pp.neighbors": "Neighbour graph",
    "tl.umap": "UMAP",
    "tl.leiden": "Leiden",
    "tl.rank_genes_groups": "Marker test",
    "pp.preprocess_backed": "Out-of-core head",
    "whole pipeline": "Whole pipeline",
}
CONFIG_LABELS = {
    "all": "All features",
    "no_metal": "GPU disabled",
    "no_accelerate": "Accelerate disabled",
    "p_cores_only": "Performance cores only",
    "one_core": "One core",
    "no_zero_copy": "Zero-copy disabled",
    "sequential_umap": "Sequential UMAP",
}


def step_label(step: str) -> str:
    return STEP_LABELS.get(step, step)


FONT = dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=13)


PAPER = False  # `--paper`: no in-figure titles (the caption carries them), into docs/figures/paper/


def save(fig: go.Figure, name: str, width: int, height: int) -> None:
    out = OUT / "paper" if PAPER else OUT
    out.mkdir(parents=True, exist_ok=True)
    if PAPER:
        fig.update_layout(title=None, margin=dict(l=20, r=20, t=20, b=20))
    else:
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig.update_layout(font=FONT)
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="#222", ticks="outside"
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="#222", ticks="outside"
    )
    for ext in ("svg", "png"):
        fig.write_image(
            out / f"{name}.{ext}", width=width, height=height, scale=1 if ext == "svg" else 3
        )
    print(f"wrote {out.relative_to(ROOT)}/{name}.svg/.png")


# ----------------------------------------------------------------------------- helpers


def box(fig, x0, y0, x1, y1, text, color, text_color="#222", size=13, line=None, opacity=0.22):
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fillcolor=color,
        opacity=opacity,
        line=dict(color=line or color, width=1.5),
        layer="below",
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2,
        text=text,
        showarrow=False,
        font=dict(size=size, color=text_color),
        align="center",
    )


def arrow(fig, x0, y0, x1, y1, text="", color="#555", width=2, at=0.5):
    """An arrow from (x0, y0) to (x1, y1); `text` sits at fraction `at` along it."""
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=width,
        arrowcolor=color,
        text="",
    )
    if text:
        fig.add_annotation(
            x=x0 + at * (x1 - x0),
            y=y0 + at * (y1 - y0),
            text=text,
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor="rgba(255,255,255,0.9)",
        )


def blank_axes(fig, w, h):
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[0, h], scaleanchor="x", scaleratio=1)


# ----------------------------------------------------------------------------- F1


def f1_chip_and_library():
    """Architecture: (a) software layers over the hardware units, (b) execution unit per stage."""
    fig = go.Figure()
    blank_axes(fig, 100, 92)
    c = COLORS
    grey = "#444"

    def label(x, y, text, size=12, anchor="left", color="#222", bold=False):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{text}</b>" if bold else text,
            showarrow=False,
            xanchor=anchor,
            font=dict(size=size, color=color),
        )

    # ---- (a) software layers over the hardware units
    label(1, 90.5, "(a) Software layers and hardware units", 13, bold=True)
    label(1, 84, "Application layer", 11, color=grey)
    box(fig, 22, 81, 98, 87, "Python API: AnnData input and output", c[0], size=12)
    label(1, 75.5, "Binding layer", 11, color=grey)
    box(fig, 22, 72.5, 98, 78.5, "Zero-copy access to NumPy and SciPy buffers", c[0], size=12)
    label(1, 67, "Compute layer (Rust)", 11, color=grey)
    box(fig, 22, 64, 46, 70, "Multithreaded CPU kernels", c[1], size=12)
    box(fig, 48, 64, 72, 70, "Dense linear algebra<br>(Accelerate BLAS)", c[1], size=12)
    box(fig, 74, 64, 98, 70, "GPU kernels (Metal)", c[1], size=12)
    label(1, 55.5, "Hardware layer<br>(Apple M3 Pro SoC)", 11, color=grey)
    box(fig, 22, 52.5, 46, 58.5, "CPU: 5 performance cores,<br>6 efficiency cores", c[2], size=12)
    box(fig, 48, 52.5, 72, 58.5, "AMX matrix coprocessor", c[3], size=12)
    box(fig, 74, 52.5, 98, 58.5, "GPU: 14 cores", c[5], size=12)
    box(
        fig,
        22,
        44,
        78,
        50,
        "Unified memory: 18 GB, 150 GB/s, shared by CPU, AMX and GPU",
        c[6],
        size=12,
    )
    box(fig, 86, 44, 98, 50, "SSD", c[7], size=12)
    arrow(fig, 60, 81, 60, 78.5, color=grey)
    arrow(fig, 60, 72.5, 60, 70, color=grey)
    for x in (34, 60, 86):
        arrow(fig, x, 64, x, 58.5, color=grey)
    arrow(fig, 34, 52.5, 34, 50, color=grey)
    arrow(fig, 60, 52.5, 60, 50, color=grey)
    arrow(fig, 86, 52.5, 86, 50, color=grey)
    arrow(fig, 86, 47, 78, 47, color=grey)

    # ---- (b) execution unit per pipeline stage
    label(1, 38.5, "(b) Execution unit of each pipeline stage", 13, bold=True)
    cols = ["Pipeline stage", "GPU enabled (default)", "GPU disabled", "Out-of-core mode"]
    xs = [1, 36, 58, 79]
    rows = [
        (
            "Quality control, normalisation, log transform",
            "CPU, 11 threads",
            "CPU, 11 threads",
            "SSD, row blocks, CPU",
        ),
        ("Highly variable genes", "CPU, 11 threads", "CPU, 11 threads", "SSD, row blocks, CPU"),
        ("Scaling", "CPU, 11 threads", "CPU, 11 threads", "SSD, row blocks, CPU"),
        ("PCA", "GPU", "CPU and AMX", "SSD, row blocks, GPU"),
        ("k-NN graph", "GPU", "CPU, 11 threads", "in memory, GPU"),
        ("UMAP", "CPU, 11 threads", "CPU, 11 threads", "in memory, CPU"),
        ("Leiden", "CPU, 11 threads", "CPU, 11 threads", "in memory, CPU"),
        ("Wilcoxon marker test", "CPU, 11 threads", "CPU, 11 threads", "in memory, CPU"),
    ]
    yh = 34.5
    for x, name in zip(xs, cols, strict=True):
        label(x, yh, name, 12, bold=True)
    fig.add_shape(
        type="line", x0=1, y0=yh - 1.8, x1=99, y1=yh - 1.8, line=dict(color="#222", width=1.2)
    )
    for i, row in enumerate(rows):
        y = yh - 4 - i * 3.6
        if i % 2 == 0:
            fig.add_shape(
                type="rect",
                x0=1,
                y0=y - 1.8,
                x1=99,
                y1=y + 1.8,
                fillcolor="#000",
                opacity=0.03,
                line=dict(width=0),
                layer="below",
            )
        for x, text in zip(xs, row, strict=True):
            label(x, y, text, 12)
    ylast = yh - 4 - (len(rows) - 1) * 3.6 - 1.8
    fig.add_shape(type="line", x0=1, y0=ylast, x1=99, y1=ylast, line=dict(color="#222", width=1.2))
    fig.update_layout(title="F1. Architecture of Metalcyte and the execution unit of each stage")
    save(fig, "F1_chip_and_library", 1100, 1010)


# ----------------------------------------------------------------------------- F2


def f2_bytes():
    """`pp.scale` then `pp.pca` as a flow of bytes: scanpy's copies against metalcyte's one pass."""
    fig = go.Figure()
    blank_axes(fig, 100, 50)
    c = COLORS
    n = "115 868 cells x 2 000 genes"
    fig.add_annotation(x=25, y=48, text=f"<b>scanpy</b> ({n})", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=75, y=48, text="<b>metalcyte</b>", showarrow=False, font=dict(size=14))
    # scanpy column: boxes sized by bytes (height ∝ MB)
    # Box sizes are the measured +MB split over the copies scanpy's scale and PCA make
    # (a dense copy, a centred-and-scaled copy, the clipped result; ARPACK's work vectors).
    steps_a = [
        ("CSR log data (f32)", 0.6, c[0]),
        ("densify (dense copy)", 0.93, c[1]),
        ("centre and scale (second copy)", 0.93, c[1]),
        ("clip, result handed back", 0.93, c[1]),
        ("ARPACK: Lanczos work vectors and copies", 5.4, c[3]),
        ("X_pca (f32)", 0.02, c[0]),
    ]
    y = 44
    for text, gb, col in steps_a:
        h = max(2.2, gb * 1.7)
        box(fig, 6, y - h, 44, y, f"{text}: {gb:.2f} GB", col, size=11)
        y -= h + 1.2
    fig.add_annotation(
        x=25,
        y=y - 1,
        text="+2.6 GB at scale, +5.4 GB at PCA (measured)<br>1.0 s + 13.6 s",
        showarrow=False,
        font=dict(size=11, color="#444"),
    )
    steps_b = [
        ("CSR log data (f32)", 0.6, c[0]),
        ("one fused pass: scale into<br>the buffer numpy allocated (f32)", 0.93, c[1]),
        (
            "randomised range finder on the same buffer<br>(GPU or AMX read it in place): sketch + Gram",  # noqa: E501
            1.3,
            c[3],
        ),
        ("X_pca (f32)", 0.02, c[0]),
    ]
    y = 44
    for text, gb, col in steps_b:
        h = max(2.6, gb * 1.7)
        box(fig, 56, y - h, 94, y, f"{text}: {gb:.2f} GB", col, size=11)
        y -= h + 1.2
    fig.add_annotation(
        x=75,
        y=y - 1,
        text="+0.9 GB at scale, +1.3 GB at PCA (measured)<br>0.03 s + 0.9 s",
        showarrow=False,
        font=dict(size=11, color="#444"),
    )
    fig.update_layout(
        title="F2. Where the bytes go: scaling then PCA on the 117k atlas (box height ∝ bytes)"
    )
    save(fig, "F2_bytes", 1100, 620)


# ----------------------------------------------------------------------------- F3


def f3_streaming():
    """Out-of-core head: (a) data flow of the four passes, (b) peak memory against in-memory."""
    fig = go.Figure()
    blank_axes(fig, 120, 70)
    c = COLORS
    grey = "#444"

    def label(x, y, text, size=12, anchor="left", color="#222", bold=False):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{text}</b>" if bold else text,
            showarrow=False,
            xanchor=anchor,
            font=dict(size=size, color=color),
        )

    # ---- (a) data flow
    label(
        1,
        68.5,
        "(a) Data flow of the out-of-core head (wall time per pass, GPU enabled)",
        13,
        bold=True,
    )
    box(
        fig,
        1,
        44,
        17,
        60,
        "<b>Input</b><br>Count matrix on SSD<br>1 001 288 cells &#215;<br>45 676 genes<br>CSR, 4.8 GB",  # noqa: E501
        c[7],
        size=11,
    )
    passes = [
        (
            "Pass 1",
            "3.6 s",
            "Cell filter, normalisation,<br>log transform.<br>Per-gene sums accumulated",
            "2 000 highly<br>variable genes",
        ),
        (
            "Pass 2",
            "5.3 s",
            "Mean and standard<br>deviation of the<br>selected genes",
            "Scaling parameters",
        ),
        (
            "Pass 3",
            "15.4 s",
            "Scaling of each block.<br>Scatter matrix accumulated<br>on the GPU",
            "50 principal axes<br>(subspace iteration)",
        ),
        (
            "Pass 4",
            "7.5 s",
            "Scaling and projection<br>of each block<br>onto the axes",
            "PCA embedding<br>953 436 &#215; 50",
        ),
    ]
    x = 21
    for name, secs, inside, out in passes:
        box(fig, x, 48, x + 22, 60, f"<b>{name}</b> ({secs})<br>{inside}", c[1], size=11)
        box(fig, x, 37, x + 22, 43.5, f"<b>Output:</b> {out}", c[2], size=11)
        arrow(fig, x + 11, 48, x + 11, 43.5, color=grey)
        arrow(fig, (17 if x == 21 else x - 3), 54, x, 54, color=grey)
        x += 25
    label(21, 33, "Each pass reads the file once in row blocks of 22 920 cells.", 11, color=grey)

    # ---- (b) peak memory
    label(1, 27.5, "(b) Peak memory of the preprocessing and PCA stages", 13, bold=True)
    x0, x1, gmax = 30, 118, 40.0

    def sx(gb):
        return x0 + (x1 - x0) * gb / gmax

    # axis
    for gb in range(0, 41, 10):
        fig.add_shape(
            type="line", x0=sx(gb), y0=5, x1=sx(gb), y1=5.8, line=dict(color="#222", width=1)
        )
        label(sx(gb), 3.5, f"{gb}", 10, anchor="center")
    fig.add_shape(type="line", x0=x0, y0=5.8, x1=x1, y1=5.8, line=dict(color="#222", width=1))
    label((x0 + x1) / 2, 1.2, "GB", 10, anchor="center")
    # in-memory pipeline: stacked segments
    segs = [
        ("count matrix", 4.7, c[7]),
        ("normalised copy", 4.7, c[0]),
        ("dense scaled matrix", 7.6, c[3]),
        ("working copies during scaling", 21.9, c[1]),
    ]
    label(1, 21, "In-memory pipeline<br>(scanpy)", 11)
    acc = 0.0
    for name, gb, col in segs:
        fig.add_shape(
            type="rect",
            x0=sx(acc),
            y0=18.5,
            x1=sx(acc + gb),
            y1=23.5,
            fillcolor=col,
            opacity=0.5,
            line=dict(color="white", width=1),
        )
        label((sx(acc) + sx(acc + gb)) / 2, 21, f"{name}<br>{gb} GB", 9, anchor="center")
        acc += gb
    # out-of-core head
    label(1, 12, "Out-of-core head<br>(Metalcyte)", 11)
    fig.add_shape(
        type="rect",
        x0=sx(0),
        y0=9.5,
        x1=sx(1.1),
        y1=14.5,
        fillcolor=c[2],
        opacity=0.6,
        line=dict(color="white", width=1),
    )
    label(sx(1.1) + 1, 12, "one row block and the embedding, 1.1 GB", 10)
    # available memory
    fig.add_shape(
        type="line",
        x0=sx(18),
        y0=7,
        x1=sx(18),
        y1=26,
        line=dict(color="#a33", width=1.5, dash="dash"),
    )
    label(sx(18), 26.5, "available memory, 18 GB", 10, anchor="center", color="#a33")
    fig.update_layout(title="F3. The out-of-core head: data flow and peak memory")
    save(fig, "F3_streaming", 1300, 760)


# ----------------------------------------------------------------------------- F4..F7 (measured)


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def f4_ablation():
    data = _load("ablation_bm117k.json")
    if not data:
        print("F4 skipped: benches/results/ablation_bm117k.json not there yet")
        return
    rows = [r for r in data["rows"] if "total_seconds" in r]
    steps = ["pp.pca", "pp.neighbors", "tl.umap", "tl.leiden", "tl.rank_genes_groups"]
    fig = go.Figure()
    for s in steps:
        fig.add_bar(
            name=step_label(s),
            x=[CONFIG_LABELS.get(r["config"], r["config"]) for r in rows],
            y=[r["steps"].get(s, 0) for r in rows],
            marker_line_width=0,
        )
    other = [r["total_seconds"] - sum(r["steps"].get(s, 0) for s in steps) for r in rows]
    fig.add_bar(
        name="Other stages",
        x=[CONFIG_LABELS.get(r["config"], r["config"]) for r in rows],
        y=other,
        marker_line_width=0,
    )
    fig.update_layout(
        barmode="stack",
        title="F4. Ablation on the 117k atlas: seconds with one Apple-specific choice switched off",
        yaxis_title="Wall time of the whole pipeline (s)",
        legend_title="",
    )
    save(fig, "F4_ablation", 1000, 560)


def f6_pipeline_117k():
    names = {
        "scanpy": "scanpy (defaults)",
        "scanpy_tuned": "scanpy (tuned)",
        "metalcyte_cpu": "Metalcyte CPU",
        "metalcyte_metal": "Metalcyte Metal",
        "metalcyte_metal_umap_parallel": "Metalcyte Metal + parallel UMAP",
    }
    runs = {k: _load(f"bm117k_{k}.json") for k in names}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print("F6 skipped: no bm117k_*.json")
        return
    steps = [s["step"] for s in next(iter(runs.values()))["steps"]]
    fig = go.Figure()
    for k, r in runs.items():
        fig.add_bar(
            name=names[k],
            x=[step_label(st) for st in steps],
            y=[s["seconds"] for s in r["steps"]],
            marker_line_width=0,
        )
    fig.update_layout(
        barmode="group",
        yaxis_type="log",
        yaxis_title="Wall time (s)",
        title="F6. The 117 308-cell bone-marrow atlas, step by step<br><sup>whole pipeline: "
        + ", ".join(f"{names[k]} {r['total_seconds']:.0f} s" for k, r in runs.items())
        + "</sup>",
    )
    save(fig, "F6_pipeline_117k", 1300, 600)


def f7_energy():
    runs = {
        k: _load(f"energy_bm117k_{k}.json") for k in ("scanpy", "metalcyte_metal", "metalcyte_cpu")
    }
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print("F7 skipped: no energy_bm117k_*.json (run benches/run_energy.sh under sudo)")
        return
    fig = go.Figure()
    run_labels = {
        "scanpy": "scanpy (defaults)",
        "metalcyte_metal": "Metalcyte (GPU)",
        "metalcyte_cpu": "Metalcyte (CPU)",
    }
    rail_labels = {"CPU": "CPU", "GPU": "GPU", "ANE": "Neural engine"}
    for rail in ("CPU", "GPU", "ANE"):
        fig.add_bar(
            name=rail_labels[rail],
            x=[run_labels.get(k, k) for k in runs],
            y=[r["net_joules"].get(rail, 0) for r in runs.values()],
            marker_line_width=0,
        )
    fig.update_layout(
        barmode="stack",
        yaxis_title="Net energy per run (J)",
        title="F7. Energy of the 117k pipeline (idle draw subtracted)",
    )
    save(fig, "F7_energy", 900, 520)


def f9_parity():
    fig = go.Figure()
    # from the 20k-cell validation against scanpy covariance_eigh (same HVG set)
    cc = [0.99999] * 10 + [0.99979] * 20 + [0.99945] * 20
    fig.add_scatter(
        x=list(range(1, 51)),
        y=cc,
        mode="markers",
        marker=dict(size=7, line_width=0),
        name="canonical correlation, streamed vs exact",
    )
    fig.update_layout(
        title="F9. Streamed PCA against scanpy's exact covariance solver, 19 770 cells: subspace agreement by component count",  # noqa: E501
        xaxis_title="Leading components compared",
        yaxis_title="Smallest canonical correlation",
        yaxis_range=[0.999, 1.00002],
    )
    save(fig, "F9_parity", 900, 480)


FIGURES = {
    "F1": f1_chip_and_library,
    "F2": f2_bytes,
    "F3": f3_streaming,
    "F4": f4_ablation,
    "F6": f6_pipeline_117k,
    "F7": f7_energy,
    "F9": f9_parity,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--paper", action="store_true", help="untitled versions for the manuscript")
    args = parser.parse_args()
    global PAPER
    PAPER = args.paper
    for key, fn in FIGURES.items():
        if args.only and key not in args.only:
            continue
        fn()
    return 0


# ----------------------------------------------------------------------------- F8


def _residency(path: Path):
    """Per-sample time, P-cluster, E-cluster and GPU active residency from a powermetrics log."""
    import re

    text = path.read_text(errors="replace")
    t, p, e, g, cpu_w, gpu_w = [], [], [], [], [], []
    clock = 0.0
    for sample in text.split("*** Sampled system activity")[1:]:
        m = re.search(r"\((\d+(?:\.\d+)?)ms elapsed\)", sample)
        if not m:
            continue
        clock += float(m.group(1)) / 1000.0

        def grab(pattern, text=sample):
            mm = re.search(pattern, text, re.M)
            return float(mm.group(1)) if mm else 0.0

        t.append(clock)
        p.append(grab(r"^P-Cluster HW active residency:\s+([\d.]+)%"))
        e.append(grab(r"^E-Cluster HW active residency:\s+([\d.]+)%"))
        g.append(grab(r"^GPU HW active residency:\s+([\d.]+)%"))
        cpu_w.append(grab(r"^CPU Power: (\d+) mW") / 1000.0)
        gpu_w.append(grab(r"^GPU Power: (\d+) mW") / 1000.0)
    return t, p, e, g, cpu_w, gpu_w


def f8_utilisation():
    from plotly.subplots import make_subplots

    runs = [("metalcyte_metal", "Metalcyte, Metal"), ("scanpy", "scanpy")]
    logs = [(RESULTS / f"energy_bm117k_{k}.powermetrics.txt", n) for k, n in runs]
    logs = [(p, n) for p, n in logs if p.exists()]
    if not logs:
        print("F8 skipped: no energy_*.powermetrics.txt")
        return
    fig = make_subplots(
        rows=len(logs),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[n for _, n in logs],
        vertical_spacing=0.14,
    )
    for row, (path, _) in enumerate(logs, start=1):
        t, p, e, g, cpu_w, _gpu_w = _residency(path)
        for y, name, col in (
            (p, "Performance cluster", COLORS[0]),
            (e, "Efficiency cluster", COLORS[1]),
            (g, "GPU", COLORS[2]),
        ):
            fig.add_scatter(
                x=t,
                y=y,
                name=name,
                mode="lines",
                line=dict(width=1.2, color=col),
                row=row,
                col=1,
                showlegend=row == 1,
            )
        fig.add_scatter(
            x=t,
            y=[w * 10 for w in cpu_w],
            name="CPU power (W &#215; 10)",
            mode="lines",
            line=dict(width=1, color=COLORS[3], dash="dot"),
            row=row,
            col=1,
            showlegend=row == 1,
        )
        fig.update_xaxes(title_text="Time (s)", row=row, col=1)
        fig.update_yaxes(title_text="Active residency (%)", range=[0, 105], row=row, col=1)
    fig.update_layout(
        title="F8. Who is busy: cluster and GPU active residency through the 117k pipeline"
        "<br><sup>powermetrics, 100 ms samples</sup>"
    )
    save(fig, "F8_utilisation", 1300, 320 * len(logs) + 120)


FIGURES["F8"] = f8_utilisation


# ----------------------------------------------------------------------------- F5


def f5_scaling():
    data = _load("scaling_embryo.json")
    if not data:
        print("F5 skipped: no scaling_embryo.json")
        return
    names = {
        "scanpy": "scanpy (defaults)",
        "scanpy_tuned": "scanpy (tuned)",
        "metalcyte_cpu": "Metalcyte CPU",
        "metalcyte_metal": "Metalcyte Metal",
    }
    from plotly.subplots import make_subplots

    steps = ["whole pipeline", "pp.pca", "pp.neighbors", "tl.umap", "tl.leiden"]
    fig = make_subplots(
        rows=1, cols=len(steps), subplot_titles=[step_label(st) for st in steps], shared_yaxes=True
    )
    for i, (key, label) in enumerate(names.items()):
        pts = sorted(
            (p for p in data["points"] if p["config"] == key and p["outcome"] == "ok"),
            key=lambda p: p["size"],
        )
        if not pts:
            continue
        x = [p["n_cells"] or p["size"] for p in pts]
        for col, step in enumerate(steps, start=1):
            y = [
                p["total_seconds"] if step == "whole pipeline" else p["steps"].get(step)
                for p in pts
            ]
            fig.add_scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                legendgroup=key,
                showlegend=col == 1,
                marker=dict(size=7, line_width=0, color=COLORS[i]),
                line=dict(color=COLORS[i]),
                row=1,
                col=col,
            )
    methods = _load("knn_methods_embryo.json")
    if methods:
        col = steps.index("pp.neighbors") + 1
        fig.add_scatter(
            x=[r["size"] for r in methods["rows"]],
            y=[r["approximate_s"] for r in methods["rows"]],
            mode="lines+markers",
            name="Metalcyte approximate",
            marker=dict(size=7, line_width=0, color=COLORS[4]),
            line=dict(color=COLORS[4], dash="dash"),
            row=1,
            col=col,
        )
    fig.update_xaxes(
        type="log",
        title_text="Cells",
        tickvals=[1e4, 1e5, 1e6],
        ticktext=["10k", "100k", "1M"],
        range=[3.9, 6.1],
    )
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Wall time (s)", row=1, col=1)
    fig.update_layout(
        title="F5. Seconds against cells on random subsamples of the 1 M-cell embryo atlas (log-log)"  # noqa: E501
    )
    save(fig, "F5_scaling", 1500, 480)


# ----------------------------------------------------------------------------- F10


def f10_umap_1m():
    path = RESULTS / "embryo1m_metalcyte_metal.h5ad"
    if not path.exists():
        print(
            "F10 skipped: run pipeline_1m.py --save benches/results/embryo1m_metalcyte_metal.h5ad"
        )
        return
    import anndata

    a = anndata.read_h5ad(path)
    xy = a.obsm["X_umap"]
    types = a.obs["cell_type"].astype(str) if "cell_type" in a.obs else a.obs["leiden"].astype(str)
    counts = types.value_counts()
    keep = counts.index[:20]
    label = types.where(types.isin(keep), "other")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(a))[:300_000]  # a 300k subsample draws at a sane size
    fig = go.Figure()
    for i, t in enumerate([*list(keep), "other"]):
        m = label.to_numpy()[order] == t
        if not m.any():
            continue
        fig.add_scattergl(
            x=xy[order][m, 0],
            y=xy[order][m, 1],
            mode="markers",
            name=f"{t} ({int((label == t).sum()):,})",
            marker=dict(
                size=2,
                opacity=0.6,
                line_width=0,
                color=COLORS[i % len(COLORS)] if t != "other" else "#cccccc",
            ),
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(
        title=f"F10. {len(a):,} embryo cells, UMAP by author cell type (top 20 types; 300k points drawn)",  # noqa: E501
        legend=dict(itemsizing="constant", font=dict(size=10)),
    )
    save(fig, "F10_umap_1m", 1300, 1000)


# ----------------------------------------------------------------------------- F11


def f11_agreement():
    data = _load("agreement_bm117k.json")
    if not data:
        print("F11 skipped: no agreement_bm117k.json")
        return
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Intermediate results", "Leiden clustering", "Marker genes per cell type"],
    )
    left = {
        "Variable genes, Jaccard": data["hvg_jaccard"],
        "PCA, canonical correlation (10)": data["pca_canonical_min_top10"],
        "PCA, canonical correlation (30)": data["pca_canonical_min_top30"],
        "PCA, canonical correlation (50)": data["pca_canonical_min_top50"],
        "Neighbour overlap, own PCA": data["knn_overlap_own_pca"],
        "Neighbour overlap, same PCA": data["knn_overlap_common_pca"],
    }
    fig.add_bar(
        x=list(left), y=list(left.values()), marker_line_width=0, showlegend=False, row=1, col=1
    )
    ld = data["leiden"]
    mid = {
        "ARI, scanpy against Metalcyte": ld["ari_scanpy_vs_metalcyte"],
        "NMI, scanpy against Metalcyte": ld["nmi_scanpy_vs_metalcyte"],
        "NMI, scanpy against cell type": ld["nmi_scanpy_vs_celltype"],
        "NMI, Metalcyte against cell type": ld["nmi_metalcyte_vs_celltype"],
    }
    fig.add_bar(
        x=list(mid), y=list(mid.values()), marker_line_width=0, showlegend=False, row=1, col=2
    )
    groups = sorted(data["markers"], key=lambda g: -data["markers"][g]["spearman"])
    fig.add_bar(
        x=groups,
        y=[data["markers"][g]["spearman"] for g in groups],
        name="Spearman correlation of scores",
        marker_line_width=0,
        row=1,
        col=3,
    )
    fig.add_bar(
        x=groups,
        y=[data["markers"][g]["top50_overlap"] for g in groups],
        name="Overlap of the top 50 genes",
        marker_line_width=0,
        row=1,
        col=3,
    )
    fig.update_yaxes(range=[0, 1.02])
    fig.update_xaxes(tickangle=35, row=1, col=3)
    fig.update_layout(
        barmode="group",
        title=f"F11. metalcyte against scanpy on the 117k bone-marrow atlas, {data['n_cells']:,} cells, same seeds",  # noqa: E501
    )
    save(fig, "F11_agreement", 1500, 560)


def f12_knn_methods():
    data = _load("knn_methods_embryo.json")
    if not data:
        print("F12 skipped: no knn_methods_embryo.json")
        return
    from plotly.subplots import make_subplots

    rows = data["rows"]
    x = [r["size"] for r in rows]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Wall time of the neighbour search", "Recall of the approximate search"],
        column_widths=[0.6, 0.4],
    )
    series = [
        ("exact_metal_s", "Exact, GPU", COLORS[0]),
        ("exact_cpu_s", "Exact, CPU", COLORS[1]),
        ("approximate_s", "Approximate, CPU", COLORS[2]),
    ]
    for key, name, color in series:
        pts = [(r["size"], r[key]) for r in rows if key in r]
        fig.add_scatter(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            mode="lines+markers",
            name=name,
            marker=dict(size=7, line_width=0, color=color),
            line=dict(color=color),
            row=1,
            col=1,
        )
    fig.add_scatter(
        x=x,
        y=[r.get("recall") for r in rows],
        mode="lines+markers",
        name="Recall at k = 15",
        showlegend=False,
        marker=dict(size=7, line_width=0, color=COLORS[2]),
        line=dict(color=COLORS[2]),
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.update_xaxes(
            type="log",
            title_text="Cells",
            tickvals=[1e4, 1e5, 1e6],
            ticktext=["10k", "100k", "1M"],
            range=[3.9, 6.1],
            row=1,
            col=col,
        )
    fig.update_yaxes(type="log", title_text="Wall time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Recall", range=[0.9, 1.005], row=1, col=2)
    fig.update_layout(
        title="F12. Exact against approximate neighbour search on the embryo embedding"
    )
    save(fig, "F12_knn_methods", 1100, 420)


FIGURES["F5"] = f5_scaling
FIGURES["F12"] = f12_knn_methods
FIGURES["F10"] = f10_umap_1m
FIGURES["F11"] = f11_agreement


if __name__ == "__main__":
    raise SystemExit(main())
