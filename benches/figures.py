#!/usr/bin/env python3
"""Every figure of `docs/SCALE.md` and the paper, from one script.

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

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benches" / "results"
OUT = ROOT / "docs" / "figures"
pio.templates.default = "plotly_white"
COLORS = pio.templates["plotly"].layout.colorway  # the default colorway
FONT = dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=13)


def save(fig: go.Figure, name: str, width: int, height: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.update_layout(font=FONT, margin=dict(l=20, r=20, t=50, b=20))
    for ext in ("svg", "png"):
        fig.write_image(
            OUT / f"{name}.{ext}", width=width, height=height, scale=1 if ext == "svg" else 3
        )
    print(f"wrote docs/figures/{name}.svg/.png")


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
    """The M3 Pro as blocks and the scrust layer that drives each one."""
    fig = go.Figure()
    blank_axes(fig, 100, 62)
    c = COLORS
    # software stack, left column
    box(fig, 3, 50, 30, 58, "<b>Python</b><br>scanpy-shaped API, AnnData", c[0])
    box(fig, 3, 39, 30, 47, "<b>PyO3 bindings</b><br>borrow numpy buffers, release the GIL", c[0])
    box(fig, 3, 28, 30, 36, "<b>Rust core</b><br>rayon work-stealing, f32 kernels", c[1])
    box(fig, 3, 17, 30, 25, "<b>candle + Accelerate</b><br>matmul, eigen, projection", c[1])
    box(fig, 3, 6, 30, 14, "<b>Metal kernels (MSL)</b><br>k-NN tiles, SpMM, UMAP SGD", c[1])
    arrow(fig, 16.5, 50, 16.5, 47)
    arrow(fig, 16.5, 39, 16.5, 36)
    arrow(fig, 16.5, 28, 16.5, 25)
    arrow(fig, 16.5, 17, 16.5, 14)
    # the chip, right block
    fig.add_shape(
        type="rect", x0=38, y0=3, x1=97, y1=59, line=dict(color="#888", width=1.5, dash="dot")
    )
    fig.add_annotation(
        x=67.5, y=57, text="<b>Apple M3 Pro</b> (one package)", showarrow=False, font=dict(size=14)
    )
    # P cores
    for i in range(5):
        box(fig, 41 + i * 5.2, 44, 45.6 + i * 5.2, 52, f"P{i}", c[2], size=11)
    fig.add_annotation(
        x=53, y=53.5, text="5 performance cores", showarrow=False, font=dict(size=11)
    )
    # E cores
    for i in range(6):
        box(fig, 68 + i * 4.6, 44, 72 + i * 4.6, 52, f"E{i}", c[4], size=11)
    fig.add_annotation(x=81, y=53.5, text="6 efficiency cores", showarrow=False, font=dict(size=11))
    box(fig, 41, 33, 66, 41, "<b>AMX</b> matrix units<br>(Accelerate BLAS)", c[3])
    box(fig, 68, 33, 95, 41, "<b>GPU</b> 14 cores<br>(Metal)", c[5])
    box(
        fig,
        41,
        20,
        95,
        30,
        "<b>Unified memory</b> 18 GB, 150 GB/s<br>CPU, AMX and GPU read the same buffer: no copy to a device",  # noqa: E501
        c[6],
    )
    box(
        fig, 41, 8, 95, 16, "<b>SSD</b> ~3 GB/s: a million cells stream through in row blocks", c[7]
    )
    arrow(fig, 95, 20, 95, 16)
    arrow(fig, 41, 16, 41, 20)
    # software → hardware arrows
    arrow(fig, 30, 32, 41, 48, "all 11 cores", c[1], at=0.62)
    arrow(fig, 30, 21, 41, 37, "sgemm, ssyrk", c[1], at=0.3)
    arrow(fig, 30, 10, 68, 37, "command buffers", c[1], at=0.3)
    arrow(fig, 30, 43, 41, 25, "numpy's own bytes", c[0], at=0.15)
    fig.update_layout(title="F1. What runs where: the scrust stack on an Apple silicon package")
    save(fig, "F1_chip_and_library", 1100, 700)


# ----------------------------------------------------------------------------- F2


def f2_bytes():
    """`pp.scale` then `pp.pca` as a flow of bytes: scanpy's copies against scrust's one pass."""
    fig = go.Figure()
    blank_axes(fig, 100, 50)
    c = COLORS
    n = "115 868 cells x 2 000 genes"
    fig.add_annotation(x=25, y=48, text=f"<b>scanpy</b> ({n})", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=75, y=48, text="<b>scrust</b>", showarrow=False, font=dict(size=14))
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
        title="F2. Where the bytes go: pp.scale then pp.pca on the 117k atlas (box height ∝ bytes)"
    )
    save(fig, "F2_bytes", 1100, 620)


# ----------------------------------------------------------------------------- F3


def f3_streaming():
    """The four passes of `pp.preprocess_backed` next to what an in-memory pipeline holds."""
    fig = go.Figure()
    blank_axes(fig, 120, 56)
    c = COLORS
    box(
        fig,
        2,
        20,
        14,
        36,
        "<b>counts.h5ad</b><br>1 001 288 x 45 676<br>586 M values<br>4.8 GB on SSD",
        c[7],
        size=11,
    )
    passes = [
        (
            "pass 1",
            "row block (22 920 cells)<br>drop cells < 200 genes<br>normalise, log1p in place<br>per-gene presence, Σx, Σx²",  # noqa: E501
            "HVG: 2 000 genes<br>from the sums",
            3.6,
        ),
        (
            "pass 2",
            "same transform<br>keep the 2 000 columns<br>Σx, Σx² of the log data",
            "mean, sd per gene",
            5.3,
        ),
        (
            "pass 3",
            "scale against mean, sd<br>into a dense block<br>Bᵀ B on the GPU → scatter (2000²)",
            "top-50 eigenvectors<br>by subspace iteration",
            15.4,
        ),
        ("pass 4", "scale again<br>block @ loadings on the GPU", "X_pca rows (953 436 x 50)", 7.5),
    ]
    x = 18
    for name, inside, out, secs in passes:
        box(fig, x, 22, x + 22, 36, f"<b>{name}</b> {secs:.1f} s<br>{inside}", c[1], size=10)
        box(fig, x, 10, x + 22, 18, out, c[2], size=10)
        arrow(fig, x + 11, 22, x + 11, 18)
        arrow(fig, 14, 28, x, 28) if x == 18 else arrow(fig, x - 3, 29, x, 29)
        x += 25
    fig.add_annotation(
        x=68,
        y=40,
        text="each pass streams the file once; peak memory = one block (≈ 0.6 GB) + the embedding (190 MB)",  # noqa: E501
        showarrow=False,
        font=dict(size=12),
    )
    # what scanpy would hold
    box(
        fig,
        18,
        44,
        118,
        54,
        "<b>in-memory pipeline</b>: CSR counts 4.7 GB → normalised copy 4.7 GB → dense scaled f32 7.6 GB (+ f64 working copies during scale: measured +21.9 GB) → PCA",  # noqa: E501
        c[3],
        size=11,
    )
    fig.add_annotation(
        x=68,
        y=55.5,
        text="does not fit 18 GB: on this laptop scanpy's scale step swapped and PCA never finished",  # noqa: E501
        showarrow=False,
        font=dict(size=11, color="#a33"),
    )
    fig.update_layout(
        title="F3. A million cells in four passes: pp.preprocess_backed on an 18 GB laptop (times measured, Metal)"  # noqa: E501
    )
    save(fig, "F3_streaming", 1300, 620)


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
            name=s,
            x=[r["config"] for r in rows],
            y=[r["steps"].get(s, 0) for r in rows],
            marker_line_width=0,
        )
    other = [r["total_seconds"] - sum(r["steps"].get(s, 0) for s in steps) for r in rows]
    fig.add_bar(name="everything else", x=[r["config"] for r in rows], y=other, marker_line_width=0)
    fig.update_layout(
        barmode="stack",
        title="F4. Ablation on the 117k atlas: seconds with one Apple-specific choice switched off",
        yaxis_title="seconds (whole pipeline)",
        legend_title="",
    )
    save(fig, "F4_ablation", 1000, 560)


def f6_pipeline_117k():
    names = {
        "scanpy": "scanpy (defaults)",
        "scanpy_tuned": "scanpy (tuned)",
        "scrust_cpu": "scrust CPU",
        "scrust_metal": "scrust Metal",
        "scrust_metal_umap_parallel": "scrust Metal + parallel UMAP",
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
            x=steps,
            y=[s["seconds"] for s in r["steps"]],
            marker_line_width=0,
        )
    fig.update_layout(
        barmode="group",
        yaxis_type="log",
        yaxis_title="seconds (log)",
        title="F6. The 117 308-cell bone-marrow atlas, step by step<br><sup>whole pipeline: "
        + ", ".join(f"{names[k]} {r['total_seconds']:.0f} s" for k, r in runs.items())
        + "</sup>",
    )
    save(fig, "F6_pipeline_117k", 1300, 600)


def f7_energy():
    runs = {k: _load(f"energy_bm117k_{k}.json") for k in ("scanpy", "scrust_metal", "scrust_cpu")}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print("F7 skipped: no energy_bm117k_*.json (run benches/run_energy.sh under sudo)")
        return
    fig = go.Figure()
    for rail in ("CPU", "GPU", "ANE"):
        fig.add_bar(
            name=f"{rail} rail",
            x=list(runs),
            y=[r["net_joules"].get(rail, 0) for r in runs.values()],
            marker_line_width=0,
        )
    fig.update_layout(
        barmode="stack",
        yaxis_title="net joules per pipeline run",
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
        xaxis_title="leading components compared",
        yaxis_title="smallest canonical correlation",
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
    args = parser.parse_args()
    for key, fn in FIGURES.items():
        if args.only and key not in args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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

    runs = [("scrust_metal", "scrust, Metal"), ("scanpy", "scanpy")]
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
            (p, "P cores active %", COLORS[0]),
            (e, "E cores active %", COLORS[1]),
            (g, "GPU active %", COLORS[2]),
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
            name="CPU power (W x10)",
            mode="lines",
            line=dict(width=1, color=COLORS[3], dash="dot"),
            row=row,
            col=1,
            showlegend=row == 1,
        )
        fig.update_xaxes(title_text="seconds", row=row, col=1)
        fig.update_yaxes(title_text="%", range=[0, 105], row=row, col=1)
    fig.update_layout(
        title="F8. Who is busy: cluster and GPU active residency through the 117k pipeline"
        "<br><sup>powermetrics, 100 ms samples</sup>"
    )
    save(fig, "F8_utilisation", 1300, 320 * len(logs) + 120)


FIGURES["F8"] = f8_utilisation
