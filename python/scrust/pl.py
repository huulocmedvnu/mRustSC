"""Native plotting for scrust — publication-grade matplotlib/seaborn, no scanpy.

`sr.pl` draws the figures a single-cell analysis needs from the AnnData slots scrust
writes: the PCA spectrum from `uns["pca"]`, the UMAP embedding from `obsm["X_umap"]`
coloured by an `obs` column or a gene, and the differential-expression ranking from
`uns["rank_genes_groups"]`. It depends only on matplotlib (and seaborn for palettes when
present); it never imports scanpy.

Every function shares the house style: clean spines, a subtle grid, a modern sans-serif,
categorical clusters in a perceptually even palette (seaborn ``husl``) and gene
expression in a perceptual continuous colormap (``viridis``). `show` displays the figure
and `save` writes it; both can be combined.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as _sns

    _HAS_SEABORN = True
except ImportError:  # matplotlib-only fallback; palettes degrade gracefully
    _HAS_SEABORN = False

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["pca_variance_ratio", "rank_genes_groups", "umap"]

# A safe-but-modern sans stack: DejaVu Sans is always present (no findfont warning),
# the others are used when the machine has them.
_FONT_STACK = ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"]
_ACCENT = "#2f6fd0"  # a calm blue for single-series marks
_ACCENT_2 = "#e4572e"  # a warm accent for the cumulative trend


def _style() -> dict[str, Any]:
    """rcParams for the house style, applied through `rc_context` so nothing leaks."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": _FONT_STACK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3a3a3a",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#222222",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }


def _categorical_palette(n: int, name: str = "husl") -> list:
    """`n` perceptually even colours; seaborn when available, else a matplotlib cycle."""
    if _HAS_SEABORN:
        return _sns.color_palette(name, n)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def _finish(fig: Figure, result: Any, show: bool, save: str | Path | None) -> Any:
    """Save and/or show a finished figure, mirroring scanpy's `show`/`save` contract."""
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()
        return None
    return result


def _bare(ax: Axes) -> None:
    """Strip a scatter axis down to the data: no spines, ticks or grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _expression(adata: AnnData, key: str) -> np.ndarray | None:
    """A per-cell vector for `key`: a numeric `obs` column or a gene's expression.

    Genes are read from `adata.raw` when it is set (the log-normalised matrix), else from
    `adata.X`. Returns `None` if `key` is not a numeric obs column or a known gene.
    """
    if key in adata.obs.columns:
        series = adata.obs[key]
        if series.dtype.kind in "biufc":
            return np.asarray(series, dtype=float)
        return None
    source = adata.raw if adata.raw is not None else adata
    names = source.var_names
    if key not in names:
        return None
    index = int(names.get_loc(key))
    column = source.X[:, index]
    dense = column.toarray() if hasattr(column, "toarray") else np.asarray(column)
    return np.asarray(dense, dtype=float).ravel()


def pca_variance_ratio(
    adata: AnnData,
    n_pcs: int = 30,
    *,
    show: bool = True,
    save: str | Path | None = None,
) -> Axes | None:
    """Elbow plot of the PCA spectrum: per-component bars and a cumulative trend line.

    Reads `adata.uns["pca"]["variance_ratio"]`, written by `sr.pp.pca`.
    """
    ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)
    n = int(min(n_pcs, ratio.size))
    ratio = ratio[:n]
    cumulative = np.cumsum(ratio)
    x = np.arange(1, n + 1)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(x, ratio, width=0.72, color=_ACCENT, alpha=0.9, label="per component")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance ratio")
        ax.set_xlim(0.4, n + 0.6)
        ax.margins(y=0.08)

        trend = ax.twinx()
        trend.plot(x, cumulative, color=_ACCENT_2, marker="o", markersize=4, linewidth=2)
        trend.set_ylabel("Cumulative variance", color=_ACCENT_2)
        trend.tick_params(axis="y", colors=_ACCENT_2)
        trend.set_ylim(0, min(1.0, cumulative[-1] * 1.08))
        trend.grid(False)
        trend.spines["top"].set_visible(False)
        trend.spines["right"].set_visible(True)
        trend.spines["right"].set_color(_ACCENT_2)

        ax.set_title(f"PCA variance explained — first {n} components")
        result = fig.axes
    return _finish(fig, result, show, save)


def umap(
    adata: AnnData,
    color: str | None = None,
    *,
    title: str | None = None,
    palette: str = "husl",
    frameon: bool = False,
    alpha: float = 0.85,
    size: float = 12,
    figsize: tuple[float, float] = (7, 6),
    show: bool = True,
    save: str | Path | None = None,
) -> Axes | None:
    """Scatter of `adata.obsm["X_umap"]`, coloured by a categorical `obs` column or a gene.

    A string/categorical `obs` column (e.g. ``"leiden"``) draws one colour per level with a
    legend; a numeric column or a gene name draws a `viridis` colour bar. `color=None` is a
    single-colour scatter.
    """
    coords = np.asarray(adata.obsm["X_umap"], dtype=float)
    x, y = coords[:, 0], coords[:, 1]

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=figsize)
        heading = title if title is not None else (color or "UMAP")

        categorical = (
            color is not None
            and color in adata.obs.columns
            and adata.obs[color].dtype.kind not in "biufc"
        )
        if color is None:
            ax.scatter(x, y, s=size, c=_ACCENT, alpha=alpha, linewidths=0)
        elif categorical:
            cats = adata.obs[color].astype("category")
            levels = list(cats.cat.categories)
            codes = cats.cat.codes.to_numpy()
            colours = _categorical_palette(len(levels), palette)
            for i, level in enumerate(levels):
                mask = codes == i
                ax.scatter(
                    x[mask], y[mask], s=size, color=colours[i],
                    alpha=alpha, linewidths=0, label=str(level),
                )
            ncol = 1 if len(levels) <= 14 else 2
            legend = ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1.0), title=color,
                markerscale=2.0, ncol=ncol, handletextpad=0.3, borderaxespad=0.0,
            )
            legend.get_title().set_fontweight("bold")
        else:
            values = _expression(adata, color)
            if values is None:
                raise KeyError(
                    f"{color!r} is not a numeric obs column or a gene in var_names / raw"
                )
            points = ax.scatter(
                x, y, s=size, c=values, cmap="viridis", alpha=alpha, linewidths=0
            )
            bar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.02)
            bar.set_label(color, rotation=90)
            bar.outline.set_visible(False)

        ax.set_title(heading)
        if frameon:
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        else:
            _bare(ax)
        ax.set_aspect("equal", adjustable="datalim")
        result = ax
    return _finish(fig, result, show, save)


def rank_genes_groups(
    adata: AnnData,
    n_genes: int = 10,
    n_cols: int = 4,
    *,
    show: bool = True,
    save: str | Path | None = None,
) -> np.ndarray | None:
    """Multi-panel bar chart of the top `n_genes` marker genes per group by score.

    Reads `adata.uns["rank_genes_groups"]`, written by `sr.tl.rank_genes_groups`.
    """
    record = adata.uns["rank_genes_groups"]
    names, scores = record["names"], record["scores"]
    groups = list(names.dtype.names)
    n_groups = len(groups)
    n_cols = max(1, min(n_cols, n_groups))
    n_rows = int(np.ceil(n_groups / n_cols))
    palette = _categorical_palette(n_groups, "husl")

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.1, n_rows * 2.9), squeeze=False
        )
        flat = axes.ravel()
        for index, group in enumerate(groups):
            ax = flat[index]
            gene_names = np.asarray(names[group][:n_genes], dtype=object)
            gene_scores = np.asarray(scores[group][:n_genes], dtype=float)
            positions = np.arange(gene_names.size)[::-1]
            ax.barh(positions, gene_scores, color=palette[index], alpha=0.9)
            ax.set_yticks(positions)
            ax.set_yticklabels(gene_names, fontsize=8)
            ax.set_title(f"group {group}")
            ax.set_xlabel("score")
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.margins(x=0.08)
        for spare in flat[n_groups:]:
            spare.set_visible(False)
        fig.suptitle("Top marker genes per group", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        result = axes
    return _finish(fig, result, show, save)
