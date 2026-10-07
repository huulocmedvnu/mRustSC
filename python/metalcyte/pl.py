"""Native plotting for metalcyte: embeddings rasterised on the GPU, matplotlib around them.

`mc.pl` draws the figures a single-cell analysis needs from the AnnData slots metalcyte
writes: the embeddings in `obsm` (`umap`, `tsne`, `pca`, `embedding`) coloured by an `obs`
column or a gene, the PCA spectrum from `uns["pca"]`, and the differential-expression
ranking from `uns["rank_genes_groups"]`. The embedding scatters are rendered by Metal
into one bitmap (`render_embedding`), so a million cells draw in milliseconds and a
notebook shows them as a single image; matplotlib supplies the axes, legend and colour
bar. It depends only on matplotlib (and seaborn for palettes when present); it never
imports scanpy.

Every function shares the house style: clean spines, a subtle grid, a modern sans-serif,
categorical clusters in plotly's qualitative palettes and gene expression in a
perceptual continuous colormap (``plasma``, plotly's default). `save` writes the figure
and `show` displays it; by default a figure is shown only when it is not saved.
"""

from __future__ import annotations

from collections.abc import Sequence
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

__all__ = [
    "embedding",
    "pca",
    "pca_variance_ratio",
    "rank_genes_groups",
    "render_embedding",
    "tsne",
    "umap",
]

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
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",  # keep text as text in vector output, not outlines
    }


# Plotly's qualitative palettes: the 10-colour default, then its mid-saturation Vivid
# and Safe sets for more levels, so a legend of thirty cell types still reads without
# the dark or neon colours of the larger sets. Hex values as plotly ships them.
_PLOTLY_10 = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
_PLOTLY_VIVID = [
    "#E58606",
    "#5D69B1",
    "#52BCA3",
    "#99C945",
    "#CC61B0",
    "#24796C",
    "#DAA51B",
    "#2F8AC4",
    "#764E9F",
    "#ED645A",
    "#CC3A8E",
]
_PLOTLY_SAFE = [
    "#88CCEE",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
    "#6699CC",
]


def _categorical_palette(n: int, name: str = "plotly") -> list:
    """`n` colours as RGB tuples: plotly's qualitative sets by default, a seaborn
    palette by name when seaborn is present, else a matplotlib cycle."""
    from matplotlib.colors import to_rgb

    if name == "plotly":
        hexes = _PLOTLY_10 if n <= len(_PLOTLY_10) else _PLOTLY_10 + _PLOTLY_VIVID + _PLOTLY_SAFE
        return [to_rgb(hexes[i % len(hexes)]) for i in range(n)]
    if _HAS_SEABORN:
        return [tuple(c[:3]) for c in _sns.color_palette(name, n)]
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N)[:3] for i in range(n)]


def _auto_point_size(n: int) -> float:
    """A scatter point size that keeps a dense embedding legible: smaller for more cells."""
    return float(np.clip(24000.0 / max(n, 1), 4.0, 18.0))


def _finish(
    fig: Figure,
    result: Any,
    show: bool | None,
    save: str | Path | None,
    extra_artists: Any = None,
) -> Any:
    """Save and/or show a finished figure.

    `show=None` (the default) displays the figure when nothing is saved and otherwise
    only writes the file, so a script that passes `save` never blocks on an interactive
    window. Saves with explicit `dpi=300` and `bbox_inches="tight"` because this runs
    after the `rc_context` has closed, so the style's savefig settings no longer apply.
    `extra_artists` (an outside legend) is folded into the tight bounding box so it is
    never clipped. A saved figure that is not shown is closed to release its memory.
    """
    if show is None:
        show = save is None
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", bbox_extra_artists=extra_artists)
    if show:
        plt.show()
        return None
    if save is not None:
        plt.close(fig)
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
    show: bool | None = None,
    save: str | Path | None = None,
) -> Axes | None:
    """Elbow plot of the PCA spectrum: per-component bars and a cumulative trend line.

    Reads `adata.uns["pca"]["variance_ratio"]`, written by `mc.pp.pca`.
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

        ax.set_title(f"PCA variance explained by the first {n} components")
        result = fig.axes
    return _finish(fig, result, show, save)


_DEFAULT_DPI = 300
_RENDER_MARGIN = 0.03


def _auto_pixel_size(n: int, dpi: float) -> float:
    """Point diameter in pixels that keeps a dense embedding legible at `dpi`."""
    return float(np.clip(900.0 / np.sqrt(max(n, 1)), 1.5, 24.0) * (dpi / _DEFAULT_DPI))


def _pack_rgba(colours: np.ndarray, alpha: float) -> np.ndarray:
    """`(n, 3)` or `(n, 4)` floats in [0, 1] to `0xRRGGBBAA` per point."""
    rgba = np.empty((colours.shape[0], 4), dtype=np.float64)
    rgba[:, :3] = colours[:, :3]
    rgba[:, 3] = colours[:, 3] * alpha if colours.shape[1] == 4 else alpha
    bytes_ = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint32)
    return (bytes_[:, 0] << 24) | (bytes_[:, 1] << 16) | (bytes_[:, 2] << 8) | bytes_[:, 3]


def _category_colours(adata: AnnData, key: str, palette: str) -> list:
    """One colour per level: `uns[f"{key}_colors"]` when scanpy or a user set it, else
    the palette."""
    levels = list(adata.obs[key].astype("category").cat.categories)
    stored = adata.uns.get(f"{key}_colors")
    if stored is not None and len(stored) == len(levels):
        from matplotlib.colors import to_rgb

        return [to_rgb(c) for c in stored]
    return _categorical_palette(len(levels), palette)


def render_embedding(
    adata: AnnData,
    basis: str = "X_umap",
    color: str | None = None,
    *,
    width: int = 2100,
    height: int = 1800,
    size: float | None = None,
    alpha: float = 1.0,
    palette: str = "plotly",
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    background: str = "white",
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterise `obsm[basis]` on the GPU into an RGBA image, `(height, width, 4)` uint8.

    This is the primitive behind `embedding`, `umap`, `tsne` and `pca`: a million cells
    become one bitmap in a few milliseconds, so a notebook shows them as a single image
    and a figure file holds pixels, not a million paths. Returns the image and a
    description of the colouring (`kind`, `levels`, `colours` or `vmin`/`vmax`) for a
    legend or colour bar.
    """
    from matplotlib.colors import to_rgba

    from metalcyte._shared import _extension, _resolve_device

    coords = np.ascontiguousarray(np.asarray(adata.obsm[basis], dtype=np.float32)[:, :2])
    n = coords.shape[0]
    info: dict[str, Any] = {"kind": "single"}
    if color is None:
        rgb = np.tile(np.asarray(to_rgba(_ACCENT)[:3], dtype=np.float64), (n, 1))
    elif color in adata.obs.columns and adata.obs[color].dtype.kind not in "biufc":
        cats = adata.obs[color].astype("category")
        colours = _category_colours(adata, color, palette)
        codes = cats.cat.codes.to_numpy()
        table = np.asarray([c[:3] for c in colours], dtype=np.float64)
        rgb = np.where((codes >= 0)[:, None], table[np.maximum(codes, 0)], 0.8)
        info = {"kind": "categorical", "levels": list(cats.cat.categories), "colours": colours}
    else:
        values = _expression(adata, color)
        if values is None:
            raise KeyError(f"{color!r} is not a numeric obs column or a gene in var_names / raw")
        lo = float(np.nanmin(values)) if vmin is None else vmin
        hi = float(np.nanmax(values)) if vmax is None else vmax
        unit = (values - lo) / max(hi - lo, 1e-12)
        table = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, 256))[:, :3]
        rgb = table[np.clip((np.nan_to_num(unit) * 255.0).astype(np.int64), 0, 255)]
        info = {"kind": "continuous", "vmin": lo, "vmax": hi, "cmap": cmap}
    rgba = _pack_rgba(rgb, alpha)

    finite = np.isfinite(coords).all(axis=1)
    if xlim is None or ylim is None:
        x0, x1 = (
            (float(coords[finite, 0].min()), float(coords[finite, 0].max()))
            if finite.any()
            else (0.0, 1.0)
        )
        y0, y1 = (
            (float(coords[finite, 1].min()), float(coords[finite, 1].max()))
            if finite.any()
            else (0.0, 1.0)
        )
        # Equal data scale on both axes, centred, so the cloud is not stretched.
        dx, dy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        span = max(dx / width, dy / height) * (1.0 + 2 * _RENDER_MARGIN)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xlim = xlim or (cx - span * width / 2, cx + span * width / 2)
        ylim = ylim or (cy - span * height / 2, cy + span * height / 2)
    dpi_size = _auto_pixel_size(n, _DEFAULT_DPI) if size is None else float(size)
    bg = to_rgba(background)
    bg_packed = int(_pack_rgba(np.asarray([bg[:3]]), bg[3])[0])
    image = _extension().render_points(
        coords,
        rgba.astype(np.uint32),
        int(width),
        int(height),
        float(dpi_size),
        float(xlim[0]),
        float(xlim[1]),
        float(ylim[0]),
        float(ylim[1]),
        bg_packed,
        _resolve_device(device),
    )
    info["xlim"], info["ylim"] = tuple(xlim), tuple(ylim)
    return np.asarray(image), info


def embedding(
    adata: AnnData,
    basis: str = "X_umap",
    color: str | Sequence[str] | None = None,
    *,
    title: str | Sequence[str] | None = None,
    palette: str = "plotly",
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    frameon: bool = False,
    alpha: float = 1.0,
    size: float | None = None,
    legend_loc: str = "right margin",
    legend_fontsize: float = 8.0,
    figsize: tuple[float, float] = (7, 6),
    dpi: int = _DEFAULT_DPI,
    ncols: int = 3,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    device: str | None = None,
    show: bool | None = None,
    save: str | Path | None = None,
) -> Axes | list[Axes] | None:
    """Scatter of `obsm[basis]`, rendered on the GPU, coloured by `obs` columns or genes.

    Mirrors `scanpy.pl.embedding`: `color` may be one key or several (one panel each,
    `ncols` across). A categorical column draws one colour per level and a legend, in the
    right margin or, with `legend_loc="on data"`, as labels at each level's median; a numeric
    column or a gene draws a colour bar. The points are rasterised by Metal into an image of
    `figsize * dpi` pixels, so a million cells draw in milliseconds and the figure stays
    light. `xlim`/`ylim` zoom into a window of the embedding.
    """
    keys: list[str | None] = list(color) if isinstance(color, (list, tuple)) else [color]
    titles = list(title) if isinstance(title, (list, tuple)) else [title] * len(keys)
    n_panels = len(keys)
    cols = min(ncols, n_panels)
    rows = int(np.ceil(n_panels / cols))
    width_px = round(figsize[0] * dpi)
    height_px = round(figsize[1] * dpi)
    label = basis.removeprefix("X_").upper()

    with plt.rc_context(_style()):
        # One panel keeps its full size and lets a long legend extend the saved figure;
        # several panels share the space under a constrained layout so no legend
        # overlaps the next panel.
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize[0] * cols, figsize[1] * rows),
            squeeze=False,
            layout="constrained" if n_panels > 1 else None,
        )
        extra: list[Any] = []
        flat_axes = list(axes.ravel())
        for ax in flat_axes[n_panels:]:
            ax.set_visible(False)
        for ax, key, heading in zip(flat_axes, keys, titles, strict=False):
            image, info = render_embedding(
                adata,
                basis,
                key,
                width=width_px,
                height=height_px,
                size=size,
                alpha=alpha,
                palette=palette,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                xlim=xlim,
                ylim=ylim,
                device=device,
            )
            (x0, x1), (y0, y1) = info["xlim"], info["ylim"]
            ax.imshow(image, extent=(x0, x1, y0, y1), origin="upper", interpolation="nearest")
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_title(heading if heading is not None else (key or label))
            if frameon:
                ax.set_xlabel(f"{label}1")
                ax.set_ylabel(f"{label}2")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
            else:
                _bare(ax)
            if info["kind"] == "categorical":
                levels, colours = info["levels"], info["colours"]
                if legend_loc == "on data":
                    cats = adata.obs[key].astype("category")
                    coords = np.asarray(adata.obsm[basis], dtype=float)[:, :2]
                    codes = cats.cat.codes.to_numpy()
                    for i, level in enumerate(levels):
                        inside = codes == i
                        if inside.any():
                            mx, my = np.median(coords[inside], axis=0)
                            ax.text(
                                mx,
                                my,
                                str(level),
                                fontsize=legend_fontsize,
                                ha="center",
                                va="center",
                                weight="bold",
                                path_effects=_halo(),
                            )
                elif legend_loc != "none":
                    from matplotlib.lines import Line2D

                    handles = [
                        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=6)
                        for c in colours
                    ]
                    # Up to 24 entries per column, so seventy cell types read as three
                    # columns; the axis title already names the column, so the legend
                    # carries no title of its own.
                    ncol_legend = max(1, int(np.ceil(len(levels) / 24)))
                    legend = ax.legend(
                        handles,
                        [str(level) for level in levels],
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        fontsize=legend_fontsize,
                        ncol=ncol_legend,
                        handletextpad=0.3,
                        borderaxespad=0.0,
                    )
                    extra.append(legend)
            elif info["kind"] == "continuous":
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                mappable = ScalarMappable(
                    norm=Normalize(info["vmin"], info["vmax"]), cmap=info["cmap"]
                )
                bar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
                bar.set_label(key, rotation=90)
                bar.outline.set_visible(False)
        result: Any = flat_axes[0] if n_panels == 1 else flat_axes[:n_panels]
    return _finish(fig, result, show, save, extra_artists=extra or None)


def _halo() -> list:
    """A white halo behind on-data labels so they read over any colour."""
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=2.5, foreground="white")]


def umap(adata: AnnData, color: str | Sequence[str] | None = None, **kwargs: Any) -> Any:
    """`embedding` on `obsm["X_umap"]`."""
    return embedding(adata, "X_umap", color, **kwargs)


def tsne(adata: AnnData, color: str | Sequence[str] | None = None, **kwargs: Any) -> Any:
    """`embedding` on `obsm["X_tsne"]`."""
    return embedding(adata, "X_tsne", color, **kwargs)


def pca(adata: AnnData, color: str | Sequence[str] | None = None, **kwargs: Any) -> Any:
    """`embedding` on `obsm["X_pca"]`, the first two components."""
    return embedding(adata, "X_pca", color, **kwargs)


def rank_genes_groups(
    adata: AnnData,
    n_genes: int = 10,
    n_cols: int = 4,
    *,
    show: bool | None = None,
    save: str | Path | None = None,
) -> np.ndarray | None:
    """Multi-panel bar chart of the top `n_genes` marker genes per group by score.

    Reads `adata.uns["rank_genes_groups"]`, written by `mc.tl.rank_genes_groups`.
    """
    record = adata.uns["rank_genes_groups"]
    names, scores = record["names"], record["scores"]
    groups = list(names.dtype.names)
    n_groups = len(groups)
    n_cols = max(1, min(n_cols, n_groups))
    n_rows = int(np.ceil(n_groups / n_cols))
    palette = _categorical_palette(n_groups, "husl")
    panel_height = max(2.6, 0.30 * n_genes + 0.9)  # taller panels when more genes are shown

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * panel_height), squeeze=False
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
        fig.suptitle("Top marker genes per group", fontsize=14, fontweight="bold", y=0.995)
        # Explicit padding so per-panel titles and gene labels never collide or crush.
        fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.93, bottom=0.06, left=0.09, right=0.97)
        result = axes
    return _finish(fig, result, show, save)
