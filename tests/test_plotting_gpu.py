"""The GPU-rendered embedding plots: the renderer's image, its agreement with the
renderer on the cores, and the matplotlib figures around it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import metalcyte as mc

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")


@pytest.fixture
def cells() -> AnnData:
    rng = np.random.default_rng(0)
    n = 6000
    centres = rng.normal(0, 10, (6, 2))
    labels = rng.integers(0, 6, n)
    xy = centres[labels] + rng.normal(0, 1, (n, 2))
    adata = AnnData(
        X=rng.poisson(1.0, (n, 3)).astype(np.float32),
        obs=pd.DataFrame(
            {"group": pd.Categorical(labels.astype(str)), "score": rng.random(n)},
            index=[f"c{i}" for i in range(n)],
        ),
    )
    adata.var_names = ["g0", "g1", "g2"]
    adata.obsm["X_umap"] = xy.astype(np.float32)
    return adata


def test_render_embedding_returns_an_rgba_image(cells: AnnData) -> None:
    image, info = mc.pl.render_embedding(cells, "X_umap", "group", width=300, height=200)
    assert image.shape == (200, 300, 4) and image.dtype == np.uint8
    assert info["kind"] == "categorical" and len(info["levels"]) == 6
    assert (image[..., :3] < 250).any(), "nothing was drawn"
    assert (image[..., :3] == 255).all(axis=-1).any(), "no background left"


def test_gpu_and_cpu_renderers_agree(cells: AnnData) -> None:
    gpu, _ = mc.pl.render_embedding(
        cells, "X_umap", "group", width=300, height=200, size=4, device="metal"
    )
    cpu, _ = mc.pl.render_embedding(
        cells, "X_umap", "group", width=300, height=200, size=4, device="cpu"
    )
    difference = np.abs(gpu.astype(np.int32) - cpu.astype(np.int32))
    # Without a GPU the Metal request falls back to the cores and the images are equal;
    # with one, the two rasterisers differ only by rounding at disc edges.
    assert difference.mean() < 1.0, f"mean channel difference {difference.mean():.2f}"
    assert (difference.max(axis=-1) > 64).mean() < 0.001


def test_continuous_colouring_follows_the_values(cells: AnnData) -> None:
    cells.obs["score"] = np.linspace(0.0, 1.0, cells.n_obs)
    image, info = mc.pl.render_embedding(
        cells, "X_umap", "score", width=200, height=200, cmap="viridis"
    )
    assert info["kind"] == "continuous" and info["vmin"] == 0.0 and info["vmax"] == 1.0
    image, _ = mc.pl.render_embedding(cells, "X_umap", "g1", width=100, height=100)
    assert image.shape == (100, 100, 4)


def test_umap_figures_save(cells: AnnData, tmp_path) -> None:
    ax = mc.pl.umap(cells, color="group", show=False, save=tmp_path / "cat.png")
    assert ax is not None and (tmp_path / "cat.png").stat().st_size > 1000
    axes = mc.pl.umap(
        cells,
        color=["group", "score"],
        legend_loc="on data",
        ncols=2,
        show=False,
        save=tmp_path / "two.png",
    )
    assert len(axes) == 2 and (tmp_path / "two.png").exists()
    ax = mc.pl.umap(cells, show=False, save=tmp_path / "plain.pdf", frameon=True)
    assert (tmp_path / "plain.pdf").exists()


def test_zoom_window_and_stored_colours(cells: AnnData) -> None:
    cells.uns["group_colors"] = ["#ff0000", "#00ff00", "#0000ff", "#000000", "#ffff00", "#00ffff"]
    _image, info = mc.pl.render_embedding(
        cells, "X_umap", "group", width=120, height=100, xlim=(-5, 5), ylim=(-5, 5)
    )
    assert info["xlim"] == (-5, 5) and info["ylim"] == (-5, 5)
    assert info["colours"][0] == (1.0, 0.0, 0.0)


def test_unknown_colour_key_raises(cells: AnnData) -> None:
    with pytest.raises(KeyError):
        mc.pl.render_embedding(cells, "X_umap", "no_such_key")
