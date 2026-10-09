"""UMAP and t-SNE embeddings. Owned by feat/python-tl.

Like `metalcyte.pp`, this module only moves data. The `AnnData` conventions it needs
are private helpers in `metalcyte.pp`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from metalcyte._shared import (
    _VALUE_DTYPE,
    _csr_args,
    _extension,
    _neighbor_graph,
    _representation,
    _resolve_device,
)

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["tsne", "umap"]

# Defaults that the core requires and the Python signatures do not expose.
_DEFAULT_EPOCHS = 200
_UMAP_LEARNING_RATE = 1.0
_UMAP_NEGATIVE_SAMPLE_RATE = 5
_TSNE_COMPONENTS = 2
_TSNE_ITERATIONS = 1000
# The automatic t-SNE learning rate is n_obs / early_exaggeration / 4, with this floor.
# A fixed rate of 1000 is far too large for small datasets.
_MINIMUM_LEARNING_RATE = 50.0


def _automatic_learning_rate(n_obs: int, early_exaggeration: float) -> float:
    return max(n_obs / early_exaggeration / 4.0, _MINIMUM_LEARNING_RATE)


# The core labels cells with unsigned indices; exclusion is expressed by omission.
_LABEL_DTYPE = np.uint32


def umap(
    adata: AnnData,
    *,
    n_components: int = 2,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_epochs: int | None = None,
    random_state: int = 0,
    device: str | None = None,
    parallel: bool = False,
) -> None:
    """Lay the neighbour graph out with UMAP, writing `obsm["X_umap"]`.

    `parallel=True` runs the layout optimisation with lock-free "Hogwild" SGD, as
    umap-learn's `parallel=True`: on the GPU when the device is Metal (`"auto"` with a
    usable GPU, or `"metal"`), otherwise on every CPU core. It is several times faster on
    large graphs, about nine times on the GPU at a million cells, but the layout is no
    longer reproducible from `random_state` alone, so the default stays sequential and
    deterministic.
    """
    device = _resolve_device(device)
    graph = _neighbor_graph(adata)
    extension = _extension()
    epochs = _DEFAULT_EPOCHS if n_epochs is None else n_epochs
    if parallel and device != "cpu" and hasattr(extension, "umap_metal"):
        from metalcyte import gpu_available

        if gpu_available():
            embedding = extension.umap_metal(
                *_csr_args(graph),
                n_components,
                epochs,
                min_dist,
                spread,
                _UMAP_LEARNING_RATE,
                _UMAP_NEGATIVE_SAMPLE_RATE,
                random_state,
            )
            adata.obsm["X_umap"] = np.asarray(embedding, dtype=_VALUE_DTYPE)
            return
    if parallel and hasattr(extension, "umap_parallel"):
        embedding = extension.umap_parallel(
            *_csr_args(graph),
            n_components,
            epochs,
            min_dist,
            spread,
            _UMAP_LEARNING_RATE,
            _UMAP_NEGATIVE_SAMPLE_RATE,
            random_state,
        )
        adata.obsm["X_umap"] = np.asarray(embedding, dtype=_VALUE_DTYPE)
        return
    embedding = extension.umap(
        *_csr_args(graph),
        n_components,
        epochs,
        min_dist,
        spread,
        _UMAP_LEARNING_RATE,
        _UMAP_NEGATIVE_SAMPLE_RATE,
        random_state,
        device,
    )
    adata.obsm["X_umap"] = np.asarray(embedding, dtype=_VALUE_DTYPE)


def tsne(
    adata: AnnData,
    *,
    n_pcs: int = 50,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float | None = None,
    method: str = "auto",
    random_state: int = 0,
    device: str | None = None,
) -> None:
    """Lay the principal components out with t-SNE, writing `obsm["X_tsne"]`.

    `method` is `"exact"`, `"fft"` or `"auto"`. The exact formulation holds the
    `(n, n)` affinity matrix and accepts at most 20 000 cells. `"fft"` is FFT-accelerated
    interpolation-based t-SNE (FIt-SNE): sparse affinities over the `3 * perplexity`
    nearest neighbours and the repulsive term by interpolation on a grid, which scales to
    a million cells. `"auto"` picks the exact formulation up to 20 000 cells and the FFT
    one above.
    """
    device = _resolve_device(device)
    if method not in ("auto", "exact", "fft"):
        raise ValueError(f"method must be 'auto', 'exact' or 'fft', got {method!r}")
    embedding = _representation(adata, "X_pca")[:, :n_pcs]
    if learning_rate is None:
        learning_rate = _automatic_learning_rate(embedding.shape[0], early_exaggeration)
    result = _extension().tsne(
        np.ascontiguousarray(embedding),
        _TSNE_COMPONENTS,
        perplexity,
        early_exaggeration,
        learning_rate,
        _TSNE_ITERATIONS,
        random_state,
        device,
        method,
    )
    adata.obsm["X_tsne"] = np.asarray(result, dtype=_VALUE_DTYPE)
