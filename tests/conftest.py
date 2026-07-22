"""Reference data for the cross-checks: real PBMC 3k and a small synthetic stand-in.

The session fixtures prefixed with `_` build each stage once; the public fixtures hand
out copies, because every test mutates its AnnData in place.

Stages are prepared *with scanpy*. A single-step test must differ from its reference in
exactly one step, so everything up to the step under test comes from the reference
implementation.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse

from reference_metrics import component_correlations

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE = REPO_ROOT / ".cache" / "scanpy"
VENDORED_DATA = REPO_ROOT / "data"

N_COMPS = 50
N_NEIGHBORS = 15
TARGET_SUM = 1e4

# Embeddings are compared by neighbourhood preservation: how much of a cell's `K_REF`
# nearest neighbours in the reference layout stays within its `K_CAND` nearest in ours.
K_REF = 15
K_CAND = 30
# On real data UMAP does not agree with *itself* across seeds by more than ~45%, so the
# bar is set relative to that ceiling, measured per dataset in the test. See
# docs/API_CONTRACT.md; a flat 0.80 would be unreachable and would invite silent
# loosening. The strict 0.80 is kept where it is genuinely achievable: the `blobs`
# fixture, whose clusters are smaller than K_REF, so neighbour sets are seed independent.
CEILING_FRACTION = 0.85
STRICT_PRESERVATION = 0.80

DATASETS = {"synthetic": "_synthetic", "pbmc3k": "_pbmc3k_labelled"}


def configure_datasetdir() -> None:
    """Point scanpy at the repo cache; scanpy reads `datasetdir` but never creates it."""
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    sc.settings.datasetdir = DATASET_CACHE
    # The repo already ships the two files scanpy would download, under the exact names
    # it looks for. Seeding the cache from them keeps the suite runnable offline.
    for name in ("pbmc3k_raw.h5ad", "pbmc3k_processed.h5ad"):
        source, target = VENDORED_DATA / name, DATASET_CACHE / name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)


def _download(loader: Callable[[], AnnData]) -> AnnData:
    configure_datasetdir()
    try:
        return loader()
    except Exception as exc:  # any failure to obtain the data is a skip, not an error
        pytest.skip(f"PBMC 3k is unavailable ({type(exc).__name__}: {exc})")


@pytest.fixture(scope="session")
def _pbmc3k_labelled() -> AnnData:
    """Raw PBMC 3k counts for the cells the published analysis kept, with its cell types.

    The labels come from `pbmc3k_processed`, whose matrix is already scaled, so the two
    downloads have to be joined to get counts that carry a grouping for the DE test.
    """
    counts = _download(sc.datasets.pbmc3k)
    counts.var_names_make_unique()
    published = _download(sc.datasets.pbmc3k_processed)
    adata = counts[published.obs_names].copy()
    adata.obs["group"] = published.obs["louvain"].values
    return adata


@pytest.fixture(scope="session")
def _synthetic() -> AnnData:
    """240 cells in 3 groups over 300 genes, with marker genes, rare genes and a few
    near-empty cells so the filters and HVG have something real to find."""
    rng = np.random.default_rng(0)
    n_per_group, n_groups, n_genes, n_markers = 80, 3, 300, 20
    n_cells = n_per_group * n_groups
    groups = np.repeat(np.arange(n_groups), n_per_group)

    rates = np.full((n_cells, n_genes), 0.3)
    for g in range(n_groups):
        markers = slice(g * n_markers, (g + 1) * n_markers)
        rates[groups == g, markers] = 4.0
    rates[:, -30:] = 0.002  # rare genes, for filter_genes

    depth = rng.lognormal(mean=0.0, sigma=0.3, size=(n_cells, 1))
    depth[-12:] = 0.02  # low quality cells, for filter_cells
    counts = rng.poisson(rates * depth).astype(np.float32)

    adata = AnnData(sparse.csr_matrix(counts))
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.var_names = [f"gene{j}" for j in range(n_genes)]
    adata.obs["group"] = [f"group{g}" for g in groups]
    adata.obs["group"] = adata.obs["group"].astype("category")
    return adata


@pytest.fixture(scope="session")
def _blobs() -> AnnData:
    """Six tight, well separated clusters of 18 cells.

    Clusters smaller than `K_REF` make every cell's nearest neighbours its cluster mates
    whatever the layout does, which is what makes an absolute preservation threshold
    meaningful here and only here.
    """
    rng = np.random.default_rng(1)
    n_groups, per_group, n_genes, n_markers = 6, 18, 120, 20
    groups = np.repeat(np.arange(n_groups), per_group)

    rates = np.full((n_groups * per_group, n_genes), 0.1)
    for g in range(n_groups):
        rates[groups == g, g * n_markers : (g + 1) * n_markers] = 30.0
    counts = rng.poisson(rates).astype(np.float32)

    adata = AnnData(sparse.csr_matrix(counts))
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene{j}" for j in range(n_genes)]
    adata.obs["group"] = pd.Categorical([f"group{g}" for g in groups])

    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=30, random_state=0)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
    return adata


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("synthetic"),
        pytest.param("pbmc3k", marks=pytest.mark.reference),
    ],
)
def _counts(request: pytest.FixtureRequest) -> AnnData:
    adata: AnnData = request.getfixturevalue(DATASETS[request.param])
    adata.uns["dataset_id"] = request.param  # carried by every derived stage and copy
    return adata


@pytest.fixture(scope="session")
def _lognorm(_counts: AnnData) -> AnnData:
    adata = _counts.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)
    return adata


@pytest.fixture(scope="session")
def _scaled(_lognorm: AnnData) -> AnnData:
    adata = _lognorm.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes(adata))
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    return adata


@pytest.fixture(scope="session")
def _embedded(_scaled: AnnData) -> AnnData:
    adata = _scaled.copy()
    sc.pp.pca(adata, n_comps=N_COMPS, random_state=0)
    return adata


@pytest.fixture(scope="session")
def _neighbored(_embedded: AnnData) -> AnnData:
    adata = _embedded.copy()
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca", random_state=0)
    return adata


@pytest.fixture
def counts(_counts: AnnData) -> AnnData:
    """Raw counts with a `group` label column."""
    return _counts.copy()


@pytest.fixture
def lognorm(_lognorm: AnnData) -> AnnData:
    """Counts filtered, normalised to 1e4 and log1p'd by scanpy."""
    return _lognorm.copy()


@pytest.fixture
def scaled(_scaled: AnnData) -> AnnData:
    """`lognorm` restricted to scanpy's HVGs and scaled: the input to PCA."""
    return _scaled.copy()


@pytest.fixture
def embedded(_embedded: AnnData) -> AnnData:
    """`scaled` with scanpy's `X_pca`: the input to neighbors and t-SNE."""
    return _embedded.copy()


@pytest.fixture
def neighbored(_neighbored: AnnData) -> AnnData:
    """`embedded` with scanpy's neighbour graph: the input to UMAP."""
    return _neighbored.copy()


@pytest.fixture
def blobs(_blobs: AnnData) -> AnnData:
    """Well separated clusters, already PCA'd and neighboured: the one input where an
    absolute neighbourhood preservation threshold is achievable."""
    return _blobs.copy()


@pytest.fixture
def pbmc3k(_pbmc3k_labelled: AnnData) -> AnnData:
    """Real PBMC 3k counts, for the whole-pipeline test."""
    return _pbmc3k_labelled.copy()


def n_top_genes(adata: AnnData) -> int:
    """The tutorial's 2000 HVGs, or half the genes when there are fewer."""
    return min(2000, adata.n_vars // 2)


def check_pca_agreement(
    scaled_input: AnnData,
    ours: AnnData,
    reference: AnnData,
    *,
    label: str,
    record: Callable[[str, object], None],
) -> None:
    """`abs(corr)` >= 0.99 per component and variance ratios to rtol 1e-3 — on the
    components where a randomised SVD can deliver that.

    Trailing components sit in a nearly degenerate part of the spectrum and are not
    determined by the data. scanpy's default solver is the deterministic arpack; its
    *randomised* solver is the same algorithm class as ours, and on PBMC 3k it reproduces
    arpack for only the first 7 of 50 components. So the contract is asserted where the
    reference implementation is itself reproducible, and beyond that we only require not
    being meaningfully worse than it. Both counts are recorded either way.
    """
    reference_pcs = np.asarray(reference.obsm["X_pca"], dtype=np.float64)
    n_comps = reference_pcs.shape[1]
    ceiling_run = scaled_input.copy()
    sc.pp.pca(
        ceiling_run, n_comps=n_comps, zero_center=True, random_state=0, svd_solver="randomized"
    )

    ours_corr = component_correlations(np.asarray(ours.obsm["X_pca"]), reference_pcs)
    ceiling_corr = component_correlations(np.asarray(ceiling_run.obsm["X_pca"]), reference_pcs)
    determined = ceiling_corr >= 0.99

    record(f"pca.{label}.determined_components", int(determined.sum()))
    record(f"pca.{label}.components_over_0.99", int((ours_corr >= 0.99).sum()))
    print(
        f"\npca on {label}: {int(determined.sum())}/{n_comps} components are determined "
        f"(a randomised SVD reproduces arpack there), we match {int((ours_corr >= 0.99).sum())}"
    )

    bad = np.flatnonzero(determined & (ours_corr < 0.99))
    assert bad.size == 0, (
        f"components {bad.tolist()} correlate {ours_corr[bad].round(4).tolist()} < 0.99 "
        f"although a randomised SVD reaches {ceiling_corr[bad].round(4).tolist()} there"
    )
    weak = np.flatnonzero(~determined & (ours_corr < CEILING_FRACTION * ceiling_corr))
    assert weak.size == 0, (
        f"undetermined components {weak.tolist()} correlate {ours_corr[weak].round(4).tolist()}, "
        f"below {CEILING_FRACTION:.0%} of the {ceiling_corr[weak].round(4).tolist()} a randomised "
        f"SVD reaches"
    )
    assert_allclose(
        np.asarray(ours.uns["pca"]["variance_ratio"], dtype=np.float64)[determined],
        np.asarray(reference.uns["pca"]["variance_ratio"], dtype=np.float64)[determined],
        rtol=1e-3,
        err_msg="variance ratios of the determined components",
    )
