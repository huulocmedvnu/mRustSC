"""Cross-check `regress_out`, `combat`, Moran's I and Geary's C against scanpy.

`regress_out` and `combat` rewrite every value in the matrix, so an error in either is
an error in everything computed afterwards, and neither produces an obviously wrong
number when it is wrong -- a residual is still a residual. The two autocorrelation
statistics are read directly by a user, so an error there is read directly too.

References are scanpy driven on the same input. Where a test pins a divergence rather
than an agreement, its docstring says so and says why.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from scrust_call import DEVICE, scrust_call


def csr_args(matrix: sparse.csr_matrix):
    matrix = matrix.tocsr()
    return (
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
    )


def scrust_regress_out(matrix, covariates):
    return np.asarray(
        scrust_call(
            "_scrust.regress_out",
            *csr_args(matrix),
            np.asarray(covariates, dtype=np.float32),
            DEVICE,
        )
    )


def scrust_combat(matrix, batch, n_batches, covariates=None):
    return np.asarray(
        scrust_call(
            "_scrust.combat",
            *csr_args(matrix),
            np.asarray(batch, dtype=np.uint32),
            n_batches,
            None if covariates is None else np.asarray(covariates, dtype=np.float32),
            DEVICE,
        )
    )


def scrust_autocorrelation(name, graph, matrix):
    return np.asarray(
        scrust_call(
            f"_scrust.{name}",
            graph.indptr.astype(np.uint32),
            graph.indices.astype(np.uint32),
            graph.data.astype(np.float32),
            graph.shape[0],
            *csr_args(matrix),
            DEVICE,
        )
    )


def expression(n_cells=150, n_genes=40, seed=0, sparsity=0.4):
    """Log-scale data, which is what both of these are applied to in practice."""
    rng = np.random.default_rng(seed)
    dense = rng.lognormal(0.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
    dense[rng.random(dense.shape) < sparsity] = 0.0
    return sparse.csr_matrix(np.log1p(dense).astype(np.float32))


def knn_graph(n_cells, seed=1, k=8):
    """A symmetric weighted graph of the shape `pp.neighbors` leaves in `obsp`."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_cells, 5))
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(distances, np.inf)
    dense = np.zeros((n_cells, n_cells), dtype=np.float32)
    for row in range(n_cells):
        for column in np.argsort(distances[row])[:k]:
            weight = np.float32(1.0 / (1.0 + distances[row, column]))
            dense[row, column] = dense[column, row] = weight
    return sparse.csr_matrix(dense)


# --------------------------------------------------------------------------------
# 1. regress_out
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("n_covariates", [1, 2, 3])
def test_regress_out_matches_scanpy(n_covariates):
    """Ordinary least squares per gene, on a design with an intercept."""
    matrix = expression(seed=3)
    rng = np.random.default_rng(11)
    covariates = rng.normal(size=(matrix.shape[0], n_covariates)).astype(np.float32)

    adata = AnnData(np.asarray(matrix.todense(), dtype=np.float32))
    keys = []
    for i in range(n_covariates):
        adata.obs[f"c{i}"] = covariates[:, i].astype(np.float64)
        keys.append(f"c{i}")
    sc.pp.regress_out(adata, keys)

    ours = scrust_regress_out(matrix, covariates)
    np.testing.assert_allclose(ours, np.asarray(adata.X), rtol=1e-3, atol=1e-3)


def test_regress_out_leaves_a_residual_orthogonal_to_the_design():
    """The property that defines a least-squares residual, and one scanpy cannot be
    wrong about in a way this test would share: the residual has to be orthogonal to
    every covariate and to the intercept.

    Checked independently of scanpy so that a shared misunderstanding of the design
    matrix cannot pass unnoticed.
    """
    matrix = expression(n_cells=120, n_genes=25, seed=5)
    rng = np.random.default_rng(13)
    covariates = rng.normal(size=(matrix.shape[0], 2)).astype(np.float32)

    residuals = scrust_regress_out(matrix, covariates)
    design = np.column_stack([np.ones(matrix.shape[0], dtype=np.float32), covariates])
    projected = design.T @ residuals
    scale = np.abs(residuals).max()
    np.testing.assert_allclose(projected, 0.0, atol=2e-3 * max(scale, 1.0))


def test_regress_out_refuses_a_rank_deficient_design_as_scanpy_does():
    """A covariate that never varies is the intercept again, so the design is singular
    and there is no unique fit. Both refuse rather than return one.

    scanpy fails inside the solve, with `LinAlgError: Singular matrix`. The core checks
    the design first and names the offending column. Same outcome, and refusing is the
    right one -- the alternative is `lstsq`'s minimum-norm solution, which is a
    particular answer to a question that has infinitely many, handed back without
    saying so.
    """
    matrix = expression(n_cells=80, n_genes=15, seed=7)
    constant = np.full((matrix.shape[0], 1), 2.5, dtype=np.float32)

    with pytest.raises(ValueError, match="full column rank"):
        scrust_regress_out(matrix, constant)

    adata = AnnData(np.asarray(matrix.todense(), dtype=np.float32))
    adata.obs["c"] = np.full(matrix.shape[0], 2.5)
    with pytest.raises(np.linalg.LinAlgError):
        sc.pp.regress_out(adata, ["c"])


def test_regress_out_refuses_a_duplicated_covariate_too():
    """The same design defect, arrived at differently: two covariates that are copies
    of each other. This is the shape it takes in practice -- the same column passed
    twice under two names -- and it has to be caught for the same reason.
    """
    matrix = expression(n_cells=80, n_genes=15, seed=9)
    rng = np.random.default_rng(3)
    column = rng.normal(size=(matrix.shape[0], 1)).astype(np.float32)
    duplicated = np.hstack([column, column])

    with pytest.raises(ValueError, match="full column rank"):
        scrust_regress_out(matrix, duplicated)


# --------------------------------------------------------------------------------
# 2. combat
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("n_batches", [2, 3])
def test_combat_matches_scanpy(n_batches):
    """Empirical-Bayes batch correction, against `sc.pp.combat` on the same batches."""
    matrix = expression(n_cells=180, n_genes=30, seed=17, sparsity=0.0)
    rng = np.random.default_rng(19)
    batch = rng.integers(0, n_batches, size=matrix.shape[0]).astype(np.uint32)
    # A real batch effect, so the correction has something to remove.
    dense = np.asarray(matrix.todense(), dtype=np.float32)
    for b in range(n_batches):
        dense[batch == b] += np.float32(0.5 * b)
    matrix = sparse.csr_matrix(dense)

    adata = AnnData(dense.copy())
    adata.obs["batch"] = [str(b) for b in batch]
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    theirs = sc.pp.combat(adata, key="batch", inplace=False)

    ours = scrust_combat(matrix, batch, n_batches)
    np.testing.assert_allclose(ours, theirs, rtol=1e-3, atol=1e-3)


def test_combat_shrinks_the_between_batch_spread_it_was_given():
    """Independent of scanpy: whatever the shrinkage does in detail, a per-batch offset
    that was put in has to come out smaller than it went in."""
    rng = np.random.default_rng(23)
    dense = rng.normal(0.0, 1.0, size=(200, 20)).astype(np.float32)
    batch = (np.arange(200) % 2).astype(np.uint32)
    dense[batch == 1] += np.float32(3.0)
    matrix = sparse.csr_matrix(dense)

    def spread(x):
        return float(np.abs(x[batch == 0].mean(0) - x[batch == 1].mean(0)).mean())

    before = spread(dense)
    after = spread(scrust_combat(matrix, batch, 2))
    assert before > 2.5, "the fixture stopped carrying a batch effect"
    assert after < before / 10.0, f"batch effect {before:.3g} only came down to {after:.3g}"


# --------------------------------------------------------------------------------
# 3. Moran's I and Geary's C
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["morans_i", "gearys_c"])
def test_autocorrelation_matches_scanpy(name):
    n_cells = 120
    graph = knn_graph(n_cells, seed=2)
    matrix = expression(n_cells=n_cells, n_genes=25, seed=29)

    ours = scrust_autocorrelation(name, graph, matrix)
    reference = getattr(sc.metrics, name)(graph, matrix.T.tocsr())
    np.testing.assert_allclose(ours, np.asarray(reference), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["morans_i", "gearys_c"])
def test_a_gene_that_never_varies_has_no_autocorrelation_to_report(name):
    """Both statistics divide by the variance of the gene across cells, so a constant
    gene is 0/0. scanpy yields NaN; the core has to agree rather than report a 0, which
    would read as "measured, and there is none"."""
    n_cells = 100
    graph = knn_graph(n_cells, seed=4)
    dense = np.asarray(expression(n_cells=n_cells, n_genes=10, seed=31).todense())
    dense[:, 0] = 0.0
    dense[:, 1] = 2.0
    matrix = sparse.csr_matrix(dense.astype(np.float32))

    ours = scrust_autocorrelation(name, graph, matrix)
    theirs = np.asarray(getattr(sc.metrics, name)(graph, matrix.T.tocsr()))
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
    both = np.isfinite(ours) & np.isfinite(theirs)
    np.testing.assert_allclose(ours[both], theirs[both], rtol=1e-4, atol=1e-5)


def test_a_gene_perfectly_aligned_with_the_graph_scores_near_one():
    """A sanity bound that does not depend on scanpy: Moran's I is near its maximum
    when neighbouring cells share a value, and Geary's C is near its minimum of 0.
    Two well-separated blocks, with the graph joining cells within a block only.
    """
    n_cells = 80
    half = n_cells // 2
    dense_graph = np.zeros((n_cells, n_cells), dtype=np.float32)
    for block in (slice(0, half), slice(half, n_cells)):
        dense_graph[block, block] = 1.0
    np.fill_diagonal(dense_graph, 0.0)
    graph = sparse.csr_matrix(dense_graph)

    values = np.zeros((n_cells, 1), dtype=np.float32)
    values[half:, 0] = 1.0
    matrix = sparse.csr_matrix(values)

    morans = scrust_autocorrelation("morans_i", graph, matrix)[0]
    gearys = scrust_autocorrelation("gearys_c", graph, matrix)[0]
    assert morans > 0.9, f"Moran's I should be near 1 for a graph-aligned gene, got {morans}"
    assert gearys < 0.1, f"Geary's C should be near 0 for a graph-aligned gene, got {gearys}"
