"""Cross-checks for `pp.regress_out` and `pp.combat` against scanpy.

Both are deterministic transforms of the whole matrix, so the criterion is
element-wise agreement. The measured deviations are printed rather than only
asserted: the number is the result, the threshold is only the bar.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose
from scipy import sparse

from reference_metrics import as_dense
from scrust_call import scrust_call

# scanpy accumulates the residual in f32 as we do, but forms the coefficients in
# f64; a relative 1e-4 is what that difference leaves on log-normalised data.
REGRESS_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
# combat runs the whole standardise/shrink/adjust chain in f32 where scanpy is
# promoted to f64 by pandas, so the bar is one digit looser.
COMBAT_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}

# The two continuous covariates a real workflow regresses out.
NUMERIC_KEYS = ["n_counts", "percent_mito"]


def _with_covariates(adata: AnnData) -> AnnData:
    """Two numeric obs columns of the kind a real workflow regresses out."""
    counts = np.asarray(as_dense(adata.X).sum(axis=1)).ravel()
    rng = np.random.default_rng(0)
    adata.obs["n_counts"] = counts.astype(np.float64)
    adata.obs["percent_mito"] = rng.uniform(0.01, 0.2, size=adata.n_obs)
    return adata


def _batched(adata: AnnData, *, shift: float = 1.5, scale: float = 1.6) -> AnnData:
    """Plant an additive and a multiplicative batch effect on half the cells."""
    adata = adata.copy()
    labels = np.where(np.arange(adata.n_obs) % 2 == 0, "b0", "b1")
    adata.obs["batch"] = pd.Categorical(labels)
    dense = np.asarray(as_dense(adata.X), dtype=np.float32)
    affected = labels == "b1"
    dense[affected] = dense[affected] * scale + shift
    adata.X = dense
    return adata


def _deviation(ours: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Largest absolute deviation, and the largest relative one where the
    reference is big enough for a ratio to mean anything."""
    absolute = np.abs(ours - reference)
    scale = np.abs(reference)
    relative = np.where(scale > 1e-3, absolute / np.maximum(scale, 1e-12), 0.0)
    return float(absolute.max()), float(relative.max())


def test_regress_out_matches_scanpy(lognorm: AnnData) -> None:
    adata = _with_covariates(lognorm)
    ours = adata.copy()
    scrust_call("pp.regress_out", ours, NUMERIC_KEYS)

    reference = adata.copy()
    sc.pp.regress_out(reference, NUMERIC_KEYS)

    largest, relative = _deviation(as_dense(ours.X), as_dense(reference.X))
    print(
        f"\nregress_out on {adata.uns['dataset_id']} "
        f"({adata.n_obs} cells x {adata.n_vars} genes): "
        f"largest deviation {largest:.3e}, largest relative {relative:.3e}"
    )
    assert_allclose(as_dense(ours.X), as_dense(reference.X), **REGRESS_TOLERANCE)


def test_regress_out_matches_scanpy_on_a_categorical_key(lognorm: AnnData) -> None:
    """scanpy regresses a categorical on the per-category gene mean; we one-hot
    encode it. The two designs span the same space, so the residuals must be the
    same to f32 — and that equivalence is exactly what is being asserted."""
    ours = lognorm.copy()
    scrust_call("pp.regress_out", ours, "group")

    reference = lognorm.copy()
    sc.pp.regress_out(reference, "group")

    largest, relative = _deviation(as_dense(ours.X), as_dense(reference.X))
    # The relative figure is reported but not asserted on: a residual whose true
    # value is a rounding error away from zero has no meaningful ratio, and this
    # design leaves plenty of those. The absolute deviation is the result.
    print(
        f"\nregress_out on a categorical key: largest deviation {largest:.3e}, "
        f"largest relative {relative:.3e}"
    )
    assert_allclose(as_dense(ours.X), as_dense(reference.X), rtol=1e-4, atol=1e-4)


def test_a_gene_that_is_a_linear_function_of_the_covariate_regresses_to_zero() -> None:
    """The one case with an exactly known answer: if a gene is an affine
    function of the covariates, nothing is left of it."""
    rng = np.random.default_rng(3)
    covariate = rng.normal(size=200)
    exact = np.column_stack([2.0 + 3.0 * covariate, -1.0 + 0.5 * covariate])
    noise = rng.normal(size=(200, 2))
    adata = AnnData(np.column_stack([exact, noise]).astype(np.float32))
    adata.obs["covariate"] = covariate

    scrust_call("pp.regress_out", adata, "covariate")

    residuals = as_dense(adata.X)
    assert np.abs(residuals[:, :2]).max() < 1e-4, "an exact linear gene left a residual"
    # The noise genes are untouched apart from their own (tiny) fit.
    assert np.abs(residuals[:, 2:]).max() > 1.0


def test_regress_out_rejects_a_rank_deficient_design() -> None:
    rng = np.random.default_rng(4)
    adata = AnnData(rng.normal(size=(50, 6)).astype(np.float32))
    adata.obs["first"] = rng.normal(size=50)
    adata.obs["copy"] = adata.obs["first"]

    with pytest.raises(ValueError, match="full column rank"):
        scrust_call("pp.regress_out", adata, ["first", "copy"])


def test_regress_out_rejects_an_unknown_key() -> None:
    adata = AnnData(np.ones((10, 3), dtype=np.float32))
    with pytest.raises(KeyError, match="absent"):
        scrust_call("pp.regress_out", adata, "absent")


def test_combat_rejects_mismatched_and_impossible_input() -> None:
    rng = np.random.default_rng(5)
    adata = AnnData(rng.normal(size=(20, 4)).astype(np.float32))
    adata.obs["batch"] = pd.Categorical(["a"] * 19 + ["b"])

    # One cell in a batch: no within-batch variance to estimate.
    with pytest.raises(ValueError, match="at least 2 cells"):
        scrust_call("pp.combat", adata, "batch")
    with pytest.raises(ValueError, match="could not find the key"):
        scrust_call("pp.combat", adata, "absent")


def test_combat_matches_scanpy(lognorm: AnnData) -> None:
    adata = _batched(lognorm)
    ours = adata.copy()
    scrust_call("pp.combat", ours, "batch")

    reference = adata.copy()
    sc.pp.combat(reference, "batch")

    largest, relative = _deviation(as_dense(ours.X), as_dense(reference.X))
    print(
        f"\ncombat on {adata.uns['dataset_id']}: "
        f"largest deviation {largest:.3e}, largest relative {relative:.3e}"
    )
    assert_allclose(as_dense(ours.X), as_dense(reference.X), **COMBAT_TOLERANCE)


def test_combat_shrinks_the_batch_difference(lognorm: AnnData) -> None:
    """The scientific check: the gap between the batches' gene means, which the
    fixture plants, must be far smaller after the correction.

    Averaged over genes it collapses by more than an order of magnitude. The
    worst single gene does not, and cannot: the empirical Bayes step deliberately
    shrinks each gene's estimate towards the prior its neighbours share, so a
    gene whose batch effect is unlike the rest keeps part of it. That residual is
    ComBat's, not ours, so it is measured against scanpy on the same input rather
    than against a number chosen here.
    """
    adata = _batched(lognorm)
    before = as_dense(adata.X)
    ours = adata.copy()
    scrust_call("pp.combat", ours, "batch")
    reference = adata.copy()
    sc.pp.combat(reference, "batch")

    labels = np.asarray(adata.obs["batch"] == "b1")

    def gaps(matrix: np.ndarray) -> np.ndarray:
        return np.abs(matrix[labels].mean(axis=0) - matrix[~labels].mean(axis=0))

    corrected, ceiling = as_dense(ours.X), as_dense(reference.X)
    print(
        f"\ncombat batch mean gap on {adata.uns['dataset_id']}: "
        f"mean over genes {gaps(before).mean():.4f} -> {gaps(corrected).mean():.4f}, "
        f"worst gene {gaps(before).max():.4f} -> {gaps(corrected).max():.4f} "
        f"(scanpy leaves {gaps(ceiling).max():.4f})"
    )
    assert gaps(corrected).mean() < gaps(before).mean() / 10.0
    assert gaps(corrected).max() <= gaps(ceiling).max() * 1.01


def test_combat_with_covariates_matches_scanpy(lognorm: AnnData) -> None:
    adata = _with_covariates(_batched(lognorm))
    ours = adata.copy()
    scrust_call("pp.combat", ours, "batch", covariates=["percent_mito"])

    reference = adata.copy()
    sc.pp.combat(reference, "batch", covariates=["percent_mito"])

    largest, _ = _deviation(as_dense(ours.X), as_dense(reference.X))
    print(f"\ncombat with a covariate: largest deviation {largest:.3e}")
    assert_allclose(as_dense(ours.X), as_dense(reference.X), **COMBAT_TOLERANCE)


@pytest.mark.parametrize("call", ["regress_out", "combat"])
def test_cpu_and_gpu_agree(lognorm: AnnData, call: str) -> None:
    import scrust

    if not scrust.gpu_available():
        pytest.skip("no Metal device on this machine")

    adata = _with_covariates(_batched(lognorm))
    arguments = {"regress_out": (NUMERIC_KEYS,), "combat": ("batch",)}[call]
    on_cpu, on_gpu = adata.copy(), adata.copy()
    scrust_call(f"pp.{call}", on_cpu, *arguments, device="cpu")
    scrust_call(f"pp.{call}", on_gpu, *arguments, device="gpu")

    largest, _ = _deviation(as_dense(on_gpu.X), as_dense(on_cpu.X))
    print(f"\n{call}: largest cpu/gpu deviation {largest:.3e}")
    assert_allclose(as_dense(on_gpu.X), as_dense(on_cpu.X), rtol=1e-3, atol=1e-4)


def _benchmark_fixture(n_obs: int, n_vars: int) -> AnnData:
    rng = np.random.default_rng(6)
    adata = AnnData(sparse.csr_matrix(rng.poisson(0.4, size=(n_obs, n_vars)).astype(np.float32)))
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
    adata.obs["percent_mito"] = rng.uniform(0.01, 0.2, size=n_obs)
    adata.obs["donor"] = pd.Categorical(rng.integers(0, 4, size=n_obs).astype(str))
    return adata


def _timed(call, adata: AnnData, *args, **kwargs) -> tuple[AnnData, float]:
    """Run `call` on a fresh copy and return the copy and the elapsed seconds."""
    target = adata.copy()
    started = time.perf_counter()
    call(target, *args, **kwargs)
    return target, time.perf_counter() - started


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_regress_out_beats_scanpys_per_gene_loop(device: str) -> None:
    """2 000 cells by 2 000 genes, the size the branch report quotes.

    scanpy 1.12 has two paths. On a *categorical* key it still fits one
    statsmodels GLM per gene, which is the step this branch exists to replace. On
    numeric covariates it takes a shortcut that is our own algorithm — normal
    equations once, one matmul for every gene — in f64 BLAS, so there the honest
    result is parity, not a speedup, and only the categorical case is asserted on.
    """
    import scrust

    if device == "gpu" and not scrust.gpu_available():
        pytest.skip("no Metal device on this machine")

    n_obs, n_vars = 2000, 2000
    adata = _benchmark_fixture(n_obs, n_vars)
    header = f"\nregress_out {n_obs}x{n_vars} on {device}"

    for keys, path in [("donor", "categorical, scanpy's per-gene GLM"), (NUMERIC_KEYS, "numeric")]:
        scrust_call("pp.regress_out", adata.copy(), keys, device=device)  # warm the device
        ours, mine = _timed(scrust.pp.regress_out, adata, keys, device=device)
        reference, theirs = _timed(sc.pp.regress_out, adata, keys)

        print(
            f"{header}, {path}: scrust {mine * 1e3:.0f} ms, "
            f"scanpy {theirs * 1e3:.0f} ms ({theirs / mine:.1f}x)"
        )
        assert_allclose(as_dense(ours.X), as_dense(reference.X), **REGRESS_TOLERANCE)
        if keys == "donor":
            assert mine * 5.0 < theirs, "the batched solve must rout a per-gene loop"
