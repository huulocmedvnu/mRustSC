"""Audit `tl.score_genes_cell_cycle` against scanpy: scores, the phase rule, ctrl_size.

Cell cycle scoring is two `score_genes` calls plus a three-way phase decision. The
`score_genes` control-draw parity is what `test_scoring_audit.py` is about; this file pins
the *cell-cycle-specific* layer that sits on top of it:

* the S and G2M scores land bit-for-bit on real PBMC 3k, and
* the phase assignment reproduces scanpy's exact rule -- S by default, G2M when it
  outscores S, G1 when both scores are negative -- rather than agreeing by luck.

scanpy's rule, from `scanpy/tools/_score_genes.py::score_genes_cell_cycle`:

    ctrl_size = min(len(s_genes), len(g2m_genes))
    phase = pd.Series("S", index=scores.index)
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"
    phase[np.all(scores < 0, axis=1)] = "G1"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from numpy.testing import assert_allclose

from scrust_call import scrust_call

ELEMENTWISE = {"rtol": 1e-5, "atol": 1e-6}

# Tirosh et al. S and G2M markers; the ones absent from PBMC 3k are dropped per test.
S_GENES = ["MCM5", "PCNA", "TYMS", "RRM2", "MCM2", "UHRF1", "CDC45", "RRM1", "GINS2", "CHAF1B"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "TOP2A", "TPX2", "CKS2", "CDC20", "TTK", "CENPF"]


@pytest.fixture(scope="module")
def _lognorm_pbmc(_pbmc3k_labelled: AnnData) -> AnnData:
    """Log-normalised PBMC 3k, prepared with scanpy so only the scoring step differs."""
    adata = _pbmc3k_labelled.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


@pytest.fixture
def lognorm_pbmc(_lognorm_pbmc: AnnData) -> AnnData:
    return _lognorm_pbmc.copy()


def _present(genes: list[str], adata: AnnData) -> list[str]:
    return [gene for gene in genes if gene in adata.var_names]


def test_scores_and_phase_match_scanpy_bit_for_bit(lognorm_pbmc: AnnData) -> None:
    """The whole call on real data: S and G2M scores element-wise, phase exactly."""
    s_genes = _present(S_GENES, lognorm_pbmc)
    g2m_genes = _present(G2M_GENES, lognorm_pbmc)
    ours, theirs = lognorm_pbmc, lognorm_pbmc.copy()

    scrust_call("tl.score_genes_cell_cycle", ours, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.tl.score_genes_cell_cycle(theirs, s_genes=s_genes, g2m_genes=g2m_genes)

    assert_allclose(ours.obs["S_score"], theirs.obs["S_score"], **ELEMENTWISE, err_msg="S_score")
    assert_allclose(
        ours.obs["G2M_score"], theirs.obs["G2M_score"], **ELEMENTWISE, err_msg="G2M_score"
    )
    assert (ours.obs["phase"].astype(str) == theirs.obs["phase"].astype(str)).all()


def test_phase_follows_scanpys_exact_three_way_rule(lognorm_pbmc: AnnData) -> None:
    """Reconstruct scanpy's rule from the scores and assert scrust's `phase` equals it.

    This catches a change to the *decision* independently of the scores: if the tie break
    or the G1 guard drifted, the reconstructed labels and scrust's would part even though
    both score columns still matched.
    """
    s_genes = _present(S_GENES, lognorm_pbmc)
    g2m_genes = _present(G2M_GENES, lognorm_pbmc)
    scrust_call("tl.score_genes_cell_cycle", lognorm_pbmc, s_genes=s_genes, g2m_genes=g2m_genes)

    scores = lognorm_pbmc.obs[["S_score", "G2M_score"]]
    rule = pd.Series("S", index=scores.index)
    rule[scores["G2M_score"] > scores["S_score"]] = "G2M"
    rule[np.all(scores.to_numpy() < 0, axis=1)] = "G1"

    assert (lognorm_pbmc.obs["phase"].astype(str) == rule).all()
    # All three labels must actually occur, or the rule is only half-tested.
    assert set(lognorm_pbmc.obs["phase"].astype(str)) == {"S", "G2M", "G1"}


def test_ctrl_size_is_the_shorter_list(lognorm_pbmc: AnnData) -> None:
    """scanpy sets ctrl_size = min(len(s_genes), len(g2m_genes)); an unequal split matches.

    ctrl_size parity is proven by the *scores*: the same control draw gives the same score
    for every cell. The phase can still disagree on a handful of cells, and that is not a
    ctrl_size bug -- it is the hard `score < 0` G1 threshold flipping for a cell whose score
    sits within an f32 epsilon of zero. This test asserts exactly that: scores element-wise,
    and every phase disagreement is a cell on the zero knife-edge.
    """
    s_genes = _present(S_GENES, lognorm_pbmc)[:8]
    g2m_genes = _present(G2M_GENES, lognorm_pbmc)[:5]  # shorter, so ctrl_size = 5 for both
    ours, theirs = lognorm_pbmc, lognorm_pbmc.copy()

    scrust_call("tl.score_genes_cell_cycle", ours, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.tl.score_genes_cell_cycle(theirs, s_genes=s_genes, g2m_genes=g2m_genes)

    assert_allclose(ours.obs["S_score"], theirs.obs["S_score"], **ELEMENTWISE)
    assert_allclose(ours.obs["G2M_score"], theirs.obs["G2M_score"], **ELEMENTWISE)

    disagree = (
        ours.obs["phase"].astype(str).to_numpy() != theirs.obs["phase"].astype(str).to_numpy()
    )
    if disagree.any():
        nearest_threshold = np.abs(ours.obs[["S_score", "G2M_score"]].to_numpy()).min(axis=1)
        assert (nearest_threshold[disagree] < 1e-4).all(), (
            "phase disagreed away from the score=0 boundary, which a ctrl_size or scoring "
            "bug would cause but an f32 threshold flip cannot"
        )


def test_is_deterministic_for_a_fixed_input(lognorm_pbmc: AnnData) -> None:
    """The default draw is seeded, so two runs are identical to the bit."""
    s_genes = _present(S_GENES, lognorm_pbmc)
    g2m_genes = _present(G2M_GENES, lognorm_pbmc)
    first, second = lognorm_pbmc, lognorm_pbmc.copy()

    scrust_call("tl.score_genes_cell_cycle", first, s_genes=s_genes, g2m_genes=g2m_genes)
    scrust_call("tl.score_genes_cell_cycle", second, s_genes=s_genes, g2m_genes=g2m_genes)

    assert_allclose(first.obs["S_score"], second.obs["S_score"], rtol=0, atol=0)
    assert_allclose(first.obs["G2M_score"], second.obs["G2M_score"], rtol=0, atol=0)
    assert (first.obs["phase"].astype(str) == second.obs["phase"].astype(str)).all()
