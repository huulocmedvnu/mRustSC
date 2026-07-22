"""Behaviour of the scanpy-shaped Python layer, against a recording fake core.

The Rust algorithms are stubs, so nothing numerical is asserted here. What is
asserted is what this layer owns: argument translation, the defaults the
contract fixes, which AnnData slot is written and with what dtype, the
`inplace=False` return path, and the errors.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

# `scrust/__init__.py` imports the extension eagerly, so this file needs one to be
# collectible without a compiled core. Only stand in when there is genuinely none:
# leaving a placeholder in `sys.modules` next to a working extension makes every
# later test module skip itself as "not bound yet" while reporting green.
try:
    import scrust._scrust  # noqa: F401
except ImportError:
    _PLACEHOLDER = types.ModuleType("scrust._scrust")
    _PLACEHOLDER.gpu_available = lambda: False
    sys.modules["scrust._scrust"] = _PLACEHOLDER

from scrust import pp, tl

N_OBS = 6
N_VARS = 4
GROUPS = ["a", "b", "a", "c", "b", "a"]


class FakeCore:
    """Stands in for `scrust._scrust`, recording every call it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def _record(self, name: str, args: tuple) -> None:
        self.calls.append((name, args))

    def args_of(self, name: str) -> tuple:
        matches = [args for called, args in self.calls if called == name]
        assert matches, f"{name} was never called; got {[c for c, _ in self.calls]}"
        return matches[0]

    @staticmethod
    def _n_rows(indptr) -> int:
        return len(indptr) - 1

    def gpu_available(self) -> bool:
        return False

    def filter_cells(self, indptr, indices, values, n_cols, min_genes, min_counts):
        self._record("filter_cells", (indptr, indices, values, n_cols, min_genes, min_counts))
        mask = np.ones(self._n_rows(indptr), dtype=bool)
        mask[0] = False
        return mask

    def filter_genes(self, indptr, indices, values, n_cols, min_cells, min_counts):
        self._record("filter_genes", (indptr, indices, values, n_cols, min_cells, min_counts))
        mask = np.ones(n_cols, dtype=bool)
        mask[-1] = False
        return mask

    def normalize_total(self, indptr, indices, values, n_cols, target_sum, device):
        self._record("normalize_total", (indptr, indices, values, n_cols, target_sum, device))
        return indptr, indices, values * 2

    def log1p(self, indptr, indices, values, n_cols):
        self._record("log1p", (indptr, indices, values, n_cols))
        return indptr, indices, np.log1p(values)

    def scale(self, indptr, indices, values, n_cols, zero_center, max_value, device):
        self._record("scale", (indptr, indices, values, n_cols, zero_center, max_value, device))
        return np.zeros((self._n_rows(indptr), n_cols), dtype=np.float64)

    def highly_variable_genes(self, indptr, indices, values, n_cols, n_top_genes, flavor, device):
        self._record(
            "highly_variable_genes",
            (indptr, indices, values, n_cols, n_top_genes, flavor, device),
        )
        return {
            "highly_variable": np.arange(n_cols) < 2,
            "means": np.arange(n_cols, dtype=np.float64),
            "normalised_dispersions": np.arange(n_cols, dtype=np.float64),
        }

    def pca(self, indptr, indices, values, n_cols, n_components, zero_center, seed, device):
        self._record(
            "pca", (indptr, indices, values, n_cols, n_components, zero_center, seed, device)
        )
        n_rows = self._n_rows(indptr)
        return {
            "embedding": np.zeros((n_rows, n_components), dtype=np.float64),
            "components": np.zeros((n_components, n_cols), dtype=np.float64),
            "explained_variance": np.zeros(n_components, dtype=np.float64),
            "explained_variance_ratio": np.zeros(n_components, dtype=np.float64),
        }

    def knn(self, embedding, k, device):
        self._record("knn", (embedding, k, device))
        n_rows = embedding.shape[0]
        indices = np.tile(np.arange(k, dtype=np.uint32), (n_rows, 1))
        return indices, np.ones((n_rows, k), dtype=np.float64)

    def connectivities(self, indices, distances):
        self._record("connectivities", (indices, distances))
        n_rows, k = indices.shape
        return (
            np.arange(0, n_rows * k + 1, k, dtype=np.uint32),
            indices.ravel(),
            distances.ravel(),
        )

    def umap(
        self,
        indptr,
        indices,
        values,
        n_cols,
        n_components,
        n_epochs,
        min_dist,
        spread,
        learning_rate,
        negative_sample_rate,
        seed,
        device,
    ):
        self._record(
            "umap",
            (
                indptr,
                indices,
                values,
                n_cols,
                n_components,
                n_epochs,
                min_dist,
                spread,
                learning_rate,
                negative_sample_rate,
                seed,
                device,
            ),
        )
        return np.zeros((self._n_rows(indptr), n_components), dtype=np.float64)

    def tsne(
        self,
        embedding,
        n_components,
        perplexity,
        early_exaggeration,
        learning_rate,
        n_iterations,
        seed,
        device,
    ):
        self._record(
            "tsne",
            (
                embedding,
                n_components,
                perplexity,
                early_exaggeration,
                learning_rate,
                n_iterations,
                seed,
                device,
            ),
        )
        return np.zeros((embedding.shape[0], n_components), dtype=np.float64)

    def rank_genes_groups_wilcoxon(
        self, indptr, indices, values, n_cols, labels, n_groups, reference, tie_correct, device
    ):
        self._record(
            "rank_genes_groups_wilcoxon",
            (indptr, indices, values, n_cols, labels, n_groups, reference, tie_correct, device),
        )
        stats = np.zeros((n_groups, n_cols), dtype=np.float64)
        return {
            "scores": stats,
            "p_values": stats,
            "adjusted_p_values": stats,
            "log2_fold_changes": stats,
        }


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> FakeCore:
    """Stand a recording double in for the compiled core.

    `from scrust import _scrust` resolves the attribute on the package, which is
    bound once at import; patching `sys.modules` alone would leave a real
    extension in place and silently test it instead of the double.
    """
    import scrust

    fake = FakeCore()
    monkeypatch.setitem(sys.modules, "scrust._scrust", fake)
    monkeypatch.setattr(scrust, "_scrust", fake, raising=False)
    return fake


def _counts() -> np.ndarray:
    return np.arange(N_OBS * N_VARS, dtype=np.float32).reshape(N_OBS, N_VARS)


def _adata(x=None) -> AnnData:
    return AnnData(
        X=sp.csr_matrix(_counts()) if x is None else x,
        obs=pd.DataFrame(
            {"group": pd.Categorical(GROUPS)},
            index=[f"cell{i}" for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"gene{j}" for j in range(N_VARS)]),
    )


# --- matrix conversion -------------------------------------------------------


@pytest.mark.parametrize(
    "matrix",
    [
        sp.csr_matrix(_counts()),
        sp.csc_matrix(_counts()),
        sp.csr_array(_counts()),
        _counts(),
    ],
    ids=["csr", "csc", "csr_array", "dense"],
)
def test_any_supported_x_reaches_the_core_as_csr(core: FakeCore, matrix) -> None:
    pp.log1p(_adata(matrix))
    indptr, indices, values, n_cols = core.args_of("log1p")
    assert n_cols == N_VARS
    assert len(indptr) == N_OBS + 1
    assert indptr.dtype == np.uint32
    assert indices.dtype == np.uint32
    assert values.dtype == np.float32


def test_other_sparse_formats_are_converted(core: FakeCore) -> None:
    # AnnData refuses COO in X, so the conversion is exercised on the helper.
    assert isinstance(pp._as_csr(sp.coo_matrix(_counts())), sp.csr_matrix)


def test_unsupported_x_raises_a_clear_error() -> None:
    with pytest.raises(TypeError, match=r"scipy\.sparse matrix or a numpy\.ndarray"):
        pp._csr_args([[1, 2], [3, 4]])


# --- filtering ---------------------------------------------------------------


def test_filter_cells_subsets_obs_and_forwards_both_criteria(core: FakeCore) -> None:
    adata = _adata()
    assert pp.filter_cells(adata, min_genes=2, min_counts=7) is None
    assert adata.n_obs == N_OBS - 1
    assert adata.obs_names[0] == "cell1"
    assert core.args_of("filter_cells")[4:] == (2, 7)


def test_filter_cells_not_inplace_returns_the_mask(core: FakeCore) -> None:
    adata = _adata()
    mask = pp.filter_cells(adata, min_genes=2, inplace=False)
    assert mask.dtype == bool
    assert mask.shape == (N_OBS,)
    assert adata.n_obs == N_OBS
    assert core.args_of("filter_cells")[4:] == (2, None)


def test_filter_genes_subsets_var(core: FakeCore) -> None:
    adata = _adata()
    pp.filter_genes(adata, min_cells=3)
    assert adata.n_vars == N_VARS - 1
    assert core.args_of("filter_genes")[4:] == (3, None)


def test_filter_without_a_criterion_raises(core: FakeCore) -> None:
    with pytest.raises(ValueError, match="min_genes or min_counts"):
        pp.filter_cells(_adata())
    with pytest.raises(ValueError, match="min_cells or min_counts"):
        pp.filter_genes(_adata())


# --- normalisation -----------------------------------------------------------


def test_normalize_total_writes_x_with_contract_defaults(core: FakeCore) -> None:
    adata = _adata()
    assert pp.normalize_total(adata) is None
    assert sp.issparse(adata.X)
    assert adata.X.shape == (N_OBS, N_VARS)
    assert core.args_of("normalize_total")[4:] == (None, "auto")


def test_normalize_total_not_inplace_leaves_x_alone(core: FakeCore) -> None:
    adata = _adata()
    before = adata.X.toarray()
    result = pp.normalize_total(adata, target_sum=1e4, inplace=False)
    assert sp.issparse(result)
    assert np.array_equal(adata.X.toarray(), before)
    assert core.args_of("normalize_total")[4] == 1e4


def test_log1p_writes_x_and_records_the_base(core: FakeCore) -> None:
    adata = _adata()
    assert pp.log1p(adata) is None
    assert sp.issparse(adata.X)
    assert adata.uns["log1p"] == {"base": None}


def test_log1p_not_inplace_returns_the_matrix(core: FakeCore) -> None:
    adata = _adata()
    result = pp.log1p(adata, inplace=False)
    assert result.shape == (N_OBS, N_VARS)
    assert "log1p" not in adata.uns


# --- highly variable genes ---------------------------------------------------


def test_highly_variable_genes_writes_the_three_var_columns(core: FakeCore) -> None:
    adata = _adata()
    assert pp.highly_variable_genes(adata) is None
    assert adata.var["highly_variable"].dtype == bool
    assert adata.var["means"].to_numpy().dtype == np.float32
    assert adata.var["dispersions_norm"].to_numpy().dtype == np.float32
    assert core.args_of("highly_variable_genes")[4:] == (2000, "seurat", "auto")


def test_highly_variable_genes_not_inplace_returns_a_frame(core: FakeCore) -> None:
    adata = _adata()
    table = pp.highly_variable_genes(adata, n_top_genes=3, flavor="cell_ranger", inplace=False)
    assert list(table.columns) == ["highly_variable", "means", "dispersions_norm"]
    assert list(table.index) == list(adata.var_names)
    assert "highly_variable" not in adata.var
    assert core.args_of("highly_variable_genes")[4:] == (3, "cell_ranger", "auto")


# --- scaling -----------------------------------------------------------------


def test_scale_writes_a_dense_f32_x(core: FakeCore) -> None:
    adata = _adata()
    assert pp.scale(adata) is None
    assert isinstance(adata.X, np.ndarray)
    assert adata.X.dtype == np.float32
    assert adata.X.shape == (N_OBS, N_VARS)
    assert core.args_of("scale")[4:] == (True, None, "auto")


def test_scale_not_inplace_returns_the_array(core: FakeCore) -> None:
    adata = _adata()
    scaled = pp.scale(adata, zero_center=False, max_value=10.0, inplace=False)
    assert scaled.dtype == np.float32
    assert sp.issparse(adata.X)
    assert core.args_of("scale")[4:] == (False, 10.0, "auto")


# --- pca ---------------------------------------------------------------------


def test_pca_writes_every_contract_slot(core: FakeCore) -> None:
    adata = _adata()
    assert pp.pca(adata, n_comps=3) is None
    assert adata.obsm["X_pca"].shape == (N_OBS, 3)
    assert adata.obsm["X_pca"].dtype == np.float32
    assert adata.varm["PCs"].shape == (N_VARS, 3)
    assert adata.varm["PCs"].dtype == np.float32
    assert adata.uns["pca"]["variance_ratio"].shape == (3,)
    assert adata.uns["pca"]["variance_ratio"].dtype == np.float32


def test_pca_forwards_contract_defaults_and_maps_random_state_to_seed(core: FakeCore) -> None:
    pp.pca(_adata())
    assert core.args_of("pca")[4:] == (50, True, 0, "auto")


def test_pca_passes_device_through(core: FakeCore) -> None:
    pp.pca(_adata(), random_state=7, device="cpu")
    assert core.args_of("pca")[6:] == (7, "cpu")


# --- neighbors ---------------------------------------------------------------


def _with_pca(core: FakeCore, n_comps: int = 3) -> AnnData:
    adata = _adata()
    pp.pca(adata, n_comps=n_comps)
    return adata


def test_neighbors_writes_both_graphs_and_the_uns_entry(core: FakeCore) -> None:
    adata = _with_pca(core)
    assert pp.neighbors(adata) is None
    for key in ("distances", "connectivities"):
        assert sp.issparse(adata.obsp[key])
        assert adata.obsp[key].shape == (N_OBS, N_OBS)
    assert adata.uns["neighbors"]["distances_key"] == "distances"
    assert adata.uns["neighbors"]["connectivities_key"] == "connectivities"
    assert adata.uns["neighbors"]["params"] == {
        "n_neighbors": 15,
        "method": "umap",
        "use_rep": "X_pca",
    }


def test_neighbors_uses_the_pca_representation_and_forwards_k(core: FakeCore) -> None:
    adata = _with_pca(core)
    pp.neighbors(adata, n_neighbors=4, device="cpu")
    embedding, k, device = core.args_of("knn")
    assert embedding.shape == (N_OBS, 3)
    assert embedding.dtype == np.float32
    # scanpy counts the cell itself among n_neighbors; the core does not.
    assert (k, device) == (3, "cpu")
    # connectivities is fed the knn result, not the matrix.
    assert core.args_of("connectivities")[0].shape == (N_OBS, 3)


def test_neighbors_honours_use_rep_x(core: FakeCore) -> None:
    pp.neighbors(_adata(), use_rep="X")
    assert core.args_of("knn")[0].shape == (N_OBS, N_VARS)


def test_neighbors_without_the_representation_raises(core: FakeCore) -> None:
    with pytest.raises(KeyError, match="X_pca"):
        pp.neighbors(_adata())


# --- embeddings --------------------------------------------------------------


def _with_neighbors(core: FakeCore) -> AnnData:
    adata = _with_pca(core)
    pp.neighbors(adata, n_neighbors=3)
    return adata


def test_umap_lays_out_the_connectivities_graph(core: FakeCore) -> None:
    adata = _with_neighbors(core)
    assert tl.umap(adata) is None
    assert adata.obsm["X_umap"].shape == (N_OBS, 2)
    assert adata.obsm["X_umap"].dtype == np.float32
    indptr, _, _, n_cols, *params = core.args_of("umap")
    assert (len(indptr) - 1, n_cols) == (N_OBS, N_OBS)
    assert params == [2, 200, 0.5, 1.0, 1.0, 5, 0, "auto"]


def test_umap_forwards_overrides(core: FakeCore) -> None:
    adata = _with_neighbors(core)
    tl.umap(adata, n_components=3, min_dist=0.1, spread=2.0, n_epochs=100, random_state=5)
    assert list(core.args_of("umap")[4:]) == [3, 100, 0.1, 2.0, 1.0, 5, 5, "auto"]
    assert adata.obsm["X_umap"].shape == (N_OBS, 3)


def test_umap_without_neighbors_raises(core: FakeCore) -> None:
    with pytest.raises(KeyError, match="connectivities"):
        tl.umap(_adata())


def test_tsne_slices_the_pca_to_n_pcs(core: FakeCore) -> None:
    adata = _with_pca(core, n_comps=10)
    assert tl.tsne(adata, n_pcs=4) is None
    embedding, *params = core.args_of("tsne")
    assert embedding.shape == (N_OBS, 4)
    # learning_rate defaults to scikit-learn's "auto" rule, which floors at 50.
    assert params == [2, 30.0, 12.0, 50.0, 1000, 0, "auto"]
    assert adata.obsm["X_tsne"].shape == (N_OBS, 2)
    assert adata.obsm["X_tsne"].dtype == np.float32


def test_tsne_defaults_take_all_available_components(core: FakeCore) -> None:
    adata = _with_pca(core, n_comps=3)
    tl.tsne(adata)
    assert core.args_of("tsne")[0].shape == (N_OBS, 3)


# --- rank_genes_groups -------------------------------------------------------

_DE_FIELDS = ("names", "scores", "pvals", "pvals_adj", "logfoldchanges")


def test_rank_genes_groups_builds_scanpy_structured_arrays(core: FakeCore) -> None:
    adata = _adata()
    assert tl.rank_genes_groups(adata, "group") is None
    result = adata.uns["rank_genes_groups"]
    assert set(result) == {*_DE_FIELDS, "params"}
    for field in _DE_FIELDS:
        assert result[field].dtype.names == ("a", "b", "c")
        assert result[field].shape == (N_VARS,)
    assert result["names"]["a"].dtype == np.dtype("O")
    assert result["scores"].dtype["a"] == np.float32
    assert result["logfoldchanges"].dtype["a"] == np.float32
    assert result["pvals"].dtype["a"] == np.float64
    assert result["pvals_adj"].dtype["a"] == np.float64


def test_rank_genes_groups_names_are_gene_names_in_core_order(core: FakeCore) -> None:
    adata = _adata()
    tl.rank_genes_groups(adata, "group")
    # The fake ranks genes back to front, so names must be reversed var_names.
    assert list(adata.uns["rank_genes_groups"]["names"]["a"]) == list(adata.var_names)[::-1]


def test_rank_genes_groups_writes_the_params_subdict(core: FakeCore) -> None:
    adata = _adata()
    tl.rank_genes_groups(adata, "group")
    assert adata.uns["rank_genes_groups"]["params"] == {
        "groupby": "group",
        "reference": "rest",
        "method": "wilcoxon",
        "use_raw": False,
        "layer": None,
        "corr_method": "benjamini-hochberg",
    }


def test_rank_genes_groups_encodes_labels_and_rest_reference(core: FakeCore) -> None:
    tl.rank_genes_groups(_adata(), "group")
    _, _, _, _, labels, n_groups, reference, tie_correct, device = core.args_of(
        "rank_genes_groups_wilcoxon"
    )
    assert labels.dtype == np.uint32
    assert list(labels) == [0, 1, 0, 2, 1, 0]
    # "rest" is the core's None; the unsigned boundary has no room for a sentinel.
    assert (n_groups, reference, tie_correct, device) == (3, None, False, "auto")


def test_rank_genes_groups_named_reference_is_labelled_but_not_reported(core: FakeCore) -> None:
    adata = _adata()
    tl.rank_genes_groups(adata, "group", groups=["a", "b"], reference="c")
    labels, n_groups, reference = core.args_of("rank_genes_groups_wilcoxon")[4:7]
    assert list(labels) == [0, 1, 0, 2, 1, 0]
    assert (n_groups, reference) == (3, 2)
    assert adata.uns["rank_genes_groups"]["names"].dtype.names == ("a", "b")


def test_rank_genes_groups_excludes_cells_outside_the_selected_groups(core: FakeCore) -> None:
    tl.rank_genes_groups(_adata(), "group", groups=["a", "c"])
    labels, n_groups = core.args_of("rank_genes_groups_wilcoxon")[4:6]
    # Cells outside the selected groups are dropped before the call rather
    # than labelled negatively, which the unsigned boundary cannot carry.
    assert list(labels) == [0, 0, 1, 0]
    assert n_groups == 2


def test_rank_genes_groups_rejects_unknown_input(core: FakeCore) -> None:
    with pytest.raises(ValueError, match="wilcoxon"):
        tl.rank_genes_groups(_adata(), "group", method="t-test")
    with pytest.raises(ValueError, match="unknown groups"):
        tl.rank_genes_groups(_adata(), "group", groups=["z"])
    with pytest.raises(KeyError, match="louvain"):
        tl.rank_genes_groups(_adata(), "louvain")


def test_rank_genes_groups_result_is_readable_by_scanpy(core: FakeCore) -> None:
    scanpy = pytest.importorskip("scanpy")
    adata = _adata()
    tl.rank_genes_groups(adata, "group")
    frame = scanpy.get.rank_genes_groups_df(adata, group="a")
    assert list(frame["names"]) == list(adata.var_names)[::-1]
    assert set(_DE_FIELDS) <= set(frame.columns) | {"names"}
