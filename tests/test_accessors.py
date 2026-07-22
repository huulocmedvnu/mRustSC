"""The accessor layer and the runtime settings, against scanpy and against hand maths.

`scrust.get` is pure AnnData plumbing, so unlike the algorithm branches every
assertion here can be exact: scanpy's frames are compared column by column with
their dtypes, and `aggregate` is compared both to scanpy and to values worked out
by hand on a matrix small enough to read.
"""

from __future__ import annotations

import contextlib
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

# `scrust/__init__.py` imports the extension eagerly, so this file needs one to be
# collectible without a compiled core. Nothing under test calls into it.
try:
    import scrust._scrust  # noqa: F401
except ImportError:
    _PLACEHOLDER = types.ModuleType("scrust._scrust")
    _PLACEHOLDER.gpu_available = lambda: False
    sys.modules["scrust._scrust"] = _PLACEHOLDER

from scrust import get
from scrust.settings import Settings, Verbosity

N_OBS = 8
N_VARS = 5
# Group "c" holds exactly one cell, which is the case that makes a sample variance
# undefined and a median trivial.
GROUPS = ["a", "b", "a", "c", "b", "a", "b", "a"]
GENE_NAMES = [f"gene{j}" for j in range(N_VARS)]
DE_GROUPS = ("a", "b")


@pytest.fixture
def dense_counts() -> np.ndarray:
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, size=(N_OBS, N_VARS)).astype(np.float32)
    counts[0, 0] = 0.0  # a structural zero, so count_nonzero is not trivially n_obs
    return counts


def make_adata(counts: np.ndarray, *, sparse: bool) -> AnnData:
    """A small AnnData carrying one of everything the accessors read."""
    adata = AnnData(sp.csr_matrix(counts) if sparse else counts.copy())
    adata.obs_names = [f"cell{i}" for i in range(N_OBS)]
    adata.var_names = GENE_NAMES
    adata.obs["group"] = pd.Categorical(GROUPS)
    adata.obs["n_counts"] = counts.sum(axis=1)
    adata.obs["batch"] = pd.Categorical(["x", "y"] * (N_OBS // 2))
    adata.var["dispersion"] = np.linspace(0.5, 1.5, N_VARS, dtype=np.float32)
    adata.varm["loadings"] = np.arange(N_VARS * 2, dtype=np.float32).reshape(N_VARS, 2)
    adata.obsm["X_pca"] = np.arange(N_OBS * 2, dtype=np.float32).reshape(N_OBS, 2)
    adata.layers["raw"] = adata.X.copy()
    return adata


@pytest.fixture(params=["dense", "sparse"])
def adata(request: pytest.FixtureRequest, dense_counts: np.ndarray) -> AnnData:
    return make_adata(dense_counts, sparse=request.param == "sparse")


def assert_matches_scanpy(ours: pd.DataFrame, theirs: pd.DataFrame) -> None:
    """Columns, index, values and dtypes all identical — callers pass these on to plotting."""
    pd.testing.assert_frame_equal(ours, theirs, check_exact=True)


# --------------------------------------------------------------------------------
# obs_df / var_df
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys",
    [
        (),
        ["gene1"],
        ["n_counts"],
        ["gene3", "n_counts", "group", "gene0"],
        ["group", "group"],  # a repeated key stays repeated
    ],
)
def test_obs_df_matches_scanpy(adata: AnnData, keys: list[str]) -> None:
    assert_matches_scanpy(get.obs_df(adata, keys), sc.get.obs_df(adata, keys))


def test_obs_df_accepts_a_bare_string_key(adata: AnnData) -> None:
    assert_matches_scanpy(get.obs_df(adata, "gene2"), sc.get.obs_df(adata, "gene2"))


def test_obs_df_with_obsm_and_layer(adata: AnnData) -> None:
    assert_matches_scanpy(
        get.obs_df(adata, ["gene0", "group"], obsm_keys=[("X_pca", 0), ("X_pca", 1)], layer="raw"),
        sc.get.obs_df(adata, ["gene0", "group"], [("X_pca", 0), ("X_pca", 1)], layer="raw"),
    )


@pytest.mark.parametrize("keys", [(), ["cell2"], ["dispersion"], ["cell0", "dispersion", "cell5"]])
def test_var_df_matches_scanpy(adata: AnnData, keys: list[str]) -> None:
    assert_matches_scanpy(get.var_df(adata, keys), sc.get.var_df(adata, keys))


def test_var_df_with_varm(adata: AnnData) -> None:
    assert_matches_scanpy(
        get.var_df(adata, ["dispersion"], varm_keys=[("loadings", 1)]),
        sc.get.var_df(adata, ["dispersion"], [("loadings", 1)]),
    )


def test_unknown_key_raises_like_scanpy(adata: AnnData) -> None:
    with pytest.raises(KeyError, match="Could not find keys"):
        get.obs_df(adata, ["not_a_thing"])
    with pytest.raises(KeyError):
        sc.get.obs_df(adata, ["not_a_thing"])


def test_gene_shadowing_an_obs_column_is_ambiguous(dense_counts: np.ndarray) -> None:
    """Preferring either side silently would hand back the wrong data, so it must raise."""
    adata = make_adata(dense_counts, sparse=False)
    adata.obs["gene1"] = np.arange(N_OBS, dtype=np.float32)

    with pytest.raises(KeyError, match="found in both"):
        get.obs_df(adata, ["gene1"])
    with pytest.raises(KeyError, match="found in both"):
        sc.get.obs_df(adata, ["gene1"])


def test_cell_shadowing_a_var_column_is_ambiguous(dense_counts: np.ndarray) -> None:
    adata = make_adata(dense_counts, sparse=False)
    adata.var["cell0"] = np.arange(N_VARS, dtype=np.float32)

    with pytest.raises(KeyError, match="found in both"):
        get.var_df(adata, ["cell0"])
    with pytest.raises(KeyError, match="found in both"):
        sc.get.var_df(adata, ["cell0"])


# --------------------------------------------------------------------------------
# rank_genes_groups_df
# --------------------------------------------------------------------------------


def write_de_result(adata: AnnData, *, method: str = "wilcoxon") -> None:
    """Write the structure `tl.rank_genes_groups` writes, dtypes included.

    Built by hand rather than by running scanpy's DE so that the accessor is
    tested against the contract's layout — `float64` p-values above all — and not
    against whatever a particular scanpy version happens to produce.
    """
    dtypes = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }
    values = {
        "names": [GENE_NAMES[::-1], GENE_NAMES],
        "scores": [[5.0, 4.0, 0.5, -1.0, -6.0], [7.0, 2.0, 0.0, -3.0, -8.0]],
        "logfoldchanges": [[3.0, 2.0, 0.25, -1.5, -4.0], [4.0, 1.0, 0.0, -2.0, -5.0]],
        # A p-value that underflows float32 to zero, which is why the field is float64.
        "pvals": [[1e-300, 1e-3, 0.4, 0.02, 1e-8], [1e-9, 0.01, 0.9, 0.03, 1e-40]],
        "pvals_adj": [[5e-300, 5e-3, 0.5, 0.04, 5e-8], [5e-9, 0.05, 0.95, 0.06, 5e-40]],
    }
    adata.uns["rank_genes_groups"] = {
        "params": {"groupby": "group", "reference": "rest", "method": method},
        **{
            field: np.rec.fromarrays(
                [np.asarray(column, dtype=dtypes[field]) for column in columns],
                dtype=[(name, dtypes[field]) for name in DE_GROUPS],
            )
            for field, columns in values.items()
        },
    }


@pytest.fixture
def de_adata(dense_counts: np.ndarray) -> AnnData:
    adata = make_adata(dense_counts, sparse=False)
    write_de_result(adata)
    return adata


@pytest.mark.parametrize("group", [None, "a", ["a"], ["b", "a"]])
def test_rank_genes_groups_df_matches_scanpy(de_adata: AnnData, group) -> None:
    assert_matches_scanpy(
        get.rank_genes_groups_df(de_adata, group), sc.get.rank_genes_groups_df(de_adata, group)
    )


def test_rank_genes_groups_df_keeps_float64_pvalues(de_adata: AnnData) -> None:
    """A float32 p-value column would round the smallest p-values to exactly zero."""
    frame = get.rank_genes_groups_df(de_adata, "a")
    assert frame["pvals"].dtype == np.float64
    assert frame["pvals_adj"].dtype == np.float64
    assert frame["pvals"].min() == pytest.approx(1e-300)


@pytest.mark.parametrize(
    "filters",
    [
        {"pval_cutoff": 0.05},
        {"log2fc_min": 1.0},
        {"log2fc_max": 1.0},
        {"pval_cutoff": 0.05, "log2fc_min": 0.5},
        {"pval_cutoff": 1e-300},  # filters everything out
    ],
)
def test_rank_genes_groups_df_filters_match_scanpy(de_adata: AnnData, filters: dict) -> None:
    for group in (None, "a"):
        assert_matches_scanpy(
            get.rank_genes_groups_df(de_adata, group, **filters),
            sc.get.rank_genes_groups_df(de_adata, group, **filters),
        )


def test_rank_genes_groups_df_single_group_drops_the_group_column(de_adata: AnnData) -> None:
    assert "group" not in get.rank_genes_groups_df(de_adata, "a").columns
    assert "group" in get.rank_genes_groups_df(de_adata, None).columns


def test_rank_genes_groups_df_logreg_has_no_pvalues(dense_counts: np.ndarray) -> None:
    adata = make_adata(dense_counts, sparse=False)
    write_de_result(adata, method="logreg")
    assert_matches_scanpy(
        get.rank_genes_groups_df(adata, None), sc.get.rank_genes_groups_df(adata, None)
    )


def test_rank_genes_groups_df_missing_key_raises(adata: AnnData) -> None:
    with pytest.raises(KeyError, match="rank_genes_groups"):
        get.rank_genes_groups_df(adata, None)
    with pytest.raises(KeyError):
        sc.get.rank_genes_groups_df(adata, None)


def test_rank_genes_groups_df_missing_group_raises(de_adata: AnnData) -> None:
    with pytest.raises(KeyError):
        get.rank_genes_groups_df(de_adata, "not_a_group")
    with pytest.raises(KeyError):
        sc.get.rank_genes_groups_df(de_adata, "not_a_group")


# --------------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------------


def expected_reduction(counts: np.ndarray, func: str) -> np.ndarray:
    """The reduction worked out directly from the dense matrix, group by group."""
    blocks = [counts[np.asarray(GROUPS) == name] for name in sorted(set(GROUPS))]
    reduce = {
        "sum": lambda block: block.sum(axis=0),
        "mean": lambda block: block.mean(axis=0),
        "count_nonzero": lambda block: (block != 0).sum(axis=0),
        "median": lambda block: np.median(block, axis=0),
        # ddof=1 is undefined for a single observation and numpy yields nan there.
        "var": lambda block: np.var(block, axis=0, ddof=1),
    }[func]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # numpy's own "ddof <= 0" notice
        return np.stack([reduce(block.astype(np.float64)) for block in blocks])


@pytest.mark.parametrize("func", ["sum", "mean", "count_nonzero", "var", "median"])
def test_aggregate_matches_hand_computation(
    adata: AnnData, dense_counts: np.ndarray, func: str
) -> None:
    with pytest.warns(RuntimeWarning) if func == "var" else _no_warning_check():
        result = get.aggregate(adata, "group", func)

    assert list(result.obs_names) == sorted(set(GROUPS))
    assert list(result.var_names) == GENE_NAMES
    np.testing.assert_allclose(
        result.layers[func], expected_reduction(dense_counts, func), rtol=1e-12
    )


@pytest.mark.parametrize("func", ["sum", "mean", "count_nonzero", "var", "median"])
def test_aggregate_matches_scanpy(adata: AnnData, func: str) -> None:
    with pytest.warns(RuntimeWarning) if func == "var" else _no_warning_check():
        ours = get.aggregate(adata, "group", func)
    with pytest.warns(RuntimeWarning) if func == "var" else _no_warning_check():
        theirs = sc.get.aggregate(adata, "group", func)

    np.testing.assert_allclose(ours.layers[func], theirs.layers[func], rtol=1e-12)
    assert list(ours.obs_names) == list(theirs.obs_names)
    assert list(ours.var_names) == list(theirs.var_names)
    pd.testing.assert_frame_equal(ours.obs, theirs.obs)


def test_aggregate_single_cell_group_has_no_variance(adata: AnnData) -> None:
    """A group of one has no sample variance; reporting 0 would look like a real result."""
    with pytest.warns(RuntimeWarning, match="var is nan"):
        result = get.aggregate(adata, "group", ["mean", "var"])

    single = list(result.obs_names).index("c")
    assert np.isnan(result.layers["var"][single]).all()
    assert np.isfinite(result.layers["var"][[i for i in range(3) if i != single]]).all()
    assert np.isfinite(result.layers["mean"]).all()


def test_aggregate_reports_group_sizes_and_several_functions(adata: AnnData) -> None:
    with pytest.warns(RuntimeWarning):
        result = get.aggregate(adata, "group", ["sum", "mean", "count_nonzero", "var", "median"])

    assert set(result.layers) == {"sum", "mean", "count_nonzero", "var", "median"}
    assert result.X is None
    assert result.obs["n_obs_aggregated"].tolist() == [4, 3, 1]
    assert result.layers["count_nonzero"].dtype == np.int64


def test_aggregate_over_a_layer(adata: AnnData, dense_counts: np.ndarray) -> None:
    np.testing.assert_allclose(
        get.aggregate(adata, "group", "sum", layer="raw").layers["sum"],
        expected_reduction(dense_counts, "sum"),
    )


def test_aggregate_over_two_columns_matches_scanpy(adata: AnnData) -> None:
    ours = get.aggregate(adata, ["group", "batch"], "sum")
    theirs = sc.get.aggregate(adata, ["group", "batch"], "sum")

    assert list(ours.obs_names) == list(theirs.obs_names)
    pd.testing.assert_frame_equal(ours.obs, theirs.obs)
    np.testing.assert_allclose(ours.layers["sum"], theirs.layers["sum"])


def test_aggregate_along_genes_matches_scanpy(adata: AnnData) -> None:
    adata.var["programme"] = pd.Categorical(["p", "q", "p", "q", "p"])
    ours = get.aggregate(adata, "programme", "sum", axis=1)
    theirs = sc.get.aggregate(adata, "programme", "sum", axis=1)

    assert ours.shape == theirs.shape == (N_OBS, 2)
    assert list(ours.obs_names) == list(theirs.obs_names)
    assert list(ours.var_names) == list(theirs.var_names)
    np.testing.assert_allclose(ours.layers["sum"], theirs.layers["sum"])


def test_aggregate_rejects_an_unknown_function(adata: AnnData) -> None:
    with pytest.raises(ValueError, match="is not one of"):
        get.aggregate(adata, "group", "stdev")


def test_aggregate_rejects_an_unknown_grouping(adata: AnnData) -> None:
    with pytest.raises(KeyError, match="not in the annotation"):
        get.aggregate(adata, "no_such_column", "sum")


def _no_warning_check() -> contextlib.AbstractContextManager[None]:
    """A `with` block that asserts nothing, so the warning check can be parametrised."""
    return contextlib.nullcontext()


# --------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------


@pytest.fixture
def fresh_settings() -> Settings:
    """A private instance: the singleton is process-wide and tests must not leak into it."""
    return Settings()


def test_defaults_are_quiet_and_device_agnostic(fresh_settings: Settings) -> None:
    assert fresh_settings.verbosity == Verbosity.warning
    assert fresh_settings.device == "auto"
    assert fresh_settings.max_memory_gb > 0
    assert (fresh_settings.n_jobs, fresh_settings.chunk_size) == (0, 0)


@pytest.mark.parametrize(
    ("assigned", "expected"),
    [
        ("debug", Verbosity.debug),
        ("INFO", Verbosity.info),
        (0, Verbosity.error),
        (3, Verbosity.hint),
        (Verbosity.warning, Verbosity.warning),
    ],
)
def test_verbosity_round_trips(fresh_settings: Settings, assigned, expected) -> None:
    fresh_settings.verbosity = assigned
    assert fresh_settings.verbosity is expected
    assert fresh_settings.verbosity.name == expected.name
    assert int(fresh_settings.verbosity) == int(expected)


@pytest.mark.parametrize("value", ["chatty", -1, 5, None])
def test_invalid_verbosity_raises(fresh_settings: Settings, value) -> None:
    with pytest.raises(ValueError, match="verbosity"):
        fresh_settings.verbosity = value


@pytest.mark.parametrize("device", ["auto", "cpu", "gpu", "metal"])
def test_device_accepts_the_names_the_core_parses(fresh_settings: Settings, device: str) -> None:
    fresh_settings.device = device
    assert fresh_settings.resolve_device() == device


def test_invalid_device_raises(fresh_settings: Settings) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        fresh_settings.device = "cuda"
    with pytest.raises(ValueError, match="device must be one of"):
        fresh_settings.resolve_device("cuda")


def test_resolve_device_prefers_the_callers_choice(fresh_settings: Settings) -> None:
    fresh_settings.device = "cpu"
    assert fresh_settings.resolve_device("gpu") == "gpu"
    assert fresh_settings.resolve_device(None) == "cpu"


def test_log_is_silent_at_the_default_verbosity(
    fresh_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fresh_settings.log("computing neighbours") is False
    assert fresh_settings.log("a detail", level="debug") is False
    assert capsys.readouterr().err == ""


def test_log_reports_once_verbosity_is_raised(
    fresh_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh_settings.verbosity = "hint"

    assert fresh_settings.log("computing neighbours") is True
    assert fresh_settings.log("using 15 of them", level="hint") is True
    assert fresh_settings.log("per-cell detail", level="debug") is False

    assert capsys.readouterr().err == "computing neighbours\n--> using 15 of them\n"


def test_log_always_reports_warnings_and_errors(
    fresh_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fresh_settings.log("empty cells dropped", level="warning") is True
    assert capsys.readouterr().err == "WARNING: empty cells dropped\n"


def test_importing_scrust_creates_the_singleton_without_side_effects() -> None:
    import scrust

    assert isinstance(scrust.settings, Settings)
    assert scrust.settings.verbosity == Verbosity.warning
