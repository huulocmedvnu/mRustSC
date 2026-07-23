"""Audit of `de/glm.rs`, `de/dispersion.rs` and `de/hypothesis.rs`.

Reachability, established by reading `crates/scrust-py/src/` and by listing
`scrust._scrust`:

* `de/hypothesis.rs` is **partly reachable**. Its `erfc` is the only tail routine the
  crate ever uses: `de/wilcoxon.rs::two_sided_p` is literally
  `hypothesis::erfc(|z| / sqrt 2)`, and that runs on every call to the bound
  `_scrust.rank_genes_groups_wilcoxon`. So `erfc` -- the highest-value target, a Cody
  rational approximation with three branches -- can be exercised end to end through a
  real binding, which is what the first group of tests below does. Its `wald_test`
  is not reachable: nothing outside `hypothesis.rs` calls it.
* `de/glm.rs` (`fit_negative_binomial`) is **not reachable**: no pyfunction anywhere in
  `crates/scrust-py/src/` mentions it, and `scrust._scrust` exports no GLM entry point.
* `de/dispersion.rs` (`size_factors_median_of_ratios`,
  `dispersions_method_of_moments`, `shrink_towards_trend`) is **not reachable** either,
  for the same reason. Note that `_scrust.highly_variable_genes` reports "dispersions",
  but those come from `preprocess.rs`, not from `de/dispersion.rs`.

Because two of the three modules cannot be called from Python, the only honest thing
left to check about them is the correctness of the *transcribed reference values* their
Rust unit tests are judged against. `glm.rs` asserts its fit reproduces two hard-coded
tables of maximum likelihood coefficients to 1e-4; if those tables are wrong, that
assertion is worth nothing. The last group of tests re-derives them with `statsmodels`
(installed: 0.14.6) straight from the counts and design that appear in the same file,
parsed out of the Rust source so the check follows the source rather than a copy of it.
That is an audit of the crate's reference data, not of its shipped code, and is labelled
as such on every test. No reimplementation of the GLM is asserted against anywhere.

`dispersion.rs` carries no transcribed numeric reference, so nothing about it can be
checked from Python at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse, special, stats

from scrust_call import DEVICE, scrust_call

CORE_DE = Path(__file__).resolve().parents[1] / "crates" / "scrust-core" / "src" / "de"
GLM_RS = CORE_DE / "glm.rs"
HYPOTHESIS_RS = CORE_DE / "hypothesis.rs"

# Cody's erfc switches formula at |x| = 0.46875 and again at |x| = 4. `two_sided_p`
# feeds it |z| / sqrt(2), so the branch cuts sit at these z.
LOWER_BRANCH_Z = 0.46875 * np.sqrt(2.0)
UPPER_BRANCH_Z = 4.0 * np.sqrt(2.0)


# --------------------------------------------------------------------------------------
# erfc, reached through the one binding that uses it
# --------------------------------------------------------------------------------------


def rank_ladder(n_per_group: int):
    """A matrix whose gene `k` puts group A on ranks `k+1 .. k+m`, and its exact z.

    Every value is distinct and non-zero, so there are no ties and no structural
    zeros, and the rank-sum statistic is known in closed form:
    `U = m * k`, `mu = m^2 / 2`, `sigma^2 = m^2 (2m + 1) / 12`. Sweeping `k` over
    `0 ..= m` walks z from `-z_max` to `+z_max` in equal steps, which is how the whole
    range of `erfc` gets covered by one call to the binding.
    """
    m = n_per_group
    n_cells = 2 * m
    n_genes = m + 1
    dense = np.zeros((n_cells, n_genes), dtype=np.float32)
    for k in range(n_genes):
        # group A (rows 0..m) takes ranks k+1 .. k+m
        dense[:m, k] = np.arange(k + 1, k + m + 1, dtype=np.float32)
        # group B takes the complement: 1..k, then k+m+1 .. 2m
        b_ranks = np.concatenate([np.arange(1, k + 1), np.arange(k + m + 1, 2 * m + 1)]).astype(
            np.float32
        )
        dense[m:, k] = b_ranks

    u = m * np.arange(n_genes, dtype=np.float64)
    mean = m * m / 2.0
    sigma = np.sqrt(m * m * (2.0 * m + 1.0) / 12.0)
    return sparse.csr_matrix(dense), (u - mean) / sigma


def wilcoxon(matrix):
    labels = np.repeat([0, 1], matrix.shape[0] // 2).astype(np.uint32)
    matrix = matrix.tocsr()
    return scrust_call(
        "_scrust.rank_genes_groups_wilcoxon",
        matrix.indptr.astype(np.uint32),
        matrix.indices.astype(np.uint32),
        matrix.data.astype(np.float32),
        matrix.shape[1],
        labels,
        2,
        None,
        False,
        DEVICE,
    )


@pytest.fixture(scope="module")
def ladder():
    """scrust's scores and p-values on the rank ladder, plus the exact z."""
    matrix, exact_z = rank_ladder(100)
    result = wilcoxon(matrix)
    # Row 0 is group A against the rest, which is exactly group B.
    return result["scores"][0].astype(np.float64), result["p_values"][0], exact_z


def test_the_ladder_reproduces_the_exact_rank_sum_score(ladder):
    """The fixture's closed-form z is the z scrust computes, to f32 resolution.

    Without this the erfc tests below would be comparing scrust's p-value against a
    tail argument scrust never saw. Measured: max |z_scrust - z_exact| = 4.6e-7 over
    |z| <= 12.2, which is one f32 ulp of a number of that size and nothing more. The
    p-values are f64 and are computed from the unrounded f64 score, which is why the
    erfc comparison below can be held to 1e-13 rather than to this.
    """
    scores, _, exact_z = ladder
    assert np.abs(exact_z).max() > 12.0, "the ladder must reach the far tail"
    deviation = np.abs(scores - exact_z).max()
    assert deviation < 5e-6, f"max |z_scrust - z_exact| = {deviation}"


def test_erfc_matches_scipy_across_all_three_cody_branches(ladder):
    """`two_sided_p` = `hypothesis::erfc(|z| / sqrt 2)` against `scipy.special.erfc`.

    The ladder's 101 genes step z by ~0.244 from -12.2 to +12.2, so the erfc argument
    sweeps 0 to 8.6 and crosses both of Cody's branch cuts (|x| = 0.46875 and |x| = 4);
    the branch-cut test below checks that explicitly. This is the check with the most
    power in the file: a wrong coefficient in any one of the three rational fits, or a
    mis-ordered Horner loop, changes these p-values far beyond the tolerance.

    Measured: `hypothesis::erfc` agrees with `scipy.special.erfc` to a maximum relative
    deviation of 6.6e-15 (about 30 ulps of f64) over the argument range [0, 8.64], i.e.
    over p-values from 1.0 down to 2.5e-34. The 1e-13 tolerance is set from that.
    """
    _, p_values, exact_z = ladder
    expected = special.erfc(np.abs(exact_z) / np.sqrt(2.0))
    assert np.all(p_values > 0.0), "the tail underflowed"
    relative = np.abs(p_values - expected) / expected
    worst = relative.max()
    assert worst < 1e-13, f"max relative deviation {worst} at z={exact_z[relative.argmax()]}"


def test_the_two_erfc_branch_cuts_are_actually_crossed(ladder):
    """Guards the sweep above: it is worthless if it stays inside one branch.

    Pins that the ladder puts genes on both sides of |z| = 0.46875*sqrt2 and of
    |z| = 4*sqrt2, so all three of Cody's rational approximations are evaluated.
    """
    _, _, exact_z = ladder
    magnitude = np.abs(exact_z)
    for cut in (LOWER_BRANCH_Z, UPPER_BRANCH_Z):
        assert (magnitude < cut).any() and (magnitude > cut).any(), f"cut {cut} not straddled"
    assert (magnitude < LOWER_BRANCH_Z).any()
    assert ((magnitude > LOWER_BRANCH_Z) & (magnitude < UPPER_BRANCH_Z)).any()
    assert (magnitude > UPPER_BRANCH_Z).any()


def test_the_far_tail_keeps_relative_precision_a_complement_would_lose(ladder):
    """At z = 12.2 the p-value is ~2.4e-34; `1 - cdf` in f64 would return exactly 0.

    This is why `hypothesis.rs` is written through erfc. The test has teeth because it
    compares the value, not just its non-zeroness: `1 - 2*norm.cdf(-|z|)`-style
    cancellation would give 0, and a single-branch Chebyshev erfc would be wrong in the
    third digit here. Measured: over the 36 ladder genes with |z| > 8, agreement with
    `2 * scipy.stats.norm.sf(|z|)` is 3.7e-14 relative or better, and the smallest
    p-value returned is 2.5e-34.
    """
    _, p_values, exact_z = ladder
    far = np.abs(exact_z) > 8.0
    assert far.sum() >= 6, "the ladder must reach the far tail in several genes"
    expected = 2.0 * stats.norm.sf(np.abs(exact_z[far]))
    assert np.all(p_values[far] > 0.0)
    assert np.all(p_values[far] < 1e-15)
    assert p_values.min() < 1e-30
    relative = np.abs(p_values[far] - expected) / expected
    assert relative.max() < 1e-12, f"far-tail relative deviation {relative.max()}"


def test_erfc_is_exactly_one_at_zero_and_monotone_in_the_score(ladder):
    """Two properties a broken rational fit breaks: p(0) = 1 and p decreasing in |z|.

    The centre gene of the ladder has U exactly at its mean, so the erfc argument is
    exactly 0 and the p-value must be exactly 1.0 -- Cody's near-zero branch returns
    `1 - x * (...)`, which is 1 only if the branch is entered and the sign handled.
    """
    _, p_values, exact_z = ladder
    centre = int(np.argmin(np.abs(exact_z)))
    assert exact_z[centre] == 0.0
    assert p_values[centre] == 1.0

    order = np.argsort(np.abs(exact_z))
    ordered = p_values[order]
    assert np.all(np.diff(ordered) <= 0.0), "p-value is not monotone in |z|"


def test_erfc_is_symmetric_in_the_sign_of_the_score(ladder):
    """`erfc(|z|/sqrt2)` must not depend on which group is the larger one.

    Cody's implementation reflects negative arguments as `2 - scaled`; if the absolute
    value in `two_sided_p` were dropped, the two tails would differ by orders of
    magnitude. The ladder is symmetric by construction, so paired genes must agree
    bit for bit.
    """
    _, p_values, exact_z = ladder
    n = len(exact_z)
    for low in range(n // 2):
        high = n - 1 - low
        assert exact_z[low] == pytest.approx(-exact_z[high], abs=1e-12)
        assert p_values[low] == pytest.approx(p_values[high], rel=1e-12)


def test_the_transcribed_erfc_reference_table_in_hypothesis_rs_is_correct():
    """Audit of reference data: `hypothesis.rs`'s `REFERENCE` against scipy.

    `hypothesis.rs` judges its own `erfc` and `normal_survival` against 16 hard-coded
    triples documented as `scipy.special.erfc(x)` and `scipy.stats.norm.sf(x)`. Those
    Rust tests assert a 1e-14 relative agreement, so a mistyped digit in the table
    would either mask a defect or fail spuriously. This re-derives all 32 numbers.
    Measured: every entry agrees with scipy to within 1 ulp of f64.
    """
    source = HYPOTHESIS_RS.read_text()
    block = re.search(
        r"const REFERENCE: \[\(f64, f64, f64\); 16\] = \[(.*?)\n    \];", source, re.S
    )
    assert block is not None, "REFERENCE table not found in hypothesis.rs"
    rows = re.findall(r"\(\s*([-\d.e+]+),\s*([-\d.e+]+),\s*([-\d.e+]+)\s*\)", block.group(1))
    assert len(rows) == 16, f"parsed {len(rows)} rows, expected 16"

    for x_text, erfc_text, sf_text in rows:
        x, claimed_erfc, claimed_sf = float(x_text), float(erfc_text), float(sf_text)
        true_erfc = float(special.erfc(x))
        true_sf = float(stats.norm.sf(x))
        assert claimed_erfc == pytest.approx(true_erfc, rel=1e-15), f"erfc({x})"
        assert claimed_sf == pytest.approx(true_sf, rel=1e-15), f"norm.sf({x})"


# --------------------------------------------------------------------------------------
# glm.rs: the module is unreachable, so its transcribed MLE tables are audited instead
# --------------------------------------------------------------------------------------


def parse_reference_counts() -> np.ndarray:
    """The 6x10 count matrix `glm.rs::reference_counts` is fitted against."""
    source = GLM_RS.read_text()
    block = re.search(
        r"fn reference_counts\(\) -> Array2<f32> \{\s*array!\[(.*?)\n        \]", source, re.S
    )
    assert block is not None, "reference_counts not found in glm.rs"
    rows = re.findall(r"\[([^\]]*)\]", block.group(1))
    counts = np.array(
        [[float(v) for v in row.replace(".", ".0").split(",") if v.strip()] for row in rows]
    )
    assert counts.shape == (6, 10), counts.shape
    return counts


def parse_coefficient_table(name: str) -> np.ndarray:
    source = GLM_RS.read_text()
    block = re.search(rf"const {name}: \[\[f64; 2\]; 6\] = \[(.*?)\n    \];", source, re.S)
    assert block is not None, f"{name} not found in glm.rs"
    rows = re.findall(r"\[\s*([-\d.e+]+),\s*([-\d.e+]+),?\s*\]", block.group(1))
    assert len(rows) == 6, f"parsed {len(rows)} rows of {name}"
    return np.array([[float(a), float(b)] for a, b in rows])


def reference_design_and_offsets():
    """`two_group_design(5)` and `linear_space(0.7, 1.4, 10)`, in the crate's f32."""
    design = np.zeros((10, 2))
    design[:, 0] = 1.0
    design[5:, 1] = 1.0
    i = np.arange(10, dtype=np.float32)
    size_factors = np.float32(0.7) + np.float32(1.4 - 0.7) * i / np.float32(9.0)
    return design, size_factors.astype(np.float64)


def statsmodels_nb_fit(counts, design, size_factors, alpha):
    import statsmodels.api as sm

    offset = np.log(size_factors)
    out = np.zeros((counts.shape[0], design.shape[1]))
    for gene in range(counts.shape[0]):
        family = sm.families.NegativeBinomial(alpha=alpha)
        model = sm.GLM(counts[gene], design, family=family, offset=offset)
        out[gene] = model.fit(maxiter=200, tol=1e-13).params
    return out


@pytest.mark.parametrize(
    ("name", "alpha"),
    [
        ("MAXIMUM_LIKELIHOOD_POISSON_LIMIT", 1e-8),
        ("MAXIMUM_LIKELIHOOD_DISPERSION_025", 0.25),
    ],
)
def test_glm_rs_transcribed_maximum_likelihood_tables_match_statsmodels(name, alpha):
    """Audit of reference data: glm.rs's hard-coded MLEs, re-derived with statsmodels.

    `de/glm.rs` is not exposed to Python (no pyfunction mentions it), so the fit itself
    cannot be driven from here. What can be checked is the table its Rust test asserts
    against to 1e-4: 12 coefficients per table, documented as `scipy.optimize.minimize`
    on the negative binomial log likelihood. statsmodels' IRLS is an independent route
    to the same maximum, run here on the same counts, the same `two_group_design(5)` and
    the same `log(linear_space(0.7, 1.4, 10))` offset, all parsed out of glm.rs.

    A wrong entry here would silently weaken -- or falsely fail -- the crate's only
    accuracy test of its IRLS. Measured: statsmodels reproduces every transcribed
    coefficient to 2.0e-7 absolute for the Poisson-limit table and to 2.4e-6 for the
    dispersion-0.25 table. Both are well inside the 1e-4 the Rust test allows, so the
    tables are sound as references; see the next test for what the residual means.
    """
    counts = parse_reference_counts()
    design, size_factors = reference_design_and_offsets()
    claimed = parse_coefficient_table(name)
    fitted = statsmodels_nb_fit(counts, design, size_factors, alpha)

    deviation = np.abs(fitted - claimed).max()
    assert deviation < 1e-5, f"{name}: max |statsmodels - transcribed| = {deviation}"


def negative_binomial_nll(beta, y, design, offset, alpha):
    """NB2 negative log likelihood with the dispersion held fixed, on the log scale."""
    from scipy.special import gammaln

    mean = np.exp(design @ beta + offset)
    size = 1.0 / alpha
    return -np.sum(
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * np.log(size / (size + mean))
        + y * np.log(mean / (size + mean))
    )


def test_the_dispersion_025_table_is_transcribed_past_its_own_accuracy():
    """Documents a (harmless) defect in glm.rs's reference data, not in its code.

    `MAXIMUM_LIKELIHOOD_DISPERSION_025` prints ten decimals, but the numbers are only
    right to about six. Two independent routes to the same maximum -- statsmodels' IRLS
    and a direct Nelder-Mead minimisation of the NB2 log likelihood, started from the
    table itself -- agree with each other to 2.3e-8 while both differ from the table by
    up to 2.4e-6 (worst entry: gene 4, group coefficient). The residual is scattered
    across entries rather than systematic, which is the signature of an under-converged
    optimiser in whatever generated the table, not of a different model.

    Nothing downstream breaks: glm.rs asserts agreement only to 1e-4. The test is here
    so the discrepancy is on record and so that anyone later tightening that 1e-4
    towards 1e-6 knows the table, not the IRLS, is what will fail first.

    Asserted below in the direction that has power: the two independent optimisers must
    stay far closer to each other than either is to the table.
    """
    from scipy.optimize import minimize

    counts = parse_reference_counts()
    design, size_factors = reference_design_and_offsets()
    offset = np.log(size_factors)
    claimed = parse_coefficient_table("MAXIMUM_LIKELIHOOD_DISPERSION_025")
    by_irls = statsmodels_nb_fit(counts, design, size_factors, 0.25)
    by_direct = np.array(
        [
            minimize(
                negative_binomial_nll,
                claimed[gene],
                args=(counts[gene], design, offset, 0.25),
                method="Nelder-Mead",
                options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 200000, "maxfev": 200000},
            ).x
            for gene in range(counts.shape[0])
        ]
    )

    between_optimisers = np.abs(by_irls - by_direct).max()
    from_table = np.abs(by_irls - claimed).max()
    assert between_optimisers < 1e-6, f"the two optimisers disagree by {between_optimisers}"
    assert from_table > 1e-7, (
        "the table is now accurate to the digits it prints; update this test's docstring"
    )
    assert from_table > 10.0 * between_optimisers, (
        f"table residual {from_table} is no longer distinguishable from optimiser noise "
        f"{between_optimisers}"
    )
    assert from_table < 1e-4, f"the table is outside glm.rs's own tolerance: {from_table}"


def test_the_glm_reference_tables_differ_so_the_dispersion_actually_bites():
    """The two transcribed tables must not be the same numbers.

    If `MAXIMUM_LIKELIHOOD_DISPERSION_025` had been copied from the Poisson-limit table,
    glm.rs's `matches_maximum_likelihood_at_dispersion_025` would pass while testing
    nothing about the dispersion. The two maxima differ by ~1.7e-2 in the group
    coefficient, well outside the 1e-4 that test allows.
    """
    poisson = parse_coefficient_table("MAXIMUM_LIKELIHOOD_POISSON_LIMIT")
    dispersed = parse_coefficient_table("MAXIMUM_LIKELIHOOD_DISPERSION_025")
    separation = np.abs(poisson - dispersed).max()
    assert separation > 1e-3, f"the two tables are only {separation} apart"


def test_glm_and_dispersion_are_not_reachable_from_python():
    """Pins the reachability claim this file's docstring rests on.

    If a binding for the GLM or for `de/dispersion.rs` is ever added, this fails and the
    audit above (which can only inspect transcribed constants) must be replaced by a
    real end-to-end cross-check against statsmodels. It fails just as loudly if the
    Wilcoxon binding -- the only route to `hypothesis::erfc` -- disappears.
    """
    import scrust

    exported = set(dir(scrust._scrust))
    assert "rank_genes_groups_wilcoxon" in exported
    glm_like = {
        name
        for name in exported
        if any(k in name.lower() for k in ("glm", "negative_binomial", "size_factor", "wald"))
    }
    assert glm_like == set(), f"a GLM/dispersion binding now exists: {sorted(glm_like)}"

    py_sources = (Path(__file__).resolve().parents[1] / "crates" / "scrust-py" / "src").glob("*.rs")
    mentions = [
        p.name for p in py_sources if re.search(r"\bde::glm\b|\bde::dispersion\b", p.read_text())
    ]
    assert mentions == [], f"pyo3 layer now reaches the GLM: {mentions}"
