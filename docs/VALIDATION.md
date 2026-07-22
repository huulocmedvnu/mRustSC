# Validation

All numbers below were measured on an Apple M3 Pro with the synthetic negative
binomial generator in `tests/conftest.py`, reproduced by `tests/test_end_to_end.py`
and `scripts/benchmark.py`. They describe the default pipeline from
`mlxde.factory.build_default_pipeline`.

## Correctness of the fit

| Check | Result |
| --- | --- |
| GLM coefficients vs. an exact NB maximum-likelihood solve (`scipy.optimize`, per gene) | max abs deviation 1.7e-7 at dispersion 1e-8, 2.4e-6 at dispersion 0.25 |
| Dispersion recovery, true value 0.2 | median estimate 0.178-0.185 |
| GPU vs. CPU backend, identical inputs | log2 fold changes agree to 1e-3 relative |
| Estimated vs. true log2 fold change (8 samples/group) | correlation 0.954 |
| Recall of planted genes (6 samples/group, |log2FC| >= 3) | 96-100% |

## Calibration

Under the global null (1000 genes, no true signal), Benjamini-Hochberg makes
0-1 discoveries at FDR <= 0.05. Family-wise control therefore holds where it
matters.

Raw Wald p-values are mildly anti-conservative: P(p <= 0.05) measured at
0.072-0.079 instead of 0.05. With a planted signal the empirical false discovery
proportion among called genes runs at 9-14% for a nominal 5%.

Two known causes, neither of them a defect in this implementation:

1. **Wald test on small samples.** The normal approximation to the coefficient
   distribution is optimistic at 6 samples per group. Supplying the *true*
   dispersions leaves the same inflation (5.7-9.4% measured), which shows the
   test, not the dispersion estimator, dominates.
2. **Method-of-moments dispersion.** It underestimates by roughly 10% at this
   sample size, adding a further 2-4 points. DESeq2 avoids this with a Cox-Reid
   adjusted profile likelihood and empirical Bayes shrinkage, which this
   implementation deliberately does not include (YAGNI at the current scope).

Filtering results on an effect-size threshold after the correction, as
`significant(min_abs_log2_fold_change=...)` does, is a selection step outside the
FDR guarantee. It is offered because it is what analysts ask for, not because it
preserves the nominal rate.

**Practical consequence:** treat the adjusted p-values as a ranking that is
well-calibrated at the family level and slightly optimistic per gene. For
publication-grade FDR control on small designs, confirm hits with a permutation
null or a Cox-Reid-adjusted tool.

## Real data: 10x PBMC 3k

`scripts/validate_pbmc.py` runs the default pipeline on the 10x Genomics PBMC 3k
dataset shipped with scanpy (2 700 cells, 32 738 genes, raw UMI counts), with the
published `louvain` cell-type labels. CD14+ monocytes (480 cells) and B cells
(342 cells) are pooled into five pseudobulk replicates each; 2 465 genes survive
the count filter.

| Check | Result |
| --- | --- |
| Canonical markers called in the expected direction at FDR 5% | 14/14 (LYZ, S100A8/9, CD14, FCN1, VCAN, CST3, FTL up in monocytes; MS4A1, CD79A/B, TCL1A, BANK1, CD19 up in B cells) |
| Top-100 monocyte genes shared with `scanpy.tl.rank_genes_groups` (Wilcoxon on cells) | 91/100 |
| GPU vs CPU log2 fold changes, genes converged on both | max difference 6.3e-05 |

`tests/test_scanpy_reference.py` turns this into a regression suite: scanpy's
`rank_genes_groups` (Wilcoxon over cells) is the reference for six cell-type
pairs. Measured agreement, which is where the thresholds come from:

| pair | genes tested | rank correlation | top-100 shared | direction on scanpy's top 50 |
| --- | --- | --- | --- | --- |
| CD14+ Monocytes vs B cells | 2 465 | 0.948 | 91 | 50/50 |
| CD4 T cells vs B cells | 4 309 | 0.843 | 53 | 50/50 |
| NK cells vs CD14+ Monocytes | 2 307 | 0.932 | 80 | 50/50 |
| CD8 T cells vs B cells | 1 895 | 0.867 | 78 | 50/50 |
| CD4 T cells vs CD14+ Monocytes | 4 528 | 0.961 | 85 | 50/50 |
| FCGR3A+ Monocytes vs CD4 T cells | 4 365 | 0.833 | 74 | 50/50 |

Direction never disagrees; ranking correlates 0.83-0.96. The suite asserts
correlation > 0.75, top-100 overlap >= 45, and zero direction disagreements, and
was mutation-checked: swapping the group labels fails all 19 tests, and demanding
a 0.999 correlation fails 6.

scanpy is a reference for *ordering*, not for p-values. It ranks cells with a
rank-sum test; this package fits a negative binomial GLM to pseudobulk
replicates. Where they disagree on a borderline gene, neither is automatically
right.

**Pseudo-replicates are not biological replicates.** Splitting one donor's cells
into pools measures technical variation only, so the p-values here are far
smaller than a real multi-donor design would give. This check validates the
implementation and the direction of the biology, not the effect sizes'
significance.

**Complete separation.** Six genes are expressed in one cell type and completely
absent in the other, so their maximum-likelihood fold change is infinite. IRLS
stops wherever its clamps bite: +26.2 on the float64 CPU backend and +18.5 on the
float32 GPU backend for `IL1RN`. Those two numbers are both "infinity", but they
are the one place where the backends visibly disagree — 1 gene out of 2 465. The
validation script lists such genes; treat `|log2FC| > 15` as "detected in one
group only", not as a magnitude. Shrinking these towards a prior, as DESeq2's
`lfcShrink` does, is not implemented.

## Speed

GLM fit only (the dominant cost), 12 samples, 2 coefficients, warm kernels:

| genes | mlx (GPU) | numpy (CPU) | speedup |
| --- | --- | --- | --- |
| 5 000 | 0.012 s | 0.017 s | 1.4x |
| 20 000 | 0.028 s | 0.085 s | 3.1x |
| 60 000 | 0.043 s | 0.291 s | 6.8x |

The GPU only pays off once the batch is large enough to hide kernel launch
latency, which is why the CPU backend remains the default fallback and the
reference implementation. A full CLI run over 2 000 genes takes about 1 second
end to end, most of it Python start-up and CSV parsing.
