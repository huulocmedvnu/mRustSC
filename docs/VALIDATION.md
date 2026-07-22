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
