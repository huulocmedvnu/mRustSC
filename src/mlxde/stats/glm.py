"""Negative binomial GLM fitting for every gene at once.

A differential expression run needs one GLM per gene: thousands of tiny,
independent, identically shaped problems. They are solved with a single batched
IRLS loop so that each iteration is a handful of large kernel launches over
``(n_genes, n_samples)`` and ``(n_genes, p, p)`` tensors instead of thousands of
latency-bound solves.
"""

from __future__ import annotations

import numpy as np

from mlxde.contracts import Array, ComputeBackend, DesignMatrix, GLMFit

_MINIMUM_MEAN = 1e-8
"""Fitted means are kept away from zero; the IRLS weights and working response
both divide by ``mu``."""

_MAXIMUM_LINEAR_PREDICTOR = 30.0
"""``exp(30)`` is ~1e13: far above any realistic count, far below overflow."""

_MAXIMUM_WORKING_RESIDUAL = 1e4
"""Bounds ``(y - mu) / mu`` while an early iterate is still far from the mode, so
one badly scaled gene cannot produce a non-finite normal equation."""

_RIDGE = 1e-8
"""Tikhonov term added to the information matrix. It only matters for genes
whose weights collapse to zero (all-zero counts), where it turns a singular
solve into a finite step; against realistic information matrices it is
numerically invisible."""

_PRECISION_FLOOR_ULPS = 128.0
"""Smallest relative coefficient change a backend is trusted to resolve. On
float32 this is ~1.5e-5, orders of magnitude below any standard error; on
float64 it is far below the default tolerance and therefore inert."""

_STARTING_PSEUDOCOUNT = 0.1
"""Keeps ``log(y / size_factor)`` finite for the least-squares starting values."""


class NegativeBinomialGLM:
    """GLMFitter using batched iteratively reweighted least squares.

    The model per gene ``g`` and sample ``s`` is
    ``mu[g, s] = size_factors[s] * exp(design[s] @ coefficients[g])``
    with a negative binomial likelihood of known dispersion ``dispersions[g]``.

    A gene has converged once every coefficient moves by less than
    ``tolerance * max(|coefficient|, 1)`` in one iteration; it is then left
    alone, so the reported flags always describe the returned coefficients.
    Genes that never reach the tolerance are flagged, never raised.
    """

    def __init__(
        self, backend: ComputeBackend, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> None:
        self._backend = backend
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def fit(
        self,
        counts: np.ndarray,
        design: DesignMatrix,
        size_factors: np.ndarray,
        dispersions: np.ndarray,
    ) -> GLMFit:
        """Fit one negative binomial GLM per row of ``counts``."""
        counts = np.asarray(counts, dtype=np.float64)
        size_factors = np.asarray(size_factors, dtype=np.float64)
        dispersions = np.asarray(dispersions, dtype=np.float64)
        self._validate(counts, design, size_factors, dispersions)

        backend = self._backend
        observed = backend.asarray(counts)
        offsets = backend.asarray(size_factors)
        # A trailing axis lets the (n_genes,) dispersions broadcast over samples.
        alpha = backend.asarray(np.maximum(dispersions, 0.0)[:, None])
        design_batch = backend.asarray(design.matrix[None, :, :])
        design_transposed = backend.asarray(design.matrix.T)

        tolerance = self._effective_tolerance()
        coefficients = self._starting_coefficients(observed, offsets, design_batch, counts.shape)
        # 0 = still moving, 1 = converged; kept on the device to avoid a per-gene
        # host round trip, shaped (n_genes, 1) to broadcast over coefficients.
        is_frozen = backend.asarray(np.zeros((counts.shape[0], 1)))

        n_iterations = 0
        for iteration in range(1, self._max_iterations + 1):
            n_iterations = iteration
            weights, working_response = self._working_problem(
                observed, offsets, alpha, coefficients, design_transposed
            )
            candidate = self._weighted_least_squares(weights, working_response, design_batch)
            step = backend.abs(candidate - coefficients)
            # Relative for large coefficients, absolute for small ones. No
            # max-reduction exists in the backend protocol, so count the
            # coefficients that still move by more than the tolerance instead.
            threshold = tolerance * backend.maximum(backend.abs(coefficients), 1.0)
            n_moving = self._as_column(
                backend.sum(backend.where(step < threshold, 0.0, 1.0), axis=1)
            )
            # Converged genes stop being updated, so the reported flags always
            # describe the coefficients actually returned.
            coefficients = backend.where(is_frozen > 0.0, coefficients, candidate)
            is_frozen = backend.where(n_moving > 0.0, is_frozen, 1.0)
            if float(backend.sum(is_frozen)) == counts.shape[0]:
                break

        weights, _ = self._working_problem(
            observed, offsets, alpha, coefficients, design_transposed
        )
        _, information = self._information(weights, design_batch)
        fitted_means = self._fitted_means(offsets, coefficients, design_transposed)

        return GLMFit(
            coefficients=np.asarray(backend.asnumpy(coefficients), dtype=np.float64),
            covariance=np.asarray(backend.asnumpy(backend.inverse(information)), dtype=np.float64),
            dispersions=dispersions,
            fitted_means=np.asarray(backend.asnumpy(fitted_means), dtype=np.float64),
            converged=np.asarray(backend.asnumpy(is_frozen)).reshape(-1) > 0.0,
            n_iterations=n_iterations,
        )

    @staticmethod
    def _validate(
        counts: np.ndarray,
        design: DesignMatrix,
        size_factors: np.ndarray,
        dispersions: np.ndarray,
    ) -> None:
        n_genes, n_samples = counts.shape
        if design.n_samples != n_samples:
            raise ValueError(f"design has {design.n_samples} rows, counts have {n_samples} samples")
        if size_factors.shape != (n_samples,):
            raise ValueError(f"expected {n_samples} size factors, got {size_factors.shape}")
        if dispersions.shape != (n_genes,):
            raise ValueError(f"expected {n_genes} dispersions, got {dispersions.shape}")
        if np.any(size_factors <= 0.0):
            raise ValueError("size factors must be strictly positive")

    def _effective_tolerance(self) -> float:
        """The requested tolerance, floored at the precision of the backend.

        A float32 device cannot resolve coefficient changes below a few ulps, so
        a stricter request would keep every gene iterating to ``max_iterations``
        and report convergence that did in fact happen as failure.
        """
        probe = np.asarray(self._backend.asnumpy(self._backend.asarray(np.zeros(1))))
        return max(self._tolerance, _PRECISION_FLOOR_ULPS * float(np.finfo(probe.dtype).eps))

    @staticmethod
    def _as_column(values: Array) -> Array:
        """Append a trailing axis of length one.

        The backend protocol has no reshape; ``None`` indexing is understood by
        both NumPy and MLX arrays and costs nothing on either.
        """
        return values[..., None]

    def _starting_coefficients(
        self,
        observed: Array,
        offsets: Array,
        design_batch: Array,
        counts_shape: tuple[int, int],
    ) -> Array:
        """Ordinary least squares on ``log(y / size_factor)``.

        Good starting values buy more accuracy per unit of work than extra IRLS
        iterations, and keep the first weighted solve well conditioned.
        """
        backend = self._backend
        log_response = backend.log(observed / offsets + _STARTING_PSEUDOCOUNT)
        unit_weights = backend.asarray(np.ones((counts_shape[0], counts_shape[1], 1)))
        return self._weighted_least_squares(unit_weights, log_response, design_batch)

    def _fitted_means(self, offsets: Array, coefficients: Array, design_transposed: Array) -> Array:
        backend = self._backend
        linear_predictor = backend.clip(
            backend.matmul(coefficients, design_transposed),
            -_MAXIMUM_LINEAR_PREDICTOR,
            _MAXIMUM_LINEAR_PREDICTOR,
        )
        return backend.maximum(offsets * backend.exp(linear_predictor), _MINIMUM_MEAN)

    def _working_problem(
        self,
        observed: Array,
        offsets: Array,
        alpha: Array,
        coefficients: Array,
        design_transposed: Array,
    ) -> tuple[Array, Array]:
        """IRLS weights ``mu / (1 + alpha * mu)`` and working response, as columns."""
        backend = self._backend
        fitted_means = self._fitted_means(offsets, coefficients, design_transposed)
        weights = fitted_means / (1.0 + alpha * fitted_means)
        residual = backend.clip(
            (observed - fitted_means) / fitted_means,
            -_MAXIMUM_WORKING_RESIDUAL,
            _MAXIMUM_WORKING_RESIDUAL,
        )
        # The offset is not part of eta, so it drops out of the working response.
        linear_predictor = backend.log(fitted_means) - backend.log(offsets)
        return self._as_column(weights), linear_predictor + residual

    def _information(self, weights: Array, design_batch: Array) -> tuple[Array, Array]:
        """Weighted design ``(W X)'`` and information matrix ``X' W X + ridge``.

        ``weights`` is ``(n_genes, n_samples, 1)`` so that it scales the design
        rows; the design is ``(1, n_samples, p)`` and broadcasts over genes.
        """
        backend = self._backend
        weighted_design_transposed = backend.transpose(weights * design_batch, (0, 2, 1))
        ridge = backend.asarray(_RIDGE * np.eye(design_batch.shape[2])[None, :, :])
        information = backend.matmul(weighted_design_transposed, design_batch) + ridge
        return weighted_design_transposed, information

    def _weighted_least_squares(
        self, weights: Array, response: Array, design_batch: Array
    ) -> Array:
        """Solve the batched normal equations ``(X' W X) b = X' W z``."""
        weighted_design_transposed, information = self._information(weights, design_batch)
        right_hand_side = self._backend.matmul(
            weighted_design_transposed, self._as_column(response)
        )
        return self._backend.solve(information, right_hand_side)[..., 0]
