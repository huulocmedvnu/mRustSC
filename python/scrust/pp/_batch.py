"""Removing unwanted variation. Owned by feat/regress-combat."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

__all__ = ["combat", "regress_out"]


def regress_out(
    adata: AnnData, keys: str | Sequence[str], *, device: str = "auto", inplace: bool = True
) -> None:
    """Regress each gene on `keys` and keep the residuals, as `scanpy.pp.regress_out`."""
    raise NotImplementedError("feat/regress-combat")


def combat(
    adata: AnnData,
    key: str = "batch",
    *,
    covariates: Sequence[str] | None = None,
    device: str = "auto",
    inplace: bool = True,
) -> None:
    """Empirical Bayes batch correction, as `scanpy.pp.combat`."""
    raise NotImplementedError("feat/regress-combat")
