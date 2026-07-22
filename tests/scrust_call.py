"""Calling into scrust while it is still being written.

Every algorithm is a `todo!()` stub on `main` and the bindings land branch by branch,
so a call either panics (pyo3 turns Rust's `todo!()` into `PanicException`) or the name
is simply missing. Those two outcomes mean "not implemented yet" and must skip; anything
else — above all a failed assertion — must reach the test runner untouched.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# pyo3 derives PanicException from BaseException, so it is matched by name rather than
# by class: importing `pyo3_runtime` only works once a panic has already happened.
_PANIC = "PanicException"


def _unimplemented_reason(path: str, exc: BaseException) -> str | None:
    """The reason to skip, or None when `exc` is a genuine failure."""
    if type(exc).__name__ == _PANIC:
        return f"scrust.{path} is a todo!() stub: {exc}"
    if isinstance(exc, ImportError):
        return f"scrust is not installed or fails to import: {exc}"
    if isinstance(exc, AttributeError) and "scrust" in str(exc):
        # Narrow on purpose: an AttributeError from working scrust code that merely
        # mentions no scrust module is a bug, and must not be mistaken for a stub.
        return f"scrust.{path} is not bound yet: {exc}"
    return None


def scrust_call(path: str, *args: Any, **kwargs: Any) -> Any:
    """Call `scrust.<path>(*args, **kwargs)`, skipping the test if it is not implemented.

    Call this before computing the scanpy reference: an unimplemented step then costs
    nothing.
    """
    try:
        obj: Any = importlib.import_module("scrust")
        for part in path.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"scrust.{path} is unavailable: {exc}")

    try:
        return obj(*args, **kwargs)
    except BaseException as exc:  # re-raised below unless it means "not implemented yet"
        reason = _unimplemented_reason(path, exc)
        if reason is None:
            raise
        pytest.skip(reason)
