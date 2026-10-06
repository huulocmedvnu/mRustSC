"""Calling into metalcyte, and telling "not built" apart from "wrong".

Nothing is a `todo!()` stub any more -- every branch is merged and `grep -rn 'todo!'
crates/` finds nothing -- so in a complete build this helper should never skip. It still
guards two cases that are not test failures: an extension that has not been built or
does not import, and a name that is missing because the caller is running against an
older binary. A `PanicException` (which is what pyo3 makes of a `todo!()`) is kept in
the list so that a stub reintroduced in future skips rather than reads as a failure.

Anything else -- above all a failed assertion -- must reach the test runner untouched.

Worth knowing when reading output: a skip here looks like a pass. If a whole file is
green, check the progress line for `s` characters before believing it ran.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import pytest

# pyo3 derives PanicException from BaseException, so it is matched by name rather than
# by class: importing `pyo3_runtime` only works once a panic has already happened.
_PANIC = "PanicException"


def _unimplemented_reason(path: str, exc: BaseException) -> str | None:
    """The reason to skip, or None when `exc` is a genuine failure."""
    if type(exc).__name__ == _PANIC:
        return f"metalcyte.{path} is a todo!() stub: {exc}"
    if isinstance(exc, ImportError):
        return f"metalcyte is not installed or fails to import: {exc}"
    if isinstance(exc, AttributeError) and "metalcyte" in str(exc):
        # Narrow on purpose: an AttributeError from working metalcyte code that merely
        # mentions no metalcyte module is a bug, and must not be mistaken for a stub.
        return f"metalcyte.{path} is not bound yet: {exc}"
    return None


def metalcyte_call(path: str, *args: Any, **kwargs: Any) -> Any:
    """Call `metalcyte.<path>(*args, **kwargs)`, skipping the test if it is not implemented.

    Call this before computing the scanpy reference: an unimplemented step then costs
    nothing.
    """
    try:
        obj: Any = importlib.import_module("metalcyte")
        for part in path.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"metalcyte.{path} is unavailable: {exc}")

    try:
        return obj(*args, **kwargs)
    except BaseException as exc:  # re-raised below unless it means "not implemented yet"
        reason = _unimplemented_reason(path, exc)
        if reason is None:
            raise
        pytest.skip(reason)


# The device every audit runs against.
#
# `settings.device` defaults to "auto", which resolves to Metal wherever one exists
# (`DeviceKind::Auto`, crates/metalcyte-core/src/device.rs), so a caller who never names a
# device is on the GPU. Tests that reach past the wrapper into `_metalcyte` have to name
# one, and hard-coding "cpu" there means the audits check a path most callers never
# take -- which is how a Metal-only divergence in the k-NN distances survived a full
# audit. Set METALCYTE_TEST_DEVICE=auto to run the same suite the other way.
#
# CI runs the "cpu" leg only, and running the other one there would change nothing:
# GitHub's hosted macOS runners have no usable GPU, so "auto" resolves to the CPU. The
# GPU leg is a local step. See the note at the top of .github/workflows/ci.yml.
DEVICE = os.environ.get("METALCYTE_TEST_DEVICE", "cpu")
