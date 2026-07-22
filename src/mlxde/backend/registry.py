"""Backend lookup.

New backends register a factory instead of editing the resolution logic, so this
module stays closed to modification while the set of backends stays open.
"""

from __future__ import annotations

from collections.abc import Callable

from mlxde.contracts import ComputeBackend

BackendFactory = Callable[[], ComputeBackend]

_FACTORIES: dict[str, BackendFactory] = {}
_PREFERENCE_ORDER: list[str] = []


def register_backend(name: str, factory: BackendFactory, *, preferred: bool = False) -> None:
    """Make ``name`` resolvable by :func:`get_backend`."""
    _FACTORIES[name] = factory
    if name in _PREFERENCE_ORDER:
        _PREFERENCE_ORDER.remove(name)
    _PREFERENCE_ORDER.insert(0 if preferred else len(_PREFERENCE_ORDER), name)


def available_backends() -> list[str]:
    """Names of registered backends that can actually run on this machine."""
    return [name for name in _PREFERENCE_ORDER if _instantiate(name) is not None]


def get_backend(name: str | None = None) -> ComputeBackend:
    """Return the named backend, or the best available one when ``name`` is None."""
    if name is not None:
        if name not in _FACTORIES:
            raise KeyError(f"unknown backend {name!r}; registered: {sorted(_FACTORIES)}")
        backend = _instantiate(name)
        if backend is None:
            raise RuntimeError(f"backend {name!r} is not available on this machine")
        return backend

    for candidate in _PREFERENCE_ORDER:
        backend = _instantiate(candidate)
        if backend is not None:
            return backend
    raise RuntimeError("no compute backend is available")


def _instantiate(name: str) -> ComputeBackend | None:
    try:
        backend = _FACTORIES[name]()
    except ImportError:
        return None
    return backend if backend.is_available() else None


def _numpy_backend() -> ComputeBackend:
    from mlxde.backend.numpy_backend import NumpyBackend

    return NumpyBackend()


def _mlx_backend() -> ComputeBackend:
    from mlxde.backend.mlx_backend import MLXBackend

    return MLXBackend()


register_backend("numpy", _numpy_backend)
register_backend("mlx", _mlx_backend, preferred=True)
