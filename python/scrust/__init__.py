"""Single-cell analysis with a Rust core running on the Apple GPU.

The API mirrors scanpy: `pp` for preprocessing, `tl` for tools, `metrics` and
`get` for the accessors. Functions take an `AnnData` and write their results
into the slots scanpy uses, so existing code and plotting keep working.
"""

from scrust import get, metrics, pp, tl
from scrust._scrust import gpu_available as _gpu_available_native
from scrust.settings import settings

__version__ = "0.2.0"


def gpu_available() -> bool:
    """True when a usable Metal device exists and the session has not been pinned to the CPU.

    `scrust.settings.device = "cpu"` (or `SCRUST_DEVICE=cpu` in the environment) makes this
    False, so code that branches on it stays off the GPU together with the API.
    """
    return settings.device != "cpu" and bool(_gpu_available_native())


__all__ = ["__version__", "get", "gpu_available", "metrics", "pl", "pp", "settings", "tl"]


def __getattr__(name: str):
    """Load `scrust.pl` on first access, so `import scrust` stays free of matplotlib."""
    if name == "pl":
        import scrust.pl as pl

        return pl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
