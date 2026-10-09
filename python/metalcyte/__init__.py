"""Single-cell RNA-seq analysis on Apple silicon Macs.

Metalcyte does the computation in Rust. It runs the heaviest steps on the GPU through
Metal. The CPU and the GPU share one memory, so data is not copied between them.

The functions sit in five modules. `pp` holds preprocessing, `tl` the analysis tools,
`metrics` the scores, `get` the helpers that pull tables out of the results, and `pl`
the plots. Every function takes an `AnnData` and writes its result into a fixed slot of
it, such as `obs["leiden"]` or `obsm["X_umap"]`.
"""

from metalcyte import get, metrics, pp, tl
from metalcyte._metalcyte import gpu_available as _gpu_available_native
from metalcyte.settings import settings

__version__ = "0.3.4"


def gpu_available() -> bool:
    """True when a usable Metal GPU is present and the session is not set to use only the CPU.

    `metalcyte.settings.device = "cpu"` (or `METALCYTE_DEVICE=cpu` in the environment) makes this
    False, so code that branches on it stays off the GPU together with the API.
    """
    return settings.device != "cpu" and bool(_gpu_available_native())


__all__ = ["__version__", "get", "gpu_available", "metrics", "pl", "pp", "settings", "tl"]


def __getattr__(name: str):
    """Load `metalcyte.pl` on first access, so `import metalcyte` stays free of matplotlib."""
    if name == "pl":
        import metalcyte.pl as pl

        return pl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
