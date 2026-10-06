"""`scrust` was renamed Silicell. This package re-exports it and warns once."""

import warnings as _warnings

from silicell import *  # noqa: F401,F403
from silicell import __version__, get, gpu_available, metrics, pp, settings, tl  # noqa: F401

_warnings.warn(
    "`scrust` is now `silicell`: `import silicell as si`. The `scrust` name will be removed in 0.4.",
    DeprecationWarning,
    stacklevel=2,
)
