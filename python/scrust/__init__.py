"""`scrust` was renamed Metalcyte. This package re-exports it and warns once."""

import warnings as _warnings

from metalcyte import *  # noqa: F401,F403
from metalcyte import __version__, get, gpu_available, metrics, pp, settings, tl  # noqa: F401

_warnings.warn(
    "`scrust` is now `metalcyte`: `import metalcyte as mc`. The `scrust` name will be removed in 0.4.",
    DeprecationWarning,
    stacklevel=2,
)
