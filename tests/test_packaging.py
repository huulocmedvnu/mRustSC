"""Packaging checks: metadata that silently rots, and the files a wheel must carry.

Hermetic and fast — nothing here builds a wheel. The wheel test inspects one that
a previous `maturin build` left behind, and skips when there is none.
"""

from __future__ import annotations

import importlib
import tomllib
import zipfile
from pathlib import Path

import pytest

import scrust

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WHEEL_DIRS = (REPO_ROOT / "target" / "wheels", REPO_ROOT / "dist")

# Every module that publishes an `__all__`, by import path.
PUBLIC_MODULES = ["scrust", "scrust.get", "scrust.metrics", "scrust.pp", "scrust.tl"]


def _metadata() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_version_matches_the_package() -> None:
    assert _metadata()["version"] == scrust.__version__


def test_requires_python_covers_the_running_interpreter() -> None:
    # A classifier list that has drifted past `requires-python` is the usual way
    # a release starts refusing to install on a version CI still tests.
    classifiers = _metadata()["classifiers"]
    assert "Programming Language :: Rust" in classifiers
    assert any(c.startswith("Programming Language :: Python :: 3.") for c in classifiers)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_every_exported_name_is_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"{module_name}.__all__ names nothing: {missing}"


def test_py_typed_is_in_the_source_tree() -> None:
    # Asserted against the checkout, not the installed package: `maturin develop`
    # can leave an older install in the virtualenv.
    assert (REPO_ROOT / "python" / "scrust" / "py.typed").is_file()


def _newest_wheel() -> Path | None:
    wheels = [path for directory in WHEEL_DIRS for path in directory.glob("scrust-*.whl")]
    return max(wheels, key=lambda path: path.stat().st_mtime, default=None)


def test_py_typed_ships_inside_the_wheel() -> None:
    wheel = _newest_wheel()
    if wheel is None:
        pytest.skip("no built wheel; run `maturin build --release` first")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "scrust/py.typed" in names
    assert any(name.endswith(".so") for name in names), "wheel carries no extension module"
