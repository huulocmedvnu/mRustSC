# Installing Metalcyte

Metalcyte is a compiled package. Its calculations live in an extension module written in Rust
(`metalcyte._metalcyte`), and a small Python layer calls into it. Metalcyte has no Python-only
version, so the extension must be installed for anything to work.

## From PyPI

PyPI hosts ready-built packages (wheels) for macOS on Apple silicon, Python 3.11 to 3.13:

```bash
pip install metalcyte            # numpy, scipy, pandas and anndata come with it
pip install "metalcyte[plot]"    # matplotlib and seaborn for metalcyte.pl
```

On any other platform, pip tries to build Metalcyte from its source code. That build needs a Rust
toolchain. It also links Apple's Accelerate maths library by default, so it fails outside macOS. To
get a CPU-only package there, clone the repository, remove `accelerate` from `features` in
`pyproject.toml`, and build from the clone.

## From source

You need:

- a Rust toolchain (`rustup`, stable). The workspace requires `rust-version = 1.88`.
- on macOS, the Xcode command line tools. They supply the Apple files that the extension needs to
  use Metal, Apple's interface to the GPU.
- Python 3.11 or newer.

```bash
git clone https://github.com/huulocmedvnu/metalcyte
cd metalcyte
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

`maturin develop` compiles the extension and installs it into the virtual environment. pip also
installs numpy, scipy, pandas and anndata. To produce a wheel file instead:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin build --release   # writes target/wheels/*.whl
```

Always build with `--release`. Without it, the compiler skips its optimisations and the
calculations run so slowly that the package seems broken.

Metalcyte does not require scanpy. The examples use scanpy to read files and to plot, so install it
too if you want to run them:

```bash
.venv/bin/pip install scanpy
```

### Apple Accelerate (macOS on Apple silicon)

Accelerate is Apple's library for fast linear algebra (BLAS and LAPACK, also called vecLib). On
Apple silicon it uses the chip's matrix unit. Metalcyte can send its dense matrix calculations (PCA,
Harmony, neighbour distances, diffusion maps) to Accelerate. Without it, Metalcyte uses
`matrixmultiply`, a matrix library written in Rust.

The `accelerate` cargo feature turns this on. `pyproject.toml` lists it under
`[tool.maturin] features`, so the `maturin develop` command above already includes it. To ask for
it explicitly:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release --features accelerate
```

To build without Accelerate, remove `"accelerate"` from the `features` list in `pyproject.toml`.

The choice has little effect in practice:

- `normalize_total` and `log1p` work on sparse matrices and never call BLAS, so their memory use is
  the same either way.
- Accelerate only speeds up work done on the CPU. On an M3 Pro it made Harmony ~7-8% faster and the
  full PCA to Neighbors to UMAP to Harmony pipeline ~4-9% faster. At single-cell sizes these steps
  spend only part of their time on matrix multiplication.
- With the default `device="auto"`, this work runs on the GPU through Metal anyway.

## Checking the install

```bash
python -c "import metalcyte; print(metalcyte.__version__); print(metalcyte.gpu_available())"
```

`gpu_available()` tells you whether Metalcyte found a GPU it can use through Metal:

- `True`: Metalcyte will run on the GPU.
- `False`: everything still runs, on the CPU. A machine without a usable GPU reports `False`.

The CPU and GPU versions use the same algorithms. The test suite uses the CPU version as the
reference for the GPU version. Their results agree within a small tolerance, though they can differ
in the last digits. The GPU adds numbers in a different order than the CPU, and with 32-bit floating
point numbers the order changes the last bits of the sum. Some formulas can make such tiny
differences larger.

The neighbour search handles one such case on purpose. For two identical cells, the CPU computes a
distance of exactly zero and the GPU computes 9.8e-4. Metalcyte sets squared distances below the
precision of the calculation to zero, so both devices report zero.

`settings.device` defaults to `"auto"`, which means "use the GPU if there is one". If
`gpu_available()` is `True` and you do not name a device, Metalcyte runs on the GPU. To keep a whole
session on the CPU, set `METALCYTE_DEVICE=cpu` in the environment.

A quick check from start to end, without downloading any dataset:

```python
import numpy as np, scipy.sparse as sp
from anndata import AnnData
import metalcyte as mc

adata = AnnData(sp.random(500, 200, density=0.1, format="csr", dtype=np.float32))
mc.pp.normalize_total(adata)
mc.pp.log1p(adata)
mc.pp.pca(adata, n_comps=10)
print(adata.obsm["X_pca"].shape)
```

## Running the tests

You run the tests from a copy of the source code, after installing the extension with
`maturin develop --release`. The whole suite needs scanpy. pytest loads every test file through
`tests/conftest.py`, and that file imports scanpy.

```bash
.venv/bin/pip install pytest scanpy
PYTHONPATH=$PWD/python .venv/bin/pytest -m "not reference"
```

`pyproject.toml` defines two test labels (markers):

- `reference`: comparisons with scanpy that need the PBMC 3k dataset download.
- `slow`: tests that run umap-learn or scikit-learn in full, which takes minutes. Only
  `tests/test_umap_audit.py` has this label.

`-m "not reference"` is the quick run. It selects 644 of the 710 tests. Without the filter, pytest
also runs the PBMC 3k tests. These download two h5ad files the first time and take much longer.

### Choosing the device for the tests

`METALCYTE_TEST_DEVICE` (`tests/metalcyte_call.py`) sets the device that the tests pass to
`_metalcyte`. The default is `"cpu"`. To run the same tests on the GPU, set it to `"auto"`:

```bash
METALCYTE_TEST_DEVICE=auto PYTHONPATH=$PWD/python .venv/bin/pytest -m "not reference"
```

Run both. Most users get `"auto"`, so the GPU run tests what they use.

`tests/test_device_parity.py` compares the CPU and GPU results. It skips itself when
`gpu_available()` is false. So a test run on a machine without Metal does not check the GPU code.

## Type checking

The package includes a `py.typed` file. Type checkers such as mypy and pyright then read the type
annotations in the installed package, with no extra stub package.

```bash
.venv/bin/pip install mypy
.venv/bin/python -c "import metalcyte, pathlib; print((pathlib.Path(metalcyte.__file__).parent / 'py.typed').exists())"
```

## Other platforms

Metalcyte uses the GPU through Metal, which exists only on Apple hardware.

- **macOS on Apple silicon**: supported, with the GPU.
- **Linux**: not tested. The Rust code always links Apple's Metal and Accelerate libraries. A Linux
  build would first need these to become optional.
- **macOS on Intel**: a build from source should compile, since Metal exists there too. It is not
  tested or benchmarked. Treat it as unsupported.
- **Windows**: not tested.
