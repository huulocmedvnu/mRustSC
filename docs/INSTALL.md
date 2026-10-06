# Installing Metalcyte

Metalcyte is a compiled package: a Rust extension module (`metalcyte._metalcyte`) with a
thin Python layer around it. There is no pure-Python fallback, and there are no PyPI
wheels yet, so the package is built from source with maturin inside a virtualenv.

## From source

Needed:

- a Rust toolchain (`rustup`, stable, and the workspace pins `rust-version = 1.85`),
- on macOS, the Xcode command line tools, which supply the SDK the extension links
  Metal against,
- Python 3.11 or newer.

```bash
git clone https://github.com/huulocmedvnu/metalcyte
cd metalcyte
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

`maturin develop` builds the extension and installs it into the virtualenv in
place. pip pulls in numpy, scipy, pandas and anndata. To produce a wheel instead:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin build --release   # writes target/wheels/*.whl
```

Always build `--release`. A debug build of the numerics is slow enough to look
broken.

scanpy is not a dependency. Install it alongside if you want its plotting or its
readers, which is how the examples are written:

```bash
.venv/bin/pip install scanpy
```

### Apple Accelerate BLAS (macOS ARM64)

On Apple silicon the dense linear algebra (PCA, Harmony, neighbour distances,
diffusion) can run through Apple's Accelerate (vecLib) BLAS/LAPACK in place of
the pure-Rust `matrixmultiply` backend. The `accelerate` cargo feature controls
it, and `pyproject.toml` lists it under `[tool.maturin] features`, so the
`maturin develop` command above builds with it on. To build it explicitly:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release --features accelerate
```

For a pure-Rust BLAS build, remove `"accelerate"` from the
`features` list in `pyproject.toml`. The sparse CSR paths (`normalize_total`,
`log1p`) never touch BLAS either way, so the feature cannot change their memory
profile. The gain is on the CPU path only: measured ~7-8% on Harmony and ~4-9% on
the full PCA to Neighbors to UMAP to Harmony pipeline on an M3 Pro, because at
single-cell sizes these routines are not purely matmul-bound, and the default
`device="auto"` path runs that work on Metal.

## Verifying the install

```bash
python -c "import metalcyte; print(metalcyte.__version__); print(metalcyte.gpu_available())"
```

`gpu_available()` reports whether a Metal device was found and initialised:

- `True`: the GPU path is live.
- `False`: everything still runs, on the CPU path. The CPU path is the same
  algorithm (it is the oracle the GPU path is tested against), but it is not
  bit-identical: f32 addition is not associative, so a GPU reduction lands a few
  ulps from a sequential one, and an unstable expression can amplify that. The
  neighbour search handles one such case explicitly: an identical pair of cells
  cancels to exactly zero distance on the CPU and to 9.8e-4 on Metal, and
  squared distances below the expansion's resolution are snapped to zero so the
  two devices agree. Expect agreement to tolerance, not to the last bit. A
  machine without a usable GPU reports `False`.

`settings.device` defaults to `"auto"`, which resolves to Metal wherever one
exists, so on a machine where `gpu_available()` is `True` a caller who names no
device is on the GPU. Setting `METALCYTE_DEVICE=cpu` in the environment keeps a
whole session on the CPU.

A quick end-to-end check, without any dataset download:

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

The suite runs from a source checkout, against an extension you have already
installed with `maturin develop --release`. Every test file is collected through
`tests/conftest.py`, which imports scanpy unconditionally, so scanpy is needed
for the whole suite and not only for the cross-checks:

```bash
.venv/bin/pip install pytest scanpy
PYTHONPATH=$PWD/python .venv/bin/pytest -m "not reference"
```

Two markers are declared in `pyproject.toml`:

- `reference`: cross-checks against scanpy that want the PBMC 3k download.
- `slow`: drives umap-learn or scikit-learn over a full run, which takes minutes.
  Only `tests/test_umap_audit.py` carries it.

`-m "not reference"` is the fast loop: it selects 644 of the 710 collected
tests. Dropping the filter adds the PBMC 3k legs, which download two h5ad files
on first use and take appreciably longer.

### Which device the tests run against

`METALCYTE_TEST_DEVICE` (`tests/metalcyte_call.py`) names the device the audits pass
into `_metalcyte`. It defaults to `"cpu"`. Set it to `"auto"` to run the same suite
on the GPU:

```bash
METALCYTE_TEST_DEVICE=auto PYTHONPATH=$PWD/python .venv/bin/pytest -m "not reference"
```

Both legs are worth running, because `"auto"` is the device most callers get.

`tests/test_device_parity.py` holds the two devices against each other, and its
`pytestmark` skips the whole file where `gpu_available()` is false. A run on a
machine without Metal therefore says nothing about the GPU path.

## Type checking

The package ships a `py.typed` marker, so mypy and pyright use the inline
annotations of the installed package with no stub package needed.

```bash
.venv/bin/pip install mypy
.venv/bin/python -c "import metalcyte, pathlib; print((pathlib.Path(metalcyte.__file__).parent / 'py.typed').exists())"
```

## Other platforms

The GPU path is Metal, so it exists only on Apple hardware.

- **Apple silicon macOS**: supported. GPU path active.
- **Linux**: untested. The crates link Apple's Metal and Accelerate frameworks unconditionally, so
  a Linux build would need those dependencies made optional first.
- **Intel macOS**: a source build should compile, since Metal exists there too,
  but it is neither tested nor benchmarked. Treat it as unsupported.
- **Windows**: not tested.
