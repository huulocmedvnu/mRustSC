# Installing scrust

scrust is a compiled package: a Rust extension module (`scrust._scrust`) with a
thin Python layer around it. Wheels are therefore platform-specific — there is
no pure-Python fallback.

## From PyPI

```bash
pip install scrust
```

Wheels are published for **macOS 11+ on Apple silicon (arm64), CPython 3.11,
3.12 and 3.13**. On that combination nothing else is needed: the wheel carries
the compiled extension, and pip pulls in numpy, scipy, pandas and anndata.

On any other platform pip finds no wheel and falls back to the source
distribution, which needs a Rust toolchain and, today, does not build off macOS
— see [Other platforms](#other-platforms).

scanpy is not a dependency. Install it alongside if you want its plotting or its
readers, which is how the examples are written:

```bash
pip install scrust scanpy
```

## From source

Needed:

- a Rust toolchain (`rustup`, stable; the workspace pins `rust-version = 1.85`),
- Xcode command line tools, which supply the macOS SDK the extension links
  Metal against,
- Python 3.11 or newer.

```bash
git clone https://github.com/huulocmedvnu/mRustSC
cd mRustSC
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

`maturin develop` builds the extension and installs it into the active
virtualenv in place, which is what you want while working on the Rust side.
To produce a wheel instead:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin build --release   # writes target/wheels/*.whl
```

Always build `--release`. A debug build of the numerics is slow enough to look
broken.

### Optional: Apple Accelerate BLAS (macOS ARM64)

On Apple silicon you can route the dense linear algebra — PCA, Harmony, neighbour
distances, diffusion — through Apple's Accelerate (vecLib) BLAS/LAPACK instead of
the pure-Rust `matrixmultiply` backend:

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release --features accelerate
```

It is opt-in on purpose: the default build stays 100% pure-Rust and portable to
Linux/CI, and the sparse CSR paths (`normalize_total`, `log1p`) never touch BLAS
either way, so enabling it cannot change their memory profile. The gain is modest
and CPU-path only — measured ~7-8% on Harmony and ~4-9% on the full
PCA→Neighbors→UMAP→Harmony pipeline on an M3 Pro — because at single-cell sizes
these routines are not purely matmul-bound, and the default `device="auto"` path
runs on Metal rather than CPU BLAS. Enable it when you run the dense pipeline on
CPU and want that margin.

## Verifying the install

```bash
python -c "import scrust; print(scrust.__version__); print(scrust.gpu_available())"
```

`gpu_available()` reports whether a Metal device was found and initialised:

- `True` — the GPU path is live.
- `False` — everything still runs, on the CPU path. The CPU path is the same
  algorithm (it is the oracle the GPU path is tested against), but it is not
  bit-identical: f32 addition is not associative, so a GPU reduction lands a few
  ulps from a sequential one, and an unstable expression can amplify that. It
  has bitten us — an identical pair of cells cancelled to exactly zero distance
  on the CPU and to 9.8e-4 on Metal, which cost those cells their connectivity
  of 1 in UMAP (fixed in `neighbors.rs`). Expect agreement to tolerance, not to
  the last bit. GitHub's hosted macOS runners are the usual place `False` shows
  up, as they have no usable GPU.

Note that `settings.device` defaults to `"auto"`, which resolves to Metal
wherever one exists, so on a machine where `gpu_available()` is `True` a caller
who names no device is on the GPU.

A quick end-to-end check, without any dataset download:

```python
import numpy as np, scipy.sparse as sp
from anndata import AnnData
import scrust as sr

adata = AnnData(sp.random(500, 200, density=0.1, format="csr", dtype=np.float32))
sr.pp.normalize_total(adata)
sr.pp.log1p(adata)
sr.pp.pca(adata, n_comps=10)
print(adata.obsm["X_pca"].shape)
```

## Running the tests

The suite is not shipped in the wheel; run it from a source checkout, against an
extension you have already installed with `maturin develop --release`. Every
test file is collected through `tests/conftest.py`, which imports scanpy
unconditionally, so scanpy is needed for the whole suite and not only for the
cross-checks:

```bash
.venv/bin/pip install pytest scanpy
PYTHONPATH=$PWD/python .venv/bin/pytest -m "not reference"
```

Two markers are declared in `pyproject.toml`:

- `reference` — cross-checks against scanpy that want the PBMC 3k download.
- `slow` — drives umap-learn or scikit-learn over a full run; minutes, not
  seconds. Currently only `tests/test_umap_audit.py`.

`-m "not reference"` is the fast loop: it selects 644 of the 710 collected
tests, and it is what the `quality` CI job runs. Dropping the filter adds the
PBMC-3k legs, which download two h5ad files on first use and take appreciably
longer; CI gives them their own job with a cache.

### Which device the tests run against

`SCRUST_TEST_DEVICE` (`tests/scrust_call.py`) names the device the audits pass
into `_scrust`. It defaults to `"cpu"`; set it to `"auto"` to run the same suite
on the GPU:

```bash
SCRUST_TEST_DEVICE=auto PYTHONPATH=$PWD/python .venv/bin/pytest -m "not reference"
```

Both legs are worth running before a release, because `"auto"` is the device
most callers actually get.

`tests/test_device_parity.py` is the file that holds the two devices against
each other, and its `pytestmark` skips the whole file where `gpu_available()` is
false. That includes GitHub's hosted macOS runners, so **CI never executes it,
and a green tick is not evidence that the GPU path passes.** The GPU leg runs on
developer hardware and on a self-hosted Apple-silicon runner only.

## Type checking

The package ships a `py.typed` marker, so mypy and pyright use the inline
annotations of the installed package with no stub package needed.

```bash
pip install scrust mypy
python -c "import scrust, pathlib; print((pathlib.Path(scrust.__file__).parent / 'py.typed').exists())"
```

## Other platforms

The GPU path is Metal-only, and Metal is currently an unconditional dependency
of the extension, through two routes now. The first is candle: the workspace pins
`candle-core = { version = "0.9", features = ["metal"] }` with no
`cfg(target_os = "macos")` around it, and `scrust-core` and `scrust-py` both take
it from there. The second is `scrust-gpu`, which `scrust-py` now depends on for the
wired `knn` kernel and which links the `metal` crate directly. Either route makes the
build Apple-only. The practical consequences:

- **Apple silicon macOS** — supported. Wheels published, GPU path active.
- **Intel macOS** — no wheel is published. A source build should compile, since
  Metal exists there too, but it is neither tested nor benchmarked; treat it as
  unsupported.
- **Linux and Windows** — no wheel, and a source build fails to compile because
  candle's `metal` feature does not build off Apple platforms. Gating that
  feature behind a target `cfg` would make a CPU-only build possible; until
  someone does that work, there is nothing to install.

If you are not on Apple silicon, use scanpy.
