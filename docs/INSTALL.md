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
- Xcode command line tools, which supply the Metal compiler,
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

## Verifying the install

```bash
python -c "import scrust; print(scrust.__version__); print(scrust.gpu_available())"
```

`gpu_available()` reports whether a Metal device was found and initialised:

- `True` — the GPU path is live.
- `False` — everything still runs, on the CPU path. The CPU path is the same
  algorithm (it is the oracle the GPU path is tested against), so results are
  the same; only the speed differs. GitHub's hosted macOS runners are the usual
  place this happens, as they have no usable GPU.

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

## Type checking

The package ships a `py.typed` marker, so mypy and pyright use the inline
annotations of the installed package with no stub package needed.

```bash
pip install scrust mypy
python -c "import scrust, pathlib; print((pathlib.Path(scrust.__file__).parent / 'py.typed').exists())"
```

## Other platforms

The GPU path is Metal-only, and the Metal bindings are currently an
unconditional dependency of the extension crate — they are not behind a
`cfg(target_os = "macos")`. The practical consequences:

- **Apple silicon macOS** — supported. Wheels published, GPU path active.
- **Intel macOS** — no wheel is published. A source build should compile, since
  Metal exists there too, but it is neither tested nor benchmarked; treat it as
  unsupported.
- **Linux and Windows** — no wheel, and a source build fails to compile because
  the `metal` crate does not build off Apple platforms. Gating the GPU crate
  behind a target `cfg` would make a CPU-only build possible; until someone does
  that work, there is nothing to install.

If you are not on Apple silicon, use scanpy.
