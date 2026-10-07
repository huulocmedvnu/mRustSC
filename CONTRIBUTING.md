# Contributing to Metalcyte

Bug reports, feature requests and pull requests are welcome.

## Reporting a bug

Open an issue with the Metalcyte version (`mc.__version__`), the macOS version, the chip, a minimal
example that reproduces the problem and the full error message. Say whether the problem also
happens on the CPU, with `METALCYTE_DEVICE=cpu` set in the environment.

## Development setup

You need a Rust toolchain (stable, 1.88 or newer), Python 3.11 or newer and the Xcode command line
tools.

```bash
git clone https://github.com/huulocmedvnu/metalcyte
cd metalcyte
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release --extras dev,reference,plot
```

## Before opening a pull request

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --release
.venv/bin/ruff check python tests && .venv/bin/ruff format --check python tests
METALCYTE_TEST_DEVICE=auto .venv/bin/pytest
```

New numerical code needs a test that compares its output with a reference implementation and
states the tolerance, as described in [docs/VALIDATION.md](docs/VALIDATION.md). Each claim about speed
or memory needs a script in `benches/` that reproduces it.

## Repository layout

- `crates/metalcyte-core`: algorithms in Rust
- `crates/metalcyte-gpu`: Metal kernels
- `crates/metalcyte-py`: PyO3 bindings
- `python/metalcyte`: the Python interface
- `tests`, `benches`, `docs`
