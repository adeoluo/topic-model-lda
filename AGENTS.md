# AGENTS.md

Working notes for coding agents and contributors in **Topic Model (LDA)**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Latent Dirichlet allocation, built from scratch: collapsed Gibbs sampling for topic discovery, with a Rust core that reproduces the Python sampler exactly, and UMass topic-coherence evaluation.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/gibbs.rs`: Collapsed Gibbs sampling, step for step the same as the Python sampler.
- `rust/src/lib.rs`: Rust core of topic-model-lda: collapsed Gibbs sampling for latent Dirichlet allocation.
- `rust/src/mt.rs`: CPython's `random.Random`: MT19937 with `init_by_array` seeding.
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `topic_model_lda_rs` extension module.
- `src/topic_model_lda/__init__.py`: Topic Model LDA - latent Dirichlet allocation with collapsed Gibbs sampling.
- `src/topic_model_lda/cli.py`: Command line interface for the LDA topic model.
- `src/topic_model_lda/corpus.py`: Corpus types: tokenization and an integer-indexed document collection.
- `src/topic_model_lda/data.py`: A bundled document collection with latent themes for demonstration.
- `src/topic_model_lda/evaluation.py`: Topic-coherence evaluation.
- `src/topic_model_lda/lda.py`: Latent Dirichlet allocation with collapsed Gibbs sampling, from scratch.
- `src/topic_model_lda/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/topic_model_lda/service.py`: A standard-library HTTP service exposing the trained topic model.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it) |
| `make parity` | Rust vs Python parity tests |
| `make bench-compare` | Python vs Rust training speed and a large real-text run |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the application entrypoint |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
