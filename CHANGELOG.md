# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.0] 2026-11-02

### Features

- Build the Rust sampler into the package (#4)

## [0.1.0] 2026-11-02

### Features

- Add LDA collapsed Gibbs sampler, coherence evaluation, and CLI
- Add HTTP service with observability, benchmarks, and documentation
- Rust Gibbs sampler with exact parity, and a C4 benchmark against gensim (#3)

### Bug fixes

- Keep Unicode tokens and stop /infer underflowing to zero (#9)

### Documentation

- Use plain, factual wording (#8)
- Standard README structure and language framing (#11)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Scaffold topic-model-lda project
- Bump GitHub Actions to current major versions
- Remove Dependabot configuration
