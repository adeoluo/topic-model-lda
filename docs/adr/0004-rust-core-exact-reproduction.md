# ADR 0004: A Rust core that reproduces the Python sampler exactly

## Status

Accepted

## Context

Collapsed Gibbs sampling visits every token on every sweep and computes one weight per
topic. In interpreted Python this runs at about 350,000 tokens per second, so real corpora
take minutes per sweep. A compiled sampler is the obvious fix, but a sampler that only
approximates the Python one would make the Python results unverifiable against it.

## Decision

Port the sampler to Rust (`rust/`), exposed through `LDAConfig(backend=...)`:

* Random numbers come from a port of CPython's `random.Random`: MT19937 with CPython's
  integer seeding, `random()` built from two 32-bit draws, and `randrange()` by rejection on
  `bit_length` bits. Unit tests pin values printed by CPython.
* Weights are computed in the same order with the same float operations. Their total uses
  plain addition before Python 3.12 and Neumaier compensated summation from 3.12, matching
  the interpreter's `sum()`, and the extension is told which applies.
* `backend="auto"` uses Rust when the extension is installed.

## Consequences

* The parity tests compare assignments and counts token for token, so any divergence fails CI.
* 37x faster on C4 text. The chain stays sequential: a parallel sampler would be faster on
  multi-core machines but could not match the Python sampler, so it is a separate roadmap item.

## Update: the Rust core is compiled into every install

The package now builds with maturin, so installing it compiles the Rust sampler into the
package and `backend="auto"`, the default, always uses it. The Python sampler stays as the
reference the parity tests compare against.
