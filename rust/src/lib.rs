//! Rust core of topic-model-lda: collapsed Gibbs sampling for latent Dirichlet allocation.
//!
//! The sampler reproduces `topic_model_lda.lda.LDA` exactly. It draws from a port of
//! CPython's Mersenne Twister (`random.Random`, including its integer seeding,
//! `random()` and `randrange()`), computes the same weights in the same order, and
//! totals them the way the running interpreter's `sum()` does (plain addition before
//! Python 3.12, compensated summation from 3.12). Given the same corpus, seed, and
//! iteration count, the topic assignments are identical; `tests/test_rust_parity.py`
//! checks this.

pub mod gibbs;
pub mod mt;

#[cfg(feature = "python")]
mod python;

pub use gibbs::Sampler;
pub use mt::PyRandom;
