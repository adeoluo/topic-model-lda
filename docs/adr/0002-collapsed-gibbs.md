# ADR-0002: Collapsed Gibbs sampling for inference

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

LDA's posterior over topic assignments is intractable to compute exactly. An approximate
inference method is needed to assign topics to word tokens.

## Decision

Use collapsed Gibbs sampling: integrate out the topic-word and document-topic distributions
analytically and sample each token's topic from its full conditional, which depends only on
integer counts. Iterate until the counts stabilize.

## Alternatives considered

- **Variational inference.** Deterministic and often faster to converge, but more complex to
  derive and implement; Gibbs sampling is simpler and transparent.
- **Uncollapsed Gibbs sampling.** Samples the distributions too, which mixes more slowly and
  needs more bookkeeping.

## Consequences

- Inference runs on integer counts, which is simple and mixes well on small collections.
- Sampling is stochastic; a fixed seed makes it reproducible.
- Convergence is monitored by iteration count rather than a strict criterion, documented as a
  limit.
