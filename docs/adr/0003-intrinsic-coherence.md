# ADR-0003: Intrinsic topic coherence for evaluation

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Topic quality must be measured without labels. The natural question is whether a topic's top
words are semantically related.

## Decision

Use UMass topic coherence, computed from the training documents themselves: sum the log of the
smoothed co-document frequency ratio over the top word pairs of each topic. No external corpus
is required.

## Alternatives considered

- **Extrinsic coherence (UCI, word2vec).** Needs an external reference corpus or embeddings,
  adding a dependency and data.
- **Human judgment.** The gold standard but not automatable or reproducible.

## Consequences

- Evaluation is self-contained, deterministic, and offline.
- UMass is non-positive; values closer to zero are better.
- It rewards within-corpus co-occurrence, which aligns with recovering the latent themes.
