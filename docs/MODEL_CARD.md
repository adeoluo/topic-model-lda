# System Card - Topic Model (LDA)

## Overview

- **What it is:** a latent Dirichlet allocation topic model trained by collapsed Gibbs
  sampling, with topic discovery, document inference, and coherence evaluation.
- **What it is not:** a supervised classifier; topics are unlabeled and discovered.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** learning how topic models work; discovering themes in a small
  document collection.
- **Primary users:** engineers and students studying unsupervised NLP.

## Out-of-scope / prohibited use

- Not for web-scale collections (pure-Python sampling).
- The number of topics must be chosen.

## Performance

See the README: the model recovers three latent themes as distinct, coherent topics.

## Risks and limitations

- Topic count is a hyperparameter.
- Gibbs sampling is stochastic; results depend on the seed and iteration count.

## Safety mitigations

- Deterministic under a fixed seed; offline; bounded request sizes in the service.

## Environmental / cost footprint

Negligible; training runs in about a tenth of a second on the bundled documents.
