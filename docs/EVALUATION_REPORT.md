# Evaluation Report - Topic Model (LDA)

> Reproduce with `python -m topic_model_lda evaluate` and `topics`. Deterministic and offline.

## Setup

- **Collection:** 60 documents drawn from three latent themes (space, cooking, finance).
- **Model:** LDA with three topics, trained by 250 Gibbs iterations.
- **Metric:** UMass topic coherence, computed from the training documents; higher is better.

## Results

The three discovered topics are each dominated by one theme:

```
topic 0: rocket, orbit, mission, satellite, planet, launch
topic 1: bake, flour, oven, sauce, dish, recipe
topic 2: portfolio, price, trade, bank, stock, profit
```

Mean UMass coherence is about -2.98 (UMass is non-positive; closer to zero is better).

## Interpretation

With no labels, collapsed Gibbs sampling recovers the three latent themes as distinct topics
whose top words are semantically coherent. Every theme becomes its own topic and no topic mixes
themes, confirming that the from-scratch sampler learned the document-topic and topic-word
structure. The coherence score quantifies that the top words genuinely co-occur.

## Reproducibility

Sampling is seeded; the topics and coherence regenerate exactly.
