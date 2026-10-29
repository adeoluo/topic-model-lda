# Scaling and Design Tradeoffs - Topic Model (LDA)

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process trains LDA by collapsed Gibbs sampling over an in-memory collection and serves
topic and inference queries over a standard-library HTTP service. Deterministic for a fixed
seed.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Infer CPU, global interpreter lock | One replica per core |
| Large collection | Pure-Python Gibbs sampling | Train offline; load counts; sparse sampling |
| Many topics | Per-token sampling cost | Sparse or alias-method sampling |
| Streaming documents | Batch training | Online variational inference |
| Many replicas | Repeated startup training | Train once; ship the model as an artifact |

## Substantial tradeoffs

- **Collapsed Gibbs sampling vs variational inference.** Gibbs sampling is simple and
  transparent and mixes well on small collections, at the cost of being stochastic and iterative;
  variational inference is deterministic and often faster but more complex to derive.
- **Fixed vs inferred topic count.** A fixed topic count is simple but must be chosen;
  nonparametric variants (hierarchical Dirichlet processes) infer it at far greater complexity.
- **Intrinsic vs extrinsic coherence.** Intrinsic UMass coherence is self-contained and offline;
  extrinsic measures use an external corpus or embeddings for a possibly better correlation with
  human judgment at the cost of a dependency.
- **Dense vs sparse sampling.** Dense per-token sampling is simple; sparse or alias-method
  sampling scales to many topics at the cost of more machinery.

## What would change under real load

Train offline and ship the counts, serve from process-per-core replicas, and adopt sparse
sampling or online variational inference for large or streaming collections. The generative
model and the sampling step are unchanged; only the inference engine and data plumbing change.
