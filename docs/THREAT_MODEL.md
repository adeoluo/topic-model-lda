# Threat Model - Topic Model (LDA)

## Assets

- Integrity of the learned topics and the inferred document distributions.

## Trust boundaries

- **Document text is untrusted input to the service.** It is tokenized and scored, never
  executed, and length-bounded.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Unbounded computation | Very long document | Resource use | Text length-bounded |
| Irreproducible training | Unseeded sampling | Flaky topics | Deterministic, seeded Gibbs sampling |
| Misleading topics | Noisy or tiny corpus | Incoherent topics | Coherence is reported, not hidden |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

Topic quality depends on the corpus and the chosen topic count; poor choices yield incoherent
topics, which the coherence metric surfaces rather than hides.
