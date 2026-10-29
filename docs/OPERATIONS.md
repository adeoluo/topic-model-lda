# Operations Guide - Topic Model (LDA)

The operational contract of the topic service.

## Running the service

```bash
uv run python -m topic_model_lda serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**. The model is trained once at
startup and then serves stateless queries.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status and topic count (liveness) |
| GET | `/ready` | none | readiness once the model is trained |
| GET | `/metrics` | none | per-route request, error, and latency metrics (Prometheus text) |
| GET | `/topics` | none | top words of each discovered topic |
| GET | `/evaluate` | none | mean and per-topic coherence |
| POST | `/infer` | `{"text"}` | topic distribution of a document |

Errors return JSON `{"error": ...}` with a 4xx status; empty or overlong text is rejected.

## Service level objectives

Measured by `benchmarks/stress.py` on a developer laptop:

| Objective | Target | Observed (single worker) |
|---|---|---|
| Infer latency p50 | < 1 ms | about 0.02 ms |
| Infer latency p99 | < 1 ms | about 0.05 ms |
| Throughput | > 50000 inferences/second/core | about 100000/second |
| Startup training | < 1 s | about 0.13 s |

## Operational constraints

- **Training is a startup cost; inference is cheap.** Gibbs sampling runs once at startup; each
  inference scores a document's words against the learned topic distributions.
- **The infer path is CPU bound and holds the global interpreter lock.** Throughput is flat
  across threads; scale with processes.
- **Requests are bounded**: document text is length-capped.
- **No persistence, no authentication.** Front the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| Incoherent topics | Too few iterations or a noisy corpus | Increase iterations; curate the corpus |
| `400` on infer | Empty or overlong text | Inspect the `error` field |
| Rising p99 under load | Concurrency above one worker per core | Add replicas; cap concurrency |

## Capacity planning

A single core serves tens of thousands of inferences per second. Scale with process-per-core
replicas; the service is stateless.
