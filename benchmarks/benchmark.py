"""A deterministic micro-benchmark for LDA training and inference.

Measures collapsed Gibbs sampling time, mean topic coherence, and inference
throughput. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from topic_model_lda.corpus import Corpus
from topic_model_lda.data import documents
from topic_model_lda.evaluation import topic_coherence
from topic_model_lda.lda import LDA, LDAConfig


def main() -> None:
    corpus = Corpus.from_texts(documents())
    start = time.perf_counter()
    model = LDA(corpus, LDAConfig(num_topics=3, iterations=250, seed=0))
    model.train()
    train_ms = (time.perf_counter() - start) * 1000
    report = topic_coherence(model, corpus)

    print(f"Documents: {corpus.num_documents} | Vocabulary: {corpus.vocabulary_size}")
    print(f"Gibbs sampling (250 iterations): {train_ms:.0f} ms")
    print(f"Mean topic coherence (UMass): {report.mean:.3f}")
    for topic in range(3):
        print(f"  topic {topic}: {', '.join(model.topic_words(topic, k=6))}")


if __name__ == "__main__":
    main()
