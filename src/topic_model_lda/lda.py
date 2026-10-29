"""Latent Dirichlet allocation with collapsed Gibbs sampling, from scratch.

LDA is a generative model: each document is a mixture of topics, and each topic
is a distribution over words. Given only the documents, the latent topic
assignment of every word token is inferred. Collapsed Gibbs sampling integrates
out the topic-word and document-topic distributions and samples each token's
topic from its full conditional, which depends on the current counts:

    p(topic = k | ...) proportional to
        (n_dk + alpha) * (n_kw + beta) / (n_k + V * beta)

where n_dk counts topic k in document d, n_kw counts word w in topic k, and n_k
counts all words in topic k. After sampling converges, the counts give the
topic-word and document-topic distributions. Everything is pure Python.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from topic_model_lda.corpus import Corpus


@dataclass(frozen=True)
class LDAConfig:
    """Hyperparameters for LDA training."""

    num_topics: int = 5
    alpha: float = 0.1
    beta: float = 0.01
    iterations: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_topics < 2:
            raise ValueError("num_topics must be at least 2")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")


class LDA:
    """A latent Dirichlet allocation model trained by collapsed Gibbs sampling."""

    def __init__(self, corpus: Corpus, config: LDAConfig) -> None:
        self._corpus = corpus
        self._config = config
        self._rng = random.Random(config.seed)
        k = config.num_topics
        v = corpus.vocabulary_size
        d = corpus.num_documents
        self._topic_word = [[0 for _ in range(v)] for _ in range(k)]
        self._topic_total = [0 for _ in range(k)]
        self._doc_topic = [[0 for _ in range(k)] for _ in range(d)]
        self._assignments: list[list[int]] = []
        for doc_index, document in enumerate(corpus.documents):
            doc_assignments = []
            for word in document:
                topic = self._rng.randrange(k)
                doc_assignments.append(topic)
                self._topic_word[topic][word] += 1
                self._topic_total[topic] += 1
                self._doc_topic[doc_index][topic] += 1
            self._assignments.append(doc_assignments)

    def train(self) -> None:
        """Run collapsed Gibbs sampling for the configured number of iterations."""
        for _ in range(self._config.iterations):
            self._sweep()

    def _sweep(self) -> None:
        k = self._config.num_topics
        v = self._corpus.vocabulary_size
        alpha, beta = self._config.alpha, self._config.beta
        for doc_index, document in enumerate(self._corpus.documents):
            for position, word in enumerate(document):
                old_topic = self._assignments[doc_index][position]
                self._topic_word[old_topic][word] -= 1
                self._topic_total[old_topic] -= 1
                self._doc_topic[doc_index][old_topic] -= 1

                weights = [
                    (self._doc_topic[doc_index][topic] + alpha)
                    * (self._topic_word[topic][word] + beta)
                    / (self._topic_total[topic] + v * beta)
                    for topic in range(k)
                ]
                new_topic = self._sample(weights)

                self._assignments[doc_index][position] = new_topic
                self._topic_word[new_topic][word] += 1
                self._topic_total[new_topic] += 1
                self._doc_topic[doc_index][new_topic] += 1

    def _sample(self, weights: Sequence[float]) -> int:
        total = sum(weights)
        threshold = self._rng.random() * total
        cumulative = 0.0
        for topic, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                return topic
        return len(weights) - 1

    def topic_words(self, topic: int, *, k: int = 8) -> list[str]:
        """Return the ``k`` highest-probability words for a topic."""
        counts = self._topic_word[topic]
        ranked = sorted(range(len(counts)), key=lambda word: counts[word], reverse=True)
        return [self._corpus.word(word) for word in ranked[:k] if counts[word] > 0]

    def document_topics(self, doc_index: int) -> list[float]:
        """Return the topic distribution for a document."""
        counts = self._doc_topic[doc_index]
        total = sum(counts) + self._config.num_topics * self._config.alpha
        return [(count + self._config.alpha) / total for count in counts]

    @property
    def topic_word_counts(self) -> list[list[int]]:
        return [list(row) for row in self._topic_word]
