"""Topic-coherence evaluation.

A topic is coherent when its top words tend to co-occur in the same documents.
The standard intrinsic metric is UMass coherence: for the top words of a topic,
sum the log of the smoothed ratio of the co-document frequency of each word pair
to the document frequency of the earlier word. Higher (closer to zero) is better.
This needs no external corpus - just the training documents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from topic_model_lda.corpus import Corpus
from topic_model_lda.lda import LDA


def _document_frequencies(corpus: Corpus) -> tuple[list[int], dict[tuple[int, int], int]]:
    single = [0 for _ in range(corpus.vocabulary_size)]
    pair: dict[tuple[int, int], int] = {}
    for document in corpus.documents:
        present = set(document)
        for word in present:
            single[word] += 1
        ordered = sorted(present)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                key = (ordered[i], ordered[j])
                pair[key] = pair.get(key, 0) + 1
    return single, pair


@dataclass(frozen=True)
class CoherenceReport:
    """Per-topic and mean UMass coherence."""

    per_topic: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.per_topic) / len(self.per_topic) if self.per_topic else 0.0

    def format(self) -> str:
        return f"mean topic coherence (UMass): {self.mean:.3f} (higher is better)"


def topic_coherence(model: LDA, corpus: Corpus, *, top_k: int = 6) -> CoherenceReport:
    """Compute UMass coherence for each topic from the training documents."""
    single, pair = _document_frequencies(corpus)
    index = {word: position for position, word in enumerate(corpus.vocabulary)}
    scores: list[float] = []
    for topic in range(len(model.topic_word_counts)):
        words = [index[w] for w in model.topic_words(topic, k=top_k) if w in index]
        coherence = 0.0
        for i in range(1, len(words)):
            for j in range(i):
                higher, lower = words[i], words[j]
                key = (min(higher, lower), max(higher, lower))
                co = pair.get(key, 0)
                coherence += math.log((co + 1) / (single[lower] + 1))
        scores.append(coherence)
    return CoherenceReport(per_topic=tuple(scores))
