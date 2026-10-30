"""Tests for the corpus, LDA sampler, and topic-coherence evaluation."""

from __future__ import annotations

import pytest

from topic_model_lda import (
    LDA,
    Corpus,
    LDAConfig,
    documents,
    theme_words,
    tokenize,
    topic_coherence,
)


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("The rocket is in orbit") == ["rocket", "orbit"]


def test_tokenize_preserves_unicode_digits_underscores() -> None:
    # Regression: [a-z]+ silently dropped accents, scripts, digits, underscores.
    assert tokenize("café covid19 rocket_launch 東京") == [
        "café",
        "covid19",
        "rocket_launch",
        "東京",
    ]


def test_corpus_indexing() -> None:
    corpus = Corpus.from_texts(["rocket orbit", "stock market"])
    assert corpus.num_documents == 2
    assert corpus.vocabulary_size == 4
    assert set(corpus.vocabulary) == {"rocket", "orbit", "stock", "market"}


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        LDAConfig(num_topics=1)
    with pytest.raises(ValueError):
        LDAConfig(beta=0.0)
    with pytest.raises(ValueError):
        LDAConfig(iterations=0)


def _trained(topics: int = 3, iterations: int = 300) -> tuple[LDA, Corpus]:
    corpus = Corpus.from_texts(documents(seed=0))
    model = LDA(corpus, LDAConfig(num_topics=topics, iterations=iterations, seed=0))
    model.train()
    return model, corpus


def test_recovers_coherent_topics() -> None:
    model, _ = _trained()
    # Collect each topic's top words; each should be dominated by one theme.
    themes = theme_words()
    matched_themes = set()
    for topic in range(3):
        words = set(model.topic_words(topic, k=6))
        best_theme = max(themes, key=lambda name: len(words & set(themes[name])))
        overlap = len(words & set(themes[best_theme]))
        assert overlap >= 4  # the topic is dominated by one theme
        matched_themes.add(best_theme)
    assert len(matched_themes) == 3  # all three themes recovered as distinct topics


def test_document_topics_sum_to_one() -> None:
    model, _ = _trained()
    distribution = model.document_topics(0)
    assert abs(sum(distribution) - 1.0) < 1e-9


def test_coherence_is_computed() -> None:
    model, corpus = _trained()
    report = topic_coherence(model, corpus)
    assert len(report.per_topic) == 3
    assert report.mean < 0.0  # UMass coherence is non-positive
