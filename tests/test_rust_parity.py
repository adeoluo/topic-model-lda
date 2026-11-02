"""The Rust sampler must reproduce the Python sampler exactly.

Skipped when ``topic_model_lda_rs`` is not built, unless ``LDA_REQUIRE_RUST=1`` (CI).
"""

from __future__ import annotations

import os
import random

import pytest

from topic_model_lda import lda as lda_module
from topic_model_lda.corpus import Corpus
from topic_model_lda.data import documents
from topic_model_lda.lda import LDA, LDAConfig, rust_available

if os.environ.get("LDA_REQUIRE_RUST") == "1":
    assert rust_available(), "LDA_REQUIRE_RUST=1 but topic_model_lda_rs is not importable"
pytestmark = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


def load_corpus() -> Corpus:
    return Corpus.from_texts(documents())


def train(corpus: Corpus, backend: str, **kwargs: object) -> LDA:
    model = LDA(corpus, LDAConfig(backend=backend, **kwargs))  # type: ignore[arg-type]
    model.train()
    return model


def assert_same(a: LDA, b: LDA) -> None:
    assert a.assignments == b.assignments
    assert a.topic_word_counts == b.topic_word_counts
    for d in range(len(a.assignments)):
        assert a.document_topics(d) == b.document_topics(d)


@pytest.mark.parametrize("seed", [0, 1, 2**33 + 5])
@pytest.mark.parametrize("topics", [2, 3, 7])
def test_bundled_corpus_is_identical(seed: int, topics: int) -> None:
    corpus = load_corpus()
    python = train(corpus, "python", num_topics=topics, iterations=30, seed=seed)
    rust = train(corpus, "rust", num_topics=topics, iterations=30, seed=seed)
    assert rust.backend == "rust" and python.backend == "python"
    assert_same(python, rust)


def test_random_corpus_and_hyperparameters_are_identical() -> None:
    rng = random.Random(4)
    words = [f"w{i}" for i in range(300)]
    texts = [" ".join(rng.choice(words) for _ in range(rng.randint(0, 60))) for _ in range(80)]
    corpus = Corpus.from_texts(texts)
    for alpha, beta in [(0.1, 0.01), (1.0, 0.5), (0.03, 0.2)]:
        python = train(
            corpus, "python", num_topics=12, iterations=15, alpha=alpha, beta=beta, seed=9
        )
        rust = train(corpus, "rust", num_topics=12, iterations=15, alpha=alpha, beta=beta, seed=9)
        assert_same(python, rust)


def test_untrained_state_is_identical() -> None:
    corpus = load_corpus()
    assert_same(LDA(corpus, LDAConfig(backend="python")), LDA(corpus, LDAConfig(backend="rust")))


def test_auto_prefers_rust_and_missing_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = load_corpus()
    assert LDA(corpus, LDAConfig()).backend == "rust"
    monkeypatch.setattr(lda_module, "_EXTENSION", None)
    assert LDA(corpus, LDAConfig()).backend == "python"
    with pytest.raises(RuntimeError, match="not installed"):
        LDA(corpus, LDAConfig(backend="rust"))
