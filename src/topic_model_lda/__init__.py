"""Topic Model LDA - latent Dirichlet allocation with collapsed Gibbs sampling."""

from __future__ import annotations

from topic_model_lda.corpus import Corpus, tokenize
from topic_model_lda.data import documents, theme_words
from topic_model_lda.evaluation import CoherenceReport, topic_coherence
from topic_model_lda.lda import LDA, LDAConfig

__version__ = "0.1.0"

__all__ = [
    "LDA",
    "CoherenceReport",
    "Corpus",
    "LDAConfig",
    "__version__",
    "documents",
    "theme_words",
    "tokenize",
    "topic_coherence",
]
