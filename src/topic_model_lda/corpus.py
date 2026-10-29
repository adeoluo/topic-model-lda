"""Corpus types: tokenization and an integer-indexed document collection."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z]+")
_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "it", "this", "that"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, and drop short stopwords."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


@dataclass(frozen=True)
class Corpus:
    """An integer-indexed collection of tokenized documents."""

    documents: tuple[tuple[int, ...], ...]
    vocabulary: tuple[str, ...]

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def word(self, index: int) -> str:
        return self.vocabulary[index]

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> Corpus:
        tokenized = [tokenize(text) for text in texts]
        vocabulary = sorted({token for document in tokenized for token in document})
        index = {word: position for position, word in enumerate(vocabulary)}
        documents = tuple(tuple(index[token] for token in document) for document in tokenized)
        return cls(documents=documents, vocabulary=tuple(vocabulary))
