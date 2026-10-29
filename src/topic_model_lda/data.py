"""A bundled document collection with latent themes for demonstration.

The documents are drawn from three clear themes (space, cooking, finance). A good
topic model should recover these themes as distinct topics whose top words are
semantically coherent, which the coherence metric measures.
"""

from __future__ import annotations

import random

_THEMES: dict[str, tuple[str, ...]] = {
    "space": ("rocket", "orbit", "planet", "galaxy", "astronaut", "launch", "mission", "satellite"),
    "cooking": ("recipe", "oven", "flour", "sugar", "bake", "sauce", "dish", "kitchen"),
    "finance": ("market", "stock", "invest", "profit", "trade", "bank", "price", "portfolio"),
}


def documents(*, count: int = 60, seed: int = 0) -> list[str]:
    """Generate documents, each sampling words from a single theme."""
    rng = random.Random(seed)
    themes = list(_THEMES.values())
    texts: list[str] = []
    for _ in range(count):
        words = rng.choice(themes)
        length = rng.randint(8, 14)
        texts.append(" ".join(rng.choice(words) for _ in range(length)))
    return texts


def theme_words() -> dict[str, tuple[str, ...]]:
    """Return the theme vocabularies (for checking topic recovery)."""
    return dict(_THEMES)
