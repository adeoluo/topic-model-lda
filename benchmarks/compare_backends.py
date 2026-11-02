"""Python and Rust Gibbs samplers on real web text (C4).

1. Speed: both backends train on the same C4 documents with the same seed; the script
   checks that their topic assignments are identical and reports tokens sampled per
   second.
2. Scale: the Rust sampler trains on a much larger slice of C4 and reports wall time,
   tokens per second, the learned topics, and UMass coherence.
3. Reference (with --gensim): gensim's LdaModel (online variational Bayes) and LdaMulticore
   train on the same documents and vocabulary, and their topics are scored with the same
   UMass function, so time and coherence are directly comparable.

    curl -LO https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz
    make rust-develop
    uv run --no-sync python benchmarks/compare_backends.py --c4 c4-train.00000-of-01024.json.gz
    uv run --no-sync --with gensim python benchmarks/compare_backends.py --c4 ... --gensim
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
from itertools import islice
from pathlib import Path

from topic_model_lda.corpus import Corpus
from topic_model_lda.evaluation import topic_coherence
from topic_model_lda.lda import LDA, LDAConfig, rust_available

ROOT = Path(__file__).resolve().parent.parent


def c4_texts(path: Path, count: int, max_words: int) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = (json.loads(line)["text"] for line in handle)
        return [" ".join(text.split()[:max_words]) for text in islice(rows, count)]


def prune(corpus: Corpus, min_df: int, max_df_ratio: float) -> Corpus:
    """Drop rare and near-universal words, as any practical LDA pipeline does."""
    df = [0] * corpus.vocabulary_size
    for doc in corpus.documents:
        for w in set(doc):
            df[w] += 1
    limit = max_df_ratio * corpus.num_documents
    keep = [w for w in range(corpus.vocabulary_size) if min_df <= df[w] <= limit]
    remap = {w: i for i, w in enumerate(keep)}
    docs = tuple(tuple(remap[w] for w in doc if w in remap) for doc in corpus.documents)
    return Corpus(documents=docs, vocabulary=tuple(corpus.vocabulary[w] for w in keep))


class _WordLists:
    """Adapts plain topic word lists to the interface topic_coherence reads."""

    def __init__(self, topics: list[list[str]]) -> None:
        self._topics = topics
        self.topic_word_counts = [[0] for _ in topics]

    def topic_words(self, topic: int, *, k: int = 8) -> list[str]:
        return self._topics[topic][:k]


def run_gensim(corpus: Corpus, topics: int, passes: int, workers: int) -> list[dict[str, object]]:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, LdaMulticore

    texts = [[corpus.vocabulary[w] for w in doc] for doc in corpus.documents]
    dictionary = Dictionary(texts)
    bow = [dictionary.doc2bow(t) for t in texts]
    rows = []
    for name, build in [
        (
            "gensim LdaModel (1 process)",
            lambda: LdaModel(
                bow, num_topics=topics, id2word=dictionary, passes=passes, random_state=0
            ),
        ),
        (
            f"gensim LdaMulticore ({workers} workers)",
            lambda: LdaMulticore(
                bow,
                num_topics=topics,
                id2word=dictionary,
                passes=passes,
                random_state=0,
                workers=workers,
            ),
        ),
    ]:
        start = time.perf_counter()
        model = build()
        seconds = time.perf_counter() - start
        words = [[w for w, _ in model.show_topic(t, topn=8)] for t in range(topics)]
        coherence = topic_coherence(_WordLists(words), corpus, top_k=8)  # type: ignore[arg-type]
        rows.append(
            {"library": name, "train_s": seconds, "mean_umass": coherence.mean, "topics": words}
        )
    return rows


def timed(corpus: Corpus, backend: str, topics: int, iterations: int) -> tuple[float, LDA]:
    model = LDA(
        corpus, LDAConfig(num_topics=topics, iterations=iterations, seed=0, backend=backend)
    )
    start = time.perf_counter()
    model.train()
    return time.perf_counter() - start, model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--c4", required=True, type=Path)
    ap.add_argument("--speed-docs", type=int, default=1000)
    ap.add_argument("--speed-iterations", type=int, default=5)
    ap.add_argument("--scale-docs", type=int, default=20000)
    ap.add_argument("--scale-iterations", type=int, default=100)
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--gensim", action="store_true", help="also train gensim on the scale corpus")
    ap.add_argument("--gensim-passes", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results.json")
    args = ap.parse_args()
    if not rust_available():
        raise SystemExit("build the extension first: make rust-develop")

    report: dict[str, object] = {
        "machine": f"{platform.machine()} {platform.system()}",
        "cpus": os.cpu_count(),
        "topics": args.topics,
    }

    texts = c4_texts(args.c4, args.speed_docs, args.max_words)
    small = prune(Corpus.from_texts(texts), 5, 0.5)
    tokens = sum(len(d) for d in small.documents)
    py_s, py_model = timed(small, "python", args.topics, args.speed_iterations)
    rs_s, rs_model = timed(small, "rust", args.topics, args.speed_iterations)
    identical = py_model.assignments == rs_model.assignments
    sampled = tokens * args.speed_iterations
    speed = {
        "documents": small.num_documents,
        "vocabulary": small.vocabulary_size,
        "tokens": tokens,
        "iterations": args.speed_iterations,
        "python_s": py_s,
        "rust_s": rs_s,
        "python_tokens_per_s": sampled / py_s,
        "rust_tokens_per_s": sampled / rs_s,
        "speedup": py_s / rs_s,
        "identical": identical,
    }
    report["speed"] = speed
    print(
        "| Backend | Documents | Tokens | Sweeps | Time | Tokens/s | Speedup | Same assignments |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|:---:|")
    print(
        f"| Python | {small.num_documents:,} | {tokens:,} | {args.speed_iterations} | "
        f"{py_s:.1f} s | "
        f"{sampled / py_s:,.0f} | 1x | {'yes' if identical else 'NO'} |"
    )
    print(
        f"| Rust | {small.num_documents:,} | {tokens:,} | {args.speed_iterations} | {rs_s:.2f} s | "
        f"{sampled / rs_s:,.0f} | {py_s / rs_s:.0f}x | {'yes' if identical else 'NO'} |"
    )

    texts = c4_texts(args.c4, args.scale_docs, args.max_words)
    start = time.perf_counter()
    big = prune(Corpus.from_texts(texts), 10, 0.3)
    prep_s = time.perf_counter() - start
    tokens = sum(len(d) for d in big.documents)
    rs_s, model = timed(big, "rust", args.topics, args.scale_iterations)
    coherence = topic_coherence(model, big, top_k=8)
    topics = [model.topic_words(t, k=8) for t in range(args.topics)]
    report["scale"] = {
        "documents": big.num_documents,
        "vocabulary": big.vocabulary_size,
        "tokens": tokens,
        "iterations": args.scale_iterations,
        "prep_s": prep_s,
        "train_s": rs_s,
        "tokens_per_s": tokens * args.scale_iterations / rs_s,
        "mean_umass": coherence.mean,
        "topics": topics,
    }
    print(
        f"\nRust at scale: {big.num_documents:,} documents, {big.vocabulary_size:,} words, "
        f"{tokens:,} tokens, {args.scale_iterations} sweeps in {rs_s:.1f} s "
        f"({tokens * args.scale_iterations / rs_s:,.0f} tokens/s); "
        f"mean UMass coherence {coherence.mean:.3f}"
    )
    for t, words in enumerate(topics):
        print(f"  topic {t:2d}: {' '.join(words)}")
    if args.gensim:
        rows = run_gensim(big, args.topics, args.gensim_passes, max(1, (os.cpu_count() or 2) - 1))
        report["gensim"] = rows
        print("\n| Library | Algorithm | Train time | Mean UMass (top 8, higher is better) |")
        print("|---|---|---:|---:|")
        print(
            f"| topic-model-lda, Rust | collapsed Gibbs, {args.scale_iterations} sweeps | "
            f"{rs_s:.1f} s | {coherence.mean:.2f} |"
        )
        for row in rows:
            print(
                f"| {row['library']} | online variational Bayes, {args.gensim_passes} passes | "
                f"{row['train_s']:.1f} s | {row['mean_umass']:.2f} |"
            )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
