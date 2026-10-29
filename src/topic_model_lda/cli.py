"""Command line interface for the LDA topic model."""

from __future__ import annotations

import argparse

from topic_model_lda.corpus import Corpus
from topic_model_lda.data import documents
from topic_model_lda.evaluation import topic_coherence
from topic_model_lda.lda import LDA, LDAConfig


def _train(topics: int, iterations: int, seed: int) -> tuple[LDA, Corpus]:
    corpus = Corpus.from_texts(documents())
    model = LDA(corpus, LDAConfig(num_topics=topics, iterations=iterations, seed=seed))
    model.train()
    return model, corpus


def _cmd_topics(args: argparse.Namespace) -> None:
    model, _ = _train(args.topics, args.iterations, args.seed)
    print(f"Discovered {args.topics} topics:")
    for topic in range(args.topics):
        words = model.topic_words(topic, k=args.k)
        print(f"  topic {topic}: {', '.join(words)}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    model, corpus = _train(args.topics, args.iterations, args.seed)
    report = topic_coherence(model, corpus)
    print(report.format())


def _cmd_serve(args: argparse.Namespace) -> None:
    from topic_model_lda.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="topic-model-lda", description=__doc__)
    parser.add_argument("--topics", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    topics = sub.add_parser("topics", help="print the top words of each discovered topic")
    topics.add_argument("--k", type=int, default=8)

    sub.add_parser("evaluate", help="mean topic coherence of the discovered topics")

    serve = sub.add_parser("serve", help="start the HTTP topic service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {"topics": _cmd_topics, "evaluate": _cmd_evaluate, "serve": _cmd_serve}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
