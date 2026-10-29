"""A dependency-free HTTP service exposing the trained topic model.

The service trains LDA once at startup over the bundled documents and then serves
the discovered topics, infers topics for a new document, and reports coherence. A
pure handler is separated from the socket binding for testing.

Endpoints
---------
``GET  /health``    liveness and topic count.
``GET  /ready``     readiness once the model is trained.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``GET  /topics``    the top words of each discovered topic.
``GET  /evaluate``  mean topic coherence.
``POST /infer``     body ``{"text": str}``; the topic distribution of a document.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from topic_model_lda.corpus import Corpus, tokenize
from topic_model_lda.data import documents
from topic_model_lda.evaluation import topic_coherence
from topic_model_lda.lda import LDA, LDAConfig
from topic_model_lda.observability import ServiceMetrics

JsonObject = dict[str, Any]
_MAX_CHARS = 4000


@dataclass(frozen=True)
class TopicService:
    """Holds the trained LDA model and answers topic queries."""

    model: LDA
    corpus: Corpus
    num_topics: int
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(cls, *, num_topics: int = 3, iterations: int = 250, seed: int = 0) -> TopicService:
        corpus = Corpus.from_texts(documents())
        model = LDA(corpus, LDAConfig(num_topics=num_topics, iterations=iterations, seed=seed))
        model.train()
        return cls(model=model, corpus=corpus, num_topics=num_topics)

    def topics(self, top_k: int) -> JsonObject:
        return {
            "topics": [
                {"topic": topic, "words": self.model.topic_words(topic, k=top_k)}
                for topic in range(self.num_topics)
            ]
        }

    def evaluate(self) -> JsonObject:
        report = topic_coherence(self.model, self.corpus)
        return {
            "mean_coherence": round(report.mean, 6),
            "per_topic": [round(value, 6) for value in report.per_topic],
        }

    def infer(self, text: str) -> JsonObject:
        # Score the document's words against each topic's word distribution. The
        # per-token factors are well below 1, so multiplying them directly
        # underflows to 0.0 for long documents and collapses the distribution.
        # Accumulate in log space and normalize with log-sum-exp instead.
        index = {word: position for position, word in enumerate(self.corpus.vocabulary)}
        counts = self.model.topic_word_counts
        totals = [sum(row) + 1 for row in counts]
        log_scores = [0.0 for _ in range(self.num_topics)]
        for token in tokenize(text):
            if token not in index:
                continue
            word = index[token]
            for topic in range(self.num_topics):
                log_scores[topic] += math.log((counts[topic][word] + 0.01) / totals[topic])
        highest = max(log_scores)
        weights = [math.exp(value - highest) for value in log_scores]
        total = sum(weights) or 1.0
        distribution = [weight / total for weight in weights]
        return {
            "text": text,
            "topic_distribution": [round(value, 6) for value in distribution],
            "top_topic": max(range(self.num_topics), key=lambda topic: distribution[topic]),
        }


def _route(
    method: str, path: str, body: Mapping[str, Any], service: TopicService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "num_topics": service.num_topics}
    if method == "GET" and path == "/topics":
        return 200, service.topics(8)
    if method == "GET" and path == "/evaluate":
        return 200, service.evaluate()
    if method == "POST" and path == "/infer":
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return 400, {"error": "'text' must be a non-empty string"}
        if len(text) > _MAX_CHARS:
            return 400, {"error": f"'text' exceeds {_MAX_CHARS} characters"}
        return 200, service.infer(text)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: TopicService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready", "num_topics": service.num_topics}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: TopicService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": TopicService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"topic-model-lda service listening on http://{host}:{port}")
    server.serve_forever()
