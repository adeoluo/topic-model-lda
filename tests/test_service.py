"""Tests for the dependency-free HTTP topic service (pure handler layer)."""

from __future__ import annotations

from topic_model_lda.service import TopicService, handle

_SERVICE = TopicService.trained(iterations=200)


def test_health_and_ready() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200 and payload["num_topics"] == 3
    status, payload = handle("GET", "/ready", {}, _SERVICE)
    assert status == 200 and payload["status"] == "ready"


def test_topics() -> None:
    status, payload = handle("GET", "/topics", {}, _SERVICE)
    assert status == 200
    assert len(payload["topics"]) == 3
    assert all(topic["words"] for topic in payload["topics"])


def test_evaluate() -> None:
    status, payload = handle("GET", "/evaluate", {}, _SERVICE)
    assert status == 200
    assert payload["mean_coherence"] < 0.0


def test_infer_assigns_dominant_topic() -> None:
    status, payload = handle(
        "POST", "/infer", {"text": "rocket orbit planet galaxy launch"}, _SERVICE
    )
    assert status == 200
    assert abs(sum(payload["topic_distribution"]) - 1.0) < 1e-6
    assert isinstance(payload["top_topic"], int)


def test_infer_rejects_empty() -> None:
    status, _ = handle("POST", "/infer", {"text": "  "}, _SERVICE)
    assert status == 400


def test_infer_long_document_stays_normalized() -> None:
    # Regression: multiplying per-token factors below 1 underflowed to 0.0 for
    # long documents, producing an all-zero distribution. Log-space scoring keeps
    # the distribution normalized regardless of length.
    long_text = " ".join((list(_SERVICE.corpus.vocabulary)[:6]) * 85)
    status, payload = handle("POST", "/infer", {"text": long_text}, _SERVICE)
    assert status == 200
    distribution = payload["topic_distribution"]
    assert abs(sum(distribution) - 1.0) < 1e-6
    assert any(value > 0.0 for value in distribution)


def test_metrics_and_unknown_route() -> None:
    service = TopicService.trained(iterations=120)
    handle("POST", "/infer", {"text": "stock market invest"}, service)
    text = service.metrics.render_prometheus()
    assert "service_requests_total" in text
    assert 'route="POST /infer"' in text
    status, payload = handle("GET", "/nope", {}, service)
    assert status == 404 and "error" in payload
