"""Tests for the FastAPI service (``science_rag.service``).

Covers the tornado -> FastAPI port and the optional ``dbc_pyutils`` gate
(``science_rag.service._dbc_optional``) without requiring ``dbc_pyutils`` to
be installed: the "available" branch is exercised via monkeypatched fakes
rather than the real package, so this test runs the same whether or not the
``dbc`` dependency group is installed.
"""

from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from science_rag.service import _dbc_optional
from science_rag.service.start import create_app


class StubGenerator:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class StubAgenticRAG:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.generator = StubGenerator()


class StubGraph:
    """Stands in for ``AgenticGraph.graph``: ``ainvoke`` returns an async-iterable
    ``output`` of raw string deltas, matching ``AgentStreamingGenerator``."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.last_state = None

    async def ainvoke(self, state):
        self.last_state = state
        tokens = self.tokens

        async def _stream():
            for token in tokens:
                yield token

        return {"output": _stream()}


class StubAgenticGraph:
    def __init__(self, type, model):
        self.type = type
        self.model = model
        self.graph = StubGraph(["Hello", ", ", "world!"])


def _args(**overrides):
    defaults = dict(
        embedding_model_path="fake-embedding-model",
        faiss_path="fake-faiss-path",
        article_index_path=None,
        validator_model_path=None,
        graph_type="service",
        ab_id=1,
        port=5000,
        verbose=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def stub_rag_classes(monkeypatch):
    """Avoid loading real embedding/torch/validator models in tests."""
    monkeypatch.setattr("science_rag.service.start.AgenticRAG", StubAgenticRAG)
    monkeypatch.setattr("science_rag.service.start.AgenticGraph", StubAgenticGraph)


@pytest.fixture
def dbc_unavailable(monkeypatch):
    monkeypatch.setattr(_dbc_optional, "DBC_AVAILABLE", False)
    monkeypatch.setattr(_dbc_optional, "create_instance_id", None)
    monkeypatch.setattr(_dbc_optional, "build_info", None)
    monkeypatch.setattr(_dbc_optional, "Statistics", None)
    monkeypatch.setattr(_dbc_optional, "install_base_handler", None)
    monkeypatch.setattr(_dbc_optional, "PrometheusMiddleware", None)
    monkeypatch.setattr(_dbc_optional, "metrics_endpoint", None)


class FakeStatistics:
    def __init__(self, name):
        self.name = name

    def describe(self):
        return {"name": self.name, "count": 0}


class FakeBuildInfo:
    @staticmethod
    def get_info(package):
        return {"build_number": "42", "git_revision": "deadbeef", "version": "1.2.3"}


class FakePrometheusMiddleware:
    def __init__(self, app, excluded_paths=frozenset()):
        self.app = app
        self.excluded_paths = frozenset(excluded_paths)

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


async def _fake_metrics_endpoint(request: Request):
    return PlainTextResponse("# fake prometheus metrics\n", media_type="text/plain; version=0.0.4")


@pytest.fixture
def dbc_available(monkeypatch):
    """Simulate an installed ``dbc_pyutils`` with fakes (real package isn't
    installed in this dev environment by default — it only ships in the
    ``dbc`` dependency group)."""
    monkeypatch.setattr(_dbc_optional, "DBC_AVAILABLE", True)
    monkeypatch.setattr(_dbc_optional, "create_instance_id", lambda num_digits=8: "fake-instance-id")
    monkeypatch.setattr(_dbc_optional, "build_info", FakeBuildInfo)
    monkeypatch.setattr(_dbc_optional, "Statistics", FakeStatistics)
    monkeypatch.setattr(_dbc_optional, "install_base_handler", lambda app: None)
    monkeypatch.setattr(_dbc_optional, "PrometheusMiddleware", FakePrometheusMiddleware)
    monkeypatch.setattr(_dbc_optional, "metrics_endpoint", _fake_metrics_endpoint)


def test_status_without_dbc_pyutils(stub_rag_classes, dbc_unavailable):
    app = create_app(_args())
    with TestClient(app) as client:
        response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_route_absent_without_dbc_pyutils(stub_rag_classes, dbc_unavailable):
    app = create_app(_args())
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 404


def test_status_with_dbc_pyutils(stub_rag_classes, dbc_available):
    app = create_app(_args(ab_id=7))
    with TestClient(app) as client:
        response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["build"] == "42"
    assert body["git"] == "deadbeef"
    assert body["version"] == "1.2.3"
    assert body["instance-id"] == "fake-instance-id"
    assert body["ab-id"] == 7
    assert "mem-usage" in body
    assert body["statistics"] == [{"name": "query", "count": 0}]


def test_metrics_route_present_with_dbc_pyutils(stub_rag_classes, dbc_available):
    app = create_app(_args())
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "fake prometheus metrics" in response.text


def test_chat_completions_non_streaming(stub_rag_classes, dbc_unavailable):
    app = create_app(_args())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-26b-a4b-it",
                "stream": False,
                "messages": [{"role": "user", "content": "Hej"}],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gemma-4-26b-a4b-it"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello, world!",
    }
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming(stub_rag_classes, dbc_unavailable):
    app = create_app(_args())
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "Hej"}]},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
    assert body.count('"object": "chat.completion.chunk"') == 4  # 3 tokens + final finish_reason frame
    assert body.endswith("data: [DONE]\n\n")


def test_lifespan_closes_generator_on_shutdown(stub_rag_classes, dbc_unavailable):
    app = create_app(_args())
    with TestClient(app):
        pass
    assert app.state.agentic_graph.model.generator.closed is True
