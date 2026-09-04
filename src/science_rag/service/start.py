#!/usr/bin/env python3
"""
:mod:`science_rag.service.start` -- FastAPI application + CLI entry point

=====
Start
=====

Entry point for ``streaming-service-science-rag``. Replaces the previous
tornado-based ``science_rag.service`` module; the HTTP contract
(``/v1/chat/completions``, CLI flags) is unchanged.
"""

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from science_rag.config import DEFAULT_MODEL
from science_rag.rag.agent_streaming_rag import AgenticRAG
from science_rag.rag.langgraph_graphs import AgenticGraph
from science_rag.service import _dbc_optional
from science_rag.service.endpoints import router

logger = logging.getLogger(__name__)


def _make_lifespan(agentic_rag: AgenticRAG):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        logger.info("Shutting down, closing LLM clients")
        await agentic_rag.generator.aclose()

    return lifespan


def create_app(args: argparse.Namespace) -> FastAPI:
    """Builds the FastAPI application for the given CLI args.

    ``/status``/``/metrics`` are wired up richly (build info, instance id,
    query statistics, prometheus metrics) when ``dbc_pyutils`` is installed
    (the ``dbc`` dependency group), and degrade to a bare ``{"status": "ok"}``
    with no ``/metrics`` route otherwise.
    """
    if _dbc_optional.DBC_AVAILABLE:
        instance_id = _dbc_optional.create_instance_id(num_digits=8)
        build_info = _dbc_optional.build_info.get_info("science_rag")
        stats = {"query": _dbc_optional.Statistics(name="query")}
    else:
        instance_id = None
        build_info = None
        stats = {}

    logger.info("Loading model")
    agentic_rag = AgenticRAG(
        embedding_model=args.embedding_model_path,
        faiss_index=args.faiss_path,
        jed_document_path=args.article_index_path,
        validator_model=args.validator_model_path,
    )
    agentic_graph = AgenticGraph(type=args.graph_type, model=agentic_rag)

    app = FastAPI(title="science-rag service", lifespan=_make_lifespan(agentic_rag))
    app.state.agentic_graph = agentic_graph
    app.state.default_model = DEFAULT_MODEL
    app.state.dbc_available = _dbc_optional.DBC_AVAILABLE
    app.state.instance_id = instance_id
    app.state.build_info = build_info
    app.state.stats = stats
    app.state.ab_id = args.ab_id

    app.include_router(router)

    if _dbc_optional.DBC_AVAILABLE:
        _dbc_optional.install_base_handler(app)
        app.add_middleware(_dbc_optional.PrometheusMiddleware, excluded_paths={"/metrics", "/status"})
        app.add_api_route("/metrics", _dbc_optional.metrics_endpoint, methods=["GET"])

    return app


def parse_args() -> argparse.Namespace:
    """Commandline interface"""
    port = 5000
    parser = argparse.ArgumentParser(description="query related subject")
    parser.add_argument(
        "embedding_model_path",
        metavar="embedding-model-path",
        help="path to embedding model",
    )
    parser.add_argument(
        "faiss_path",
        metavar="faiss-path",
        help="path to faiss index",
    )
    parser.add_argument(
        "--article_index_path",
        metavar="article-index-path",
        help="path to article index",
        default=None,
    )
    parser.add_argument(
        "--validator-model-path",
        dest="validator_model_path",
        help="path to validator model",
        default=None,
    )
    parser.add_argument(
        "--graph-type",
        dest="graph_type",
        help="type of langgraph graph to use. default is service. only supported value is service.",
        default="service",
    )
    parser.add_argument("-a", "--ab-id", dest="ab_id", help="ab id of service. default is 1", default=1)
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        help=f"port to expose service on. Default is {port}",
        default=port,
    )
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    if _dbc_optional.DBC_AVAILABLE:
        root_logger = _dbc_optional.setup_logging()
    else:
        logging.basicConfig(level=logging.INFO)
        root_logger = logging.getLogger()

    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    app = create_app(args)
    logger.info(f"Starting endpoint at port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=None)


if __name__ == "__main__":
    main()
