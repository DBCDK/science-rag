#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.streaming_endpoint` -- endpoint for streaming RAG

==================
Streaming Endpoint
==================

Endpoint for streaming RAG
"""

import logging
import asyncio
import json
import time
import uuid
import tornado.web as tw
from dbc_pyutils import create_instance_id
from dbc_pyutils import Statistics
from dbc_pyutils import build_info
from dbc_pyutils import setup_logging
from dbc_pyutils import StatusHandler
from dbc_pyutils import PrometheusMixIn
from dbc_pyutils import MetricsHandler
from dbc_pyutils import BaseHandler
from science_rag.rag.langgraph_graphs import AgenticGraph
from science_rag.rag.agent_streaming_rag import AgenticRAG
from science_rag.config import DEFAULT_MODEL
from science_rag.tools.llm_formatting import async_gen_wrapper


INSTANCE_ID = create_instance_id(num_digits=8)
STATS = {"query": Statistics(name="query")}
logger = setup_logging()

path_to_embeddings = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index/embeddings"
path_to_labels = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index/labels.npy"
path_to_JEDs = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds"


class GlyphGateHandler(BaseHandler):
    """
    GlyphGateHandler
    """

    def initialize(self, model, graph_type: str, info, stat_collector):
        """
        Initializes handler
        """
        self.info = info
        self.stat_collector = stat_collector
        self.static_header_content = {
            "build": self.info["build_number"],
            "git": self.info["git"],
            "version": self.info["version"],
        }
        self.model = model
        self.agentic_graph = AgenticGraph(type=graph_type, model=model)

    async def post(self):
        body = json.loads(self.request.body.decode("utf8"))
        self.version = body.get("version", "v1")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model_name = body.get("model", DEFAULT_MODEL)

        # The rag pipeline expects content to be a str, not a list of dicts
        messages = [
            {"role": msg["role"], "content": content["text"]}
            for msg in messages
            for content in (
                msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
            )
            if content.get("type", "") == "text"
        ]

        result = await self.agentic_graph.graph.ainvoke({"input": messages})

        if stream:
            async for chunk in result["output"]:
                self.write(chunk)
                await self.flush()
            self.write("data: [DONE]\n\n")
            await self.flush()
            return

        self.set_header("Content-Type", "application/json; charset=utf-8")
        output = "".join([token async for token in async_gen_wrapper(result["output"], DEFAULT_MODEL)])
        self.write(
            json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        )
        await self.flush()


class MetricsApp(PrometheusMixIn, tw.Application):
    pass


def make_app(model, graph_type):
    info = build_info.get_info("science_rag")
    handlers = [
        (
            r"/v1/chat/completions",
            GlyphGateHandler,
            dict(
                model=model,
                graph_type=graph_type,
                info=info,
                stat_collector=STATS["query"],
            ),
        ),
        (r"/metrics", MetricsHandler),
        (
            "/status",
            StatusHandler,
            dict(
                ab_id=1,
                info=info,
                instance_id=INSTANCE_ID,
                statistics=list(STATS.values()),
            ),
        ),
    ]
    return MetricsApp(handlers)


async def main(args):
    logger.info("Loading model")
    model = AgenticRAG(
        embedding_model=args.embedding_model_path,
        faiss_index=args.faiss_path,
        jed_document_path=args.article_index_path,
        validator_model=args.validator_model_path,
        use_ceph=args.use_ceph,
    )
    logger.info(f"Starting endpoint at port {args.port}")
    app = make_app(model, args.graph_type)
    app.listen(args.port)
    await asyncio.Event().wait()


def cli():
    """Commandline interface"""
    import argparse

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
        default=path_to_embeddings,
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
        help="type of langgraph graph to use. default is service. possible values are service, evaluate_router",
        default="service",
    )
    parser.add_argument(
        "--use-ceph",
        dest="use_ceph",
        action="store_true",
        help="Set this flag if running on Ceph or in dockerfile",
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

    args = parser.parse_args()
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    logger.setLevel(level)
    asyncio.run(main(args))
