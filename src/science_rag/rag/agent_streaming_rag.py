#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`science_rag.rag.agent_streaming_rag` -- agent_streaming_rag model

============
AgenticRAG
============

AgenticRAG is a rag model for mitcfu.
It takes chat messages as an input and returns a response.

example of usage:
    from science_rag.rag.agent_streaming_rag import AgenticRAG

    a_rag = AgenticRAG()
    messages = ["Hej", "Er der noget om miljø?"]

    response = a_rag(messages)
    print(f'response: {response}')
"""

import logging
import asyncio
from typing import Generator, Any
from science_rag.rag.rag import RAG

from science_rag.rag.retrievers.streaming_multilingual_retriever import (
    EmbeddingRetriever,
)

from science_rag.rag.generators.agent_streaming_generator import AgentStreamingGenerator

logger = logging.getLogger(__name__)


class AgenticRAG(RAG):
    def __init__(
        self,
        embedding_model,
        faiss_index,
        jed_document_path,
        validator_model=None,
    ):
        """
        Components used in the RAG model.
        """
        self.parser = None
        self.retriever = EmbeddingRetriever(
            model_path=embedding_model,
            embeddings_path=faiss_index,
            jed_document_path=jed_document_path,
            cross_model_path=validator_model,
        )
        self.reranker = None
        self.generator = AgentStreamingGenerator()
        self.validator = None
        self.summarizer = None
        self.latest_references = []

    async def stream_response(
        self, input: dict[str, Any], prompt_template, limit=5, *args, **kwargs
    ) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        if input.get("agent", "") == "RAG":
            if input.get("reformulated_queries"):
                results = await asyncio.gather(self.retriever.async_rerank_retrieve(input, n=limit))
            else:
                results = await asyncio.gather(self.retriever.async_retrieve(input))
            similarities, references = results[0]
            references = references[:limit]
        elif input.get("agent", "") == "FOLLOW_UP":
            input["FOLLOW_UP"] = True
            if input.get("reformulated_queries"):
                results = await asyncio.gather(self.retriever.async_rerank_retrieve(input, n=limit, follow_up=True))
            else:
                results = await asyncio.gather(self.retriever.async_retrieve(input, follow_up=True))
            similarities, references = results[0]
            similarities = similarities[:limit]
            references = references[:limit]
        else:
            references = None

        async for item in self.generator.generate(references, input, prompt_template):
            yield item

    def get_response(self, messages: list[str], *args, **kwargs) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        # processed_messages = self.parser(messages)
        similarities, references = self.retriever(messages, n=3)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        generated_answer, generated_sources = self.generator(references, messages)

        if not generated_sources or not generated_answer:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"

        validation = self.validator(
            generated_answer + "\n" + " - ".join(generated_sources),
            references,
            messages,
        )

        if validation:
            return generated_answer + "\n" + " - ".join(generated_sources)
        else:
            return "Jeg kan ikke finde svaret på dit spørgsmål. Kan du prøve at stille det på en anden måde?"

    def evaluate(self, messages: list[str]):
        """
        yields response tokens from rag request.
        """

        def gen_wrapper(stream):
            for item in stream:
                for i in item:
                    yield i

        messages = [{"role": "user", "content": messages[0]}]

        similarities, references = self.retriever(messages)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (similarity, reference) in enumerate(zip(similarities, references)):
                logger.debug(f"{i + 1}. similarity: {similarity:.2f} - {reference}\n")

        stream = self.generator(references, messages)
        response = "".join(gen_wrapper(stream))
        return references, response
