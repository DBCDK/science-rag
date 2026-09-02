#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`science_rag.rag.generators.agent_streaming_generator` -- agent_streaming_generator model

============
AgentStreamingGenerator
============

AgentStreamingGenerator generates an answer based on a list of references and the query.

example of usage:

    as_generator = AgentStreamingGenerator()
    query = "Er der noget om biblioteker?"
    references = ['På visse biblioteker kan du låne fiskestænger, så du kan fange din egen middag efter at have læst om det.']
    response = as_generator(references, query)
    print(f'response: {response}')
"""

import asyncio
import logging
import os
import random

from openai import AsyncOpenAI

from science_rag.config import GEMMA_4_26B
from science_rag.rag.rag import Generator, Reference
from science_rag.tools.message_history import clean_sources_from_messages

roles_to_ignore = ["resetter", "summarizer"]

logger = logging.getLogger(__name__)


def _vllm_base_url(url: str) -> str:
    """AsyncOpenAI wants the `.../v1` base and appends `chat/completions` itself.
    SCIENCE_RAG_VLLM_URL historically points at the full completions URL; strip the
    suffix defensively so both shapes of the env var work."""
    return url.removesuffix("/chat/completions")


def _max_tokens() -> int | None:
    """Read MITCFU_MAX_TOKENS. Unset or empty means no limit (None)."""
    value = os.environ.get("SCIENCE_RAG_MAX_TOKENS")
    return int(value) if value else None


class AgentStreamingGenerator(Generator):
    def __init__(self):
        self.streaming_delays = [0.01, 0.02, 0.03]
        self.clients: dict[str, AsyncOpenAI] = {
            GEMMA_4_26B: AsyncOpenAI(
                base_url=_vllm_base_url(
                    os.environ.get(
                        "SCIENCE_RAG_VLLM_URL",
                        "http://vllm-gemma-4-26b-a4b-1-0.ai-staging.svc.cloud.dbc.dk/v1/chat/completions",
                    )
                ),
                api_key="unused",
            ),
        }
        self.max_tokens = _max_tokens()
        # If SCIENCE_RAG_VLLM_MODEL is unset, fall back to the model key rather than
        # sending a blank "model" field to vLLM.
        self.request_model_names = {
            GEMMA_4_26B: os.environ.get("SCIENCE_RAG_VLLM_MODEL") or GEMMA_4_26B,
        }
        self.system_message = (
            "Du er Science-RAG. Du hjælper med søgninger et katalog af PDF'er. Du svarer altid på dansk."
        )
        self.missing_reference_prompt = """
Brugeren har stillet et spørgsmål du ikke kan finde nogen kilder om.
Forklar brugeren at du ikke kan finde svaret på spørgsmålet, og bed dem om at omformulere det.
"""

    async def aclose(self):
        """Closes every underlying AsyncOpenAI client. Called from the service shutdown hook."""
        for client in self.clients.values():
            await client.close()

    async def generate(
        self,
        references: list[Reference],
        input: list[dict],
        prompt_template: str = None,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"parsed_references: {references}")
        logger.info(f"Replying as {prompt_template['name']} with model {prompt_template['model']}")
        messages = input["input"]
        cleaned_messages = clean_sources_from_messages(messages)

        async for chunk in self.llm_generate(
            {
                "messages": cleaned_messages,
                "model_name": prompt_template["model"],
                "prompt_template": prompt_template["prompt"],
                "agent_type": prompt_template["name"],
            },
            references,
        ):
            yield chunk

        # filter references so that no two references have the same article_link
        if references and prompt_template["name"] in {"RAG", "FOLLOW_UP"}:
            seen_links = set()
            filtered_references = []
            for ref in references:
                if ref.article_link not in seen_links:
                    seen_links.add(ref.article_link)
                    filtered_references.append(ref)

            async for chunk in self.async_reference_generator(filtered_references):
                yield chunk

    def build_messages(
        self,
        msgs: list[dict],
        agent_type: str,
        prompt_template: str,
        parsed_references: list[Reference],
    ) -> list[dict]:
        """Builds the real system/user/assistant message list sent to the model.
        Replaces the old client-side prompt splicing (tokenizer.apply_chat_template +
        hand-rolled special tokens) - the server applies its own chat template now."""
        if agent_type in {"RAG", "FOLLOW_UP"}:
            # Only generate something if there are references.
            if parsed_references:
                system_content = f"{self.system_message}\n{prompt_template}\nDokumenter:" + ". ".join(
                    [f"{ref.article_headline}: {ref.text[:500]}" for ref in parsed_references]
                )
                return [{"role": "system", "content": system_content}, *msgs]
            # No references: use the missing-reference prompt. Preserves today's exact
            # (slightly odd) behavior - the no-references path never includes chat history.
            system_content = f"{self.system_message}\n{self.missing_reference_prompt}"
            return [{"role": "system", "content": system_content}]
        # ROUTER / REFORMULATOR / SIMPLE / FALLBACK: chat history as real messages,
        # not a hand-flattened "Chat-historik:" text block.
        system_content = f"{self.system_message}\n{prompt_template}"
        return [{"role": "system", "content": system_content}, *msgs]

    async def reference_generator(self, references: list[Reference]):
        for ref in references:
            yield "\n\n"
            yield f"- [{ref.article_headline}]({ref.article_link})"

    async def async_reference_generator(self, parsed_references: list):
        yield "\n\n**Kilder**:\n\n"
        async for ref in self.reference_generator(parsed_references):
            await asyncio.sleep(random.choice(self.streaming_delays))
            yield ref

    async def llm_generate(self, input, parsed_references):
        model_key = input["model_name"]
        client = self.clients[model_key]
        messages = self.build_messages(
            input["messages"],
            input["agent_type"],
            input["prompt_template"],
            parsed_references,
        )
        # openai.APIError subclasses (APIConnectionError, APIStatusError, RateLimitError,
        # APITimeoutError) are intentionally not caught here - they propagate up through
        # AgenticRAG.stream_response / AgenticGraph into service.py, which turns them into
        # an SSE error frame instead of silently truncating the stream.
        create_kwargs = {
            "model": self.request_model_names.get(model_key, model_key),
            "messages": messages,
            "stream": True,
            "temperature": 0.1,
        }
        if self.max_tokens is not None:
            create_kwargs["max_tokens"] = self.max_tokens
        stream = await client.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Reasoning: {reasoning}")
            if delta.content:
                yield delta.content
