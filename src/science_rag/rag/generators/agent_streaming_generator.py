#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.rag.generators.agent_streaming_generator` -- agent_streaming_generator model

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
import json
import logging
import os
import random

import aiohttp

from science_rag.config import (
    DEFAULT_MODEL,
    END_TURN_USER,
    GEMMA_4_26B,
    START_TURN_MODEL,
    START_TURN_USER,
    THOUGHT_STUB,
)
from science_rag.rag.rag import Generator, Reference
from science_rag.tools.llm_formatting import (
    clean_sources_from_messages,
    load_tokenizers,
    select_model_function,
    build_request_body,
    build_output_chunk,
)

roles_to_ignore = ["resetter", "summarizer"]

logger = logging.getLogger(__name__)

# These cannot always be easily derived from the tokenizer, so add more manually if trying out a new model


class AgentStreamingGenerator(Generator):
    def __init__(self, use_ceph=False):
        self.streaming_delays = [0.01, 0.02, 0.03]
        self.model_endpoints = {
            GEMMA_4_26B: os.environ.get(
                "SCIENCE_RAG_VLLM_URL",
                "http://vllm-gemma-4-26b-a4b-1-0.ai-staging.svc.cloud.dbc.dk/v1/chat/completions",
            ),
        }
        self.request_model_names = {
            GEMMA_4_26B: os.environ.get("SCIENCE_RAG_VLLM_MODEL", ""),
        }

        self.tokenizers = load_tokenizers(list(self.model_endpoints.keys()), use_ceph=use_ceph)
        self.model_output_function = None
        self.system_message = (
            "Du er Science-RAG. Du hjælper med søgninger et katalog af PDF'er. Du svarer altid på dansk."
        )
        self.missing_reference_prompt = """
Brugeren har stillet et spørgsmål du ikke kan finde nogen kilder om.
Forklar brugeren at du ikke kan finde svaret på spørgsmålet, og bed dem om at omformulere det.
"""
        self.session = aiohttp.ClientSession()

    async def generate(
        self,
        references: list[Reference],
        input: list[dict],
        prompt_template: str = None,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"parsed_references: {references}")
        self.model_output_function = select_model_function(prompt_template["model"])
        # remove sources from output if generated
        logger.info(f"Replying as {prompt_template['name']} with model {prompt_template['model']}")
        messages = input["input"]
        cleaned_messages = clean_sources_from_messages(messages)

        max_new_tokens = 1000 if prompt_template["name"] not in {"ROUTER", "REFORMULATOR"} else 200
        async for chunk in self.llm_generate(
            {
                "messages": cleaned_messages,
                "model": self.request_model_names.get(prompt_template["model"], prompt_template["model"]),
                "stream": True,
                "model_name": prompt_template["model"],
                "prompt_template": prompt_template["prompt"],
                "agent_type": prompt_template["name"],
                "max_tokens": max_new_tokens,
            },
            references,
        ):
            yield chunk

    async def async_llm_format(self, msgs, model_name, prompt_template, agent_type, parsed_references):
        await asyncio.sleep(0)
        return self.llm_format(msgs, model_name, prompt_template, agent_type, parsed_references)

    def __format_messages(
        self,
        messages: list[str],
        model_name: str,
        use_bos: bool = False,
        ignore_role: str = None,
    ):
        if ignore_role:
            messages = [msg for msg in messages if not msg["role"] == ignore_role]
        formatted_chat_history = self.tokenizers[model_name].apply_chat_template(messages, tokenize=False)
        if not use_bos:
            formatted_chat_history = formatted_chat_history[len(self.tokenizers[model_name].bos_token) :]
        return formatted_chat_history

    def llm_format(self, msgs, model_name, prompt_template, agent_type, parsed_references):
        # Set start token and add system prompt
        result = self.tokenizers[model_name].bos_token + START_TURN_USER[model_name]
        result += f"{self.system_message}"

        # format input for agents that need documents as context
        if agent_type in {"RAG", "FOLLOW_UP"}:
            # Only generate something of there are references.
            if parsed_references:
                # Add prompt template, set through input
                result += prompt_template
                # Format references
                result += "Dokumenter:" + (
                    ". ".join([f"{ref.article_headline}: {ref.text[:500]}" for ref in parsed_references]) + ""
                )
                # End "system" instructions.
                result += END_TURN_USER[model_name]
                # Format chat history
                result += self.__format_messages(msgs, model_name, use_bos=False)
            else:
                # If no references, use missing reference prompt
                result += self.missing_reference_prompt
                result += END_TURN_USER[model_name]
        # format agents that need the chathistory as context
        elif agent_type in {"REFORMULATOR", "ROUTER"}:
            result += prompt_template
            # chat history is used as context here. do not format it as instructions.
            result += "Chat-historik:\n\n"
            for msg in msgs:
                if msg["role"] == "assistant" or msg["role"] == "user":
                    result += "Bruger: " if msg["role"] == "user" else "Model: "
                    result += f"\n{msg['content']}"
            result += END_TURN_USER[model_name]
        else:
            result += prompt_template
            result += END_TURN_USER[model_name]
            result += self.__format_messages(msgs, model_name, use_bos=False)
        # Finally, add model start token at end of prompt
        result += START_TURN_MODEL[model_name] + THOUGHT_STUB[model_name]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Input for agent {agent_type}:{str(result)}")
        return [{"role": "user", "content": result}]

    def decode(self, input, stream=False):
        try:
            return input.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.debug(f"UnicodeDecodeError: {e}")
            return input.decode("utf-8", errors="ignore")

    async def reference_generator(self, references: list[Reference]):
        for ref in references:
            yield json.dumps(build_output_chunk(DEFAULT_MODEL, "\n"))
            yield json.dumps(build_output_chunk(DEFAULT_MODEL, "\n"))
            tokens = [f"- [{ref.article_headline}]({ref.article_link})"]
            for token in tokens:
                yield json.dumps(build_output_chunk(DEFAULT_MODEL, token))

    async def async_reference_generator(self, parsed_references: list):
        yield json.dumps(build_output_chunk(DEFAULT_MODEL, "\n"))
        await asyncio.sleep(0.01)
        yield json.dumps(build_output_chunk(DEFAULT_MODEL, "\n"))
        await asyncio.sleep(0.01)
        yield json.dumps(build_output_chunk(DEFAULT_MODEL, "**Kilder**"))
        await asyncio.sleep(0.01)
        yield json.dumps(build_output_chunk(DEFAULT_MODEL, ":"))
        await asyncio.sleep(0.01)
        yield json.dumps(build_output_chunk(DEFAULT_MODEL, "\n"))
        await asyncio.sleep(0.01)
        yield json.dumps(build_output_chunk(DEFAULT_MODEL, "\n"))
        async for ref in self.reference_generator(parsed_references):
            await asyncio.sleep(random.choice(self.streaming_delays))
            yield ref

    async def llm_generate(self, input, parsed_references):
        fetch_options = {
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            },
            "method": "POST",
            "redirect": "manual",
        }

        inputs = await asyncio.gather(
            self.async_llm_format(
                input["messages"],
                input["model_name"],
                input["prompt_template"],
                input["agent_type"],
                parsed_references,
            )
        )
        request_body = {
            "messages": inputs[0],
            "model": input["model"],
            "stream": input["stream"],
            "max_tokens": input["max_tokens"],
            "temperature": 0.1,
        }
        request_body_str = json.dumps(build_request_body(input["model_name"], request_body))
        endpoint_url = self.model_endpoints[input["model_name"]]

        async with self.session.post(
            endpoint_url,
            headers=fetch_options["headers"],
            data=request_body_str,
        ) as response:
            if response.status >= 400:
                error_body = await response.text()
                logger.info(f"Model endpoint returned status {response.status}: {error_body}")
                return
            stream_buffer = ""

            async for chunk in response.content.iter_chunked(1024):
                if chunk:
                    decoded_value = self.decode(chunk, stream=True)
                    stream_buffer += decoded_value
                    lines = stream_buffer.split("\n")
                    stream_buffer = lines.pop()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        payload = line[len("data:") :].strip() if line.startswith("data:") else line
                        if payload == "[DONE]":
                            continue
                        try:
                            obj = json.loads(payload)
                            token = self.model_output_function(obj)
                            if token == self.tokenizers[input["model_name"]].eos_token and parsed_references:
                                continue
                            yield f"data: {json.dumps(obj)}\n\n"

                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            logger.info(f"Error during streaming: {e}")

            residual = stream_buffer.strip()
            if residual:
                payload = residual[len("data:") :].strip() if residual.startswith("data:") else residual
                if payload and payload != "[DONE]":
                    try:
                        obj = json.loads(payload)
                        token = self.model_output_function(obj)
                        if not (token == self.tokenizers[input["model_name"]].eos_token and parsed_references):
                            yield f"data: {json.dumps(obj)}\n\n"
                    except Exception:
                        pass

        # filter references so that no two references have the same article_link
        if parsed_references and input["agent_type"] in {"RAG", "FOLLOW_UP"}:
            seen_links = set()
            filtered_references = []
            for ref in parsed_references:
                if ref.article_link not in seen_links:
                    seen_links.add(ref.article_link)
                    filtered_references.append(ref)

            async for ref in self.async_reference_generator(filtered_references):
                yield f"data: {ref}\n\n"
