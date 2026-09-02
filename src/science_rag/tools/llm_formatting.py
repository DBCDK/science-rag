#!/usr/bin/env python
"""
:mod:`mitcfu_rag.tools.llm_formatting -- Formatting tools for llm input and output

==============
LLM Formatting
==============

Functions for formatting input for llm's and reading the streamed output from llms.
"""

import json

from transformers import AutoTokenizer

from science_rag.config import GEMMA_4_26B, MODEL_MAP


def load_tokenizers(model_names: list[str], use_ceph: bool = False):
    """
    Loads tokenizers for the models used in these endpoints.
    If running this in k8s, load the tokenizers from the ceph mount,
    otherwise load it from huggingface or from cache.
    Make sure you are logged into your huggingface account and have access to the models.
    :param model_names:
    :return tokenizers:
    """
    tokenizers = {}
    for model_name in model_names:
        if use_ceph:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(f"/data/{model_name}")
            # tokenizers[model_name] = AutoTokenizer.from_pretrained("/data/huggingface/gemma-4-26b-a4b-it")
        else:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    return tokenizers


def gen_wrapper(stream, model_name):
    model_function = select_model_function(model_name)
    for item in stream.iter_content(chunk_size=None, decode_unicode=True):
        decoded_item = __decode(item)
        obj = json.loads(decoded_item.replace("data:", ""))
        yield model_function(obj)


async def async_gen_wrapper(stream, model_name):
    model_function = select_model_function(model_name)
    async for item in stream:
        decoded_item = __decode(item)
        obj = json.loads(decoded_item.replace("data:", ""))
        yield model_function(obj)


def select_model_function(model_name):
    if GEMMA_4_26B in model_name.lower():
        return __gemma_gen_wrapper
    raise ValueError(f"Unsupported model name: {model_name}")


def build_request_body(model_name, request_body):
    if GEMMA_4_26B in model_name.lower():
        return __gemma_request_body(request_body)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_output_chunk(model_name, content):
    if GEMMA_4_26B in model_name.lower():
        return __gemma_output_chunk(content)
    raise ValueError(f"Unsupported model name: {model_name}")


def clean_sources_from_messages(messages: list[dict]):
    cleaned_messages = []
    for message in messages:
        if message["role"] == "assistant":
            message["content"] = message["content"].lower().split("**kilder**:")[0]
            cleaned_messages.append(message)
        else:
            cleaned_messages.append(message)
    return cleaned_messages


def __gemma_output_chunk(content):
    return {"choices": [{"delta": {"content": content}}]}


def __gemma_request_body(request_body):
    return {
        "messages": request_body["messages"],
        "model": request_body["model"],
        "stream": request_body["stream"],
        "max_tokens": request_body["max_tokens"],
    }


def __gemma_gen_wrapper(obj):
    token = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
    # Gemma 4 IT ends generation with "<turn|>", not Gemma 3's "<eos>".
    if not token == "<turn|>":
        return token
    return ""


def __decode(input):
    try:
        if isinstance(input, str):
            return input
        else:
            return input.decode("utf-8").strip()
    except UnicodeDecodeError:
        return input.decode("utf-8", errors="ignore")
