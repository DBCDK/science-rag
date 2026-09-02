#!/usr/bin/env python
"""
:mod:`science_rag.tools.llm_formatting -- Formatting tools for llm input and output

==============
LLM Formatting
==============

Functions for formatting input for llm's.
"""


def clean_sources_from_messages(messages: list[dict]):
    cleaned_messages = []
    for message in messages:
        if message["role"] == "assistant":
            message["content"] = message["content"].lower().split("**kilder**:")[0]
            cleaned_messages.append(message)
        else:
            cleaned_messages.append(message)
    return cleaned_messages
