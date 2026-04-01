#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.rag.langgraph_graphs` -- utilities for handling langgraph

==================
Langgraph graphs
==================

Utilities for handling langgraph
"""

import logging
import asyncio
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict
import re

from science_rag.tools.llm_formatting import async_gen_wrapper
from science_rag.config import (
    RAG_TEMPLATE,
    SIMPLE_TEMPLATE,
    ROUTER_TEMPLATE,
    FALLBACK_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    REFORMULATE_TEMPLATE,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    input: str
    output: str
    agent: str
    prompt_template: str
    endpoint_profile: str


class AgenticGraph:
    def __init__(self, type, model):
        self.route_template = ROUTER_TEMPLATE()
        self.simple_template = SIMPLE_TEMPLATE
        self.rag_template = RAG_TEMPLATE
        self.fallback_template = FALLBACK_TEMPLATE
        self.follow_up_template = FOLLOW_UP_TEMPLATE
        self.reformulate_template = REFORMULATE_TEMPLATE
        self.model = model
        self.graph = self.create_graph(type)

    def create_graph(self, type):
        if type == "service":
            return self.create_service_graph()

    def create_service_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("route", self.route_response)
        workflow.add_node("rag_agent", self.rag_response)
        workflow.add_node("simple_agent", self.simple_response)
        workflow.add_node("fallback_agent", self.fallback_response)
        workflow.add_node("follow_up_agent", self.follow_up_response)

        workflow.add_conditional_edges(
            "route",
            lambda x: x["agent"],
            {
                "RAG": "rag_agent",
                "SIMPLE": "simple_agent",
                "FALLBACK": "fallback_agent",
                "FOLLOW_UP": "follow_up_agent",
            },
        )

        workflow.set_entry_point("route")
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("simple_agent", END)
        workflow.add_edge("fallback_agent", END)
        workflow.add_edge("follow_up_agent", END)

        return workflow.compile()

    async def route_response(self, messages):
        route_result_stream = await self.stream_response(messages, self.route_template)
        raw_response = [r async for r in async_gen_wrapper(route_result_stream, DEFAULT_MODEL)]
        route_result = "".join(raw_response)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Router result:\n{route_result}\n")
        try:
            json_response = json.loads(route_result)
            agent = json_response.get("agent", None)
        except:
            logger.info("Unable to parse response as json")
            agent = None
        if agent:
            return {"agent": agent}
        else:
            if set("RAG").issubset(set(route_result)):
                return {"agent": "RAG"}
            elif set("SIMPLE").issubset(set(route_result)):
                return {"agent": "SIMPLE"}
            elif set("FOLLOW_UP").issubset(set(route_result)):
                return {"agent": "FOLLOW_UP"}
            else:
                return {"agent": "FALLBACK"}  # create fallback here

    async def simple_response(self, messages):
        result = await self.stream_response(messages, self.simple_template)
        return {"output": result}

    async def rag_response(self, messages):
        messages["reformulated_queries"] = await self.reformulate_response(messages)
        result = await self.stream_response(messages, self.rag_template)
        return {"output": result}

    async def fallback_response(self, messages):
        result = await self.stream_response(messages, self.fallback_template)
        return {"output": result}

    async def reformulate_response(self, messages):
        reformulated_response = await self.stream_response(messages, self.reformulate_template)
        raw_response = [r async for r in async_gen_wrapper(reformulated_response, DEFAULT_MODEL)]
        reformulate_output = "".join(raw_response).replace("json", "").replace("```", "")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Reformulated response:{reformulate_output}")
        try:
            json_response = self._extract_json(reformulate_output)
            reformulated_queries = json_response.get("søgninger", [])
        except:
            logger.info("Unable to parse as json.")
            reformulated_queries = []
        return reformulated_queries

    async def follow_up_response(self, messages):
        messages["reformulated_queries"] = await self.reformulate_response(messages)
        result = await self.stream_response(messages, self.follow_up_template)
        return {"output": result}

    async def stream_response(self, messages, template):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.__generate(messages, self.model, template))

    def __generate(self, messages, model, template):
        response_stream = model.stream_response(messages, template)
        return response_stream

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extracts the first valid JSON array from a given string and returns it as a Python dictionary."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except json.JSONDecodeError:
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f"Malformed json in model_output:\n{text}")
            else:
                logger.info("Malformed json in model_output.")
        return {}
