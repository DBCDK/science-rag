#!/usr/bin/env python
"""
:mod:`science_rag.rag -- interface for rag models

All rag models must inherit from this class and implement the abstractmethods
"""

from typing import Generator, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Reference:
    id: str
    chunk: str
    article_headline: str
    article_link: str
    score: float
    text: str
    # values that can be filtered by
    keywords: list[str]
    creators: list[str]
    audience: list[str]
    materialtypes: list[str]
    languages: list[str]
    series: list[str]
    publicationdate: str

    def __str__(self):
        return f"""
id: {self.id} \n
{self.article_headline} \n
{self.article_link} \n
""{self.text}""
                """

    def __hash__(self):
        return hash(self.article_link)

    def __eq__(self, another_reference):
        if not isinstance(another_reference, Reference):
            raise TypeError("Can only compare two References")
        if self.article_link == another_reference.article_link:
            return True
        else:
            return False

    def __ne__(self, another_reference):
        if not isinstance(another_reference, Reference):
            raise TypeError("Can only compare two References")
        if self.article_link != another_reference.article_link:
            return True
        else:
            return False


class RAG(ABC):
    def __call__(self, messages: list[str], *args, **kwargs) -> str:
        return self.get_response(messages)

    @abstractmethod
    def get_response(self, messages: list[dict[str, Any]], *args, **kwargs) -> str:
        """
        Revieves a list of chat messages and returns the next response given by the chatbot.
        """
        pass

    @abstractmethod
    def stream_response(
        self, messages: list[dict[str, Any]], *args, **kwargs
    ) -> Generator[str, None, None]:
        """
        yields response tokens from rag request.
        """
        pass

    @abstractmethod
    def evaluate(self, messages: list[str]) -> tuple[list[Reference], str]:
        """
        Takes a list of chat messages as input and returns retrieved references given to the generator
        and the generated response for evaluation.
        """
        pass


class Retriever(ABC):

    def __call__(self, messages: list[str], *args, **kwargs):
        return self.retrieve(messages)

    @abstractmethod
    def retrieve(
        self, messages: list[str], *args, **kwargs
    ) -> tuple[list[float], list[Reference]]:
        """ "
        returns similarity scores and a list of references.
        """
        pass


class Generator(ABC):

    def __call__(self, references: list[Reference], query: str, *args, **kwargs) -> str:
        return self.generate(references, query)

    @abstractmethod
    def generate(self, references: list[Reference], query: str, *args, **kwargs) -> str:
        pass
