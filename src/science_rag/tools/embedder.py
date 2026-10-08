#!/usr/bin/env python
"""
:mod:`science_rag.tools.embedder -- Embeds texts

========
Embedder
========

Abtract base class for embedders, and huggingface and OpenAI-compatible implementations
"""

from abc import ABC, abstractmethod
import numpy as np
from openai import OpenAI
from sentence_transformers.SentenceTransformer import SentenceTransformer


__all__ = ["Embedder", "HuggingfaceEmbedder", "OpenAIEmbedder"]


class Embedder(ABC):
    """
    Abstract base class embedder
    """

    name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.array:
        """encode method"""
        pass


class HuggingfaceEmbedder(Embedder):
    """Huggingface embedder. Using sentencetransformer"""

    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        :param model_name:
            Name of sentence transformer model to use
        """
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.array:
        """Encodes strings"""
        return self.model.encode(texts)


class OpenAIEmbedder(Embedder):
    """Embedder backed by an OpenAI-compatible /v1/embeddings endpoint"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        """
        :param base_url:
            The `.../v1` base of the endpoint. A full `.../v1/embeddings` url is accepted too
        :param model:
            Model name sent to the endpoint
        :param api_key:
            Bearer token, if the endpoint requires one
        """
        self.name = model
        # openai.OpenAI wants the `.../v1` base and appends `embeddings` itself, and refuses to start without a key
        self.client = OpenAI(base_url=base_url.rstrip("/").removesuffix("/embeddings"), api_key=api_key or "unused")

    def encode(self, texts: list[str]) -> np.array:
        """Encodes strings"""
        # float, since not every OpenAI-compatible server supports the sdk's default base64 encoding
        response = self.client.embeddings.create(model=self.name, input=texts, encoding_format="float")
        return np.array([d.embedding for d in sorted(response.data, key=lambda d: d.index)], dtype=np.float32)
