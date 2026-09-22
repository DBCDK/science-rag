#!/usr/bin/env python
"""
:mod:`science_rag.tools.embedder -- Embeds texts

========
Embedder
========

Abtract base class for embedders, and huggingface implementation
"""

from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers.SentenceTransformer import SentenceTransformer


__all__ = ["Embedder", "HuggingfaceEmbedder"]


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
