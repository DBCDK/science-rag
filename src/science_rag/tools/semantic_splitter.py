#!/usr/bin/env python
"""
:mod:`science_rag.tools.semantic_splitter -- splits text based on semantic boundaries

=================
Semantic Splitter
=================

Splits text based on semantic bounderies. This is inspired by the
fourth method in this video https://www.youtube.com/watch?v=8OJC21T2SL4

For each sentence in the text, an embedding is created using the
previous, current, and next sentence. Once all embeddings are
generated, chunks are formed by splitting the text wherever the
semantic difference between embeddings exceeds a specified threshold.

The splits can be based on three different measures:
* percentile
  The default way to split is based on percentile.  In this method,
  all differences between sentences are calculated, and then any
  difference greater than the X percentile is split.
* standard deviation (default value is 70)
  In this method, any difference greater than X standard deviations is split.
* cosine-distance (default value is 3).
  The raw cosine distance between blocks. The threshold will be the
  same across all texts (default value is 0.12).

usage:
   In [2]: sem_split = SemanticSplitter(breakpoint_threshold=1, breakpoint_threshold_type="standard-deviation")
   In [3]: sem_split.split_text("hesten gik på marken. Den var sort. Kokken fik fri kl. 5")
   Out[3]: ['hesten gik på marken. Den var sort.', 'Kokken fik fri kl. 5']
"""

from typing import Literal, get_args

import numpy as np
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from science_rag.tools.embedder import Embedder, HuggingfaceEmbedder

__all__ = ["SemanticSplitter"]

BreakpointThresholdType = Literal["cosine-distance", "percentile", "standard-deviation"]


class SemanticSplitter:
    """Splits text based on semantic bounderies."""

    def __init__(
        self,
        breakpoint_threshold: float | None = None,
        breakpoint_threshold_type: BreakpointThresholdType | None = "percentile",
        embedder: Embedder | None = None,
    ) -> list[str]:
        """
        Initializes semantic textsplitter

        :param breakpoint_threshold:
            Threshold for splitting text.
            Default values varies across threshold types
        :param breakpoint_threshold_type:
            Threshold where splitting occur
        :param embedder:
            Embedder instance to use on chunks
        """
        (self.breakpoint_threshold, self.breakpoint_threshold_type, self.embedder) = self.__configure(
            breakpoint_threshold, breakpoint_threshold_type, embedder
        )

    def split_text(self, text: str) -> list[str]:
        """
        Splits text

        :param text:
            text to split
        """
        sentences = [s.strip() for s in sent_tokenize(text)]
        neighborhoods = []
        for i in range(len(sentences)):
            neighborhood = sentences[i]
            if i > 0:
                neighborhood = sentences[i - 1] + " " + neighborhood
            if i < len(sentences) - 1:
                neighborhood += " " + sentences[i + 1]
            neighborhoods.append(neighborhood)

        neighborhood_encodings = self.embedder.encode(neighborhoods)
        neighborhood_distances = []
        for i in range(len(sentences) - 1):
            neighborhood_distances.append(
                1
                - cosine_similarity(
                    neighborhood_encodings[i].reshape(1, -1),
                    neighborhood_encodings[i + 1].reshape(1, -1),
                ).squeeze()
            )

        threshold = self.__get_threshold(
            neighborhood_distances,
            self.breakpoint_threshold,
            self.breakpoint_threshold_type,
        )
        neighborhood_distances.append(0)
        chunks = [""]
        for sentence, distance in zip(sentences, neighborhood_distances):
            if chunks[-1]:
                chunks[-1] += " "
            chunks[-1] += sentence
            if distance > threshold:
                chunks.append("")

        return chunks

    def __get_threshold(self, distances, breakpoint_threshold, breakpoint_threshold_type) -> float:
        match breakpoint_threshold_type:
            case "cosine-distance":
                return breakpoint_threshold
            case "percentile":
                return np.percentile(distances, breakpoint_threshold)
            case "standard-deviation":
                return np.mean(distances) + breakpoint_threshold * np.std(distances)

    def __configure(self, breakpoint_threshold, breakpoint_threshold_type, embedder):
        default_thresholds = {
            "cosine-distance": 0.12,
            "percentile": 70,
            "standard-deviation": 3,
        }
        if not embedder:
            embedder = HuggingfaceEmbedder()
        if not breakpoint_threshold_type:
            breakpoint_threshold_type = "cosine-distance"
        if breakpoint_threshold_type not in get_args(BreakpointThresholdType):
            raise KeyError(f"Unknown breakpoint_threshold_type: {breakpoint_threshold_type}")
        if not breakpoint_threshold:
            breakpoint_threshold = default_thresholds[breakpoint_threshold_type]

        return breakpoint_threshold, breakpoint_threshold_type, embedder
