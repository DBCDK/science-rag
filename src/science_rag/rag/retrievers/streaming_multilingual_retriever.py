#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`mitcfu_rag.retrievers.embedding_retriever - embedding_retriever

============
EmbeddingRetriever
============

EmbeddingRetriever retrieves relevant references based on the messages from the chat sent.
There is no underlying database and EmbeddingRetriever returns an dummy document.

example of usage:
    from mitcfu_rag.embedding_retriever import EmbeddingRetriever
    e_retriever = EmbeddingRetriever()
    messages = messages = ["Hej", "Er der noget om biblioteker?"]
    refs = e_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
"""

import logging
from science_rag.rag.rag import Retriever, Reference
from science_rag.tools import KNNSearch
from science_rag.tools.llm_formatting import clean_sources_from_messages

# from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import aiohttp
import numpy as np
import torch
import json

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index"
MODEL_PATH = "/data/mitCFU-models/multilingual-e5-large"
CROSS_MODEL_PATH = "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"
EMBEDDING_SERVER_URL = "http://ai-p301:5009/v1/embeddings"


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        model_path=MODEL_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        cross_model_path=CROSS_MODEL_PATH,
        embedding_server_url=EMBEDDING_SERVER_URL,
        jed_document_path=None,
    ):
        # We should not use GPU for such small models, since they will take up the whole k8s GPU regardless of their size
        self.device = "cpu"
        # embedding model for faiss index
        self.embedding_server_url = embedding_server_url
        self.session = aiohttp.ClientSession()
        # cross-model for rerank
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(cross_model_path, device_map=self.device)
        self.cross_tokenizer = AutoTokenizer.from_pretrained(cross_model_path, device_map=self.device)
        self.cross_model.to(self.device)
        self.cross_model.eval()
        self.searcher = KNNSearch.load(embeddings_path + "/embeddings", embeddings_path + "/labels.npy")
        if jed_document_path:
            self.jed_document_path = jed_document_path
            self.all_articles = self.initiate_articles()
            self.all_materialtypes = self.identify_all_materialtypes()
        else:
            self.all_articles = {}
            self.all_materialtypes = set()
        self.validator = None
        self.task = "Given a web search query, retrieve relevant passages that answer the query"


    def initiate_articles(self):
        with open(self.jed_document_path) as f:
            all_documents = json.load(f)
        all_articles = {}

        for document in all_documents:
            for doc_id, doc in document.items():
                text = doc.get("abstract")
                if text:
                    all_articles[str(doc_id)] = self.format_doc(doc_id, doc)

        return all_articles

    def identify_all_materialtypes(self):
        # initiate materialtype filters
        all_materialtypes = set()
        for articleid, article in self.all_articles.items():
            all_materialtypes.update(set(article.materialtypes))
        return all_materialtypes

    def format_doc(self, doc_id, doc):
        # work info. Currently empty since we are using only abstracts from LangChain document page_content field.
        text = doc.get("abstract")
        subjects = []
        creators_person = []
        creators_publisher = []
        audience_subject = []
        material_types_empty = []
        languages = []
        series_titles = []
        # the doc_id is composed of a string.pdf + page number + chunk number. Here we split them up.
        # an example could be Fight the Bite.pdf_side58_chunk0 --> Fight the Bite.pdf, side 58, chunk0
        # currently, the chunk number is not used.
        pdf_title = doc_id.rsplit("_side", maxsplit=1)[0]
        page_number = doc_id.rsplit("_side", maxsplit=1)[1].rsplit("_chunk", maxsplit=1)[0]
        return Reference(
            id=str(doc_id),
            # article_headline=doc.get("titles").get("full")[0],
            article_headline="-",
            article_link=pdf_title + "#page=" + page_number,
            score=0.0,
            text=text,
            chunk="Not chunked",
            keywords=subjects,
            creators=creators_person + creators_publisher,
            audience=audience_subject,
            # materialtypes=material_types_general + material_types_specific,
            materialtypes=material_types_empty,
            publicationdate=doc.get("firstPublicationDate", None),
            languages=languages,
            series=series_titles,
        )

    async def async_retrieve(self, messages: list[str], n: int = 5, follow_up=True):
        return await self.retrieve(messages, n, follow_up=follow_up)

    async def async_rerank_retrieve(self, messages: list[str], n: int = 5, follow_up=True):
        return await self.rerank_retrieve(messages, n, follow_up=follow_up)

    async def retrieve(self, input: list[str], n: int = 3, follow_up: bool = False):
        messages = clean_sources_from_messages(input["input"])
        if follow_up:
            query = f"Instruct: {self.task}\nQuery: {', '.join([message['content'] for message in messages])}"
        else:
            query = f"Instruct: {self.task}\nQuery: {messages[-1]['content']}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Retrieving resources for {query}")
        return await self.get_docs(query, n)


    async def get_docs(self, query: str, limit: int = 3):
        if not self.session:
            raise RuntimeError("EmbeddingRetriever must be used as an async context manager")
        payload = {
            "model": "intfloat/multilingual-e5-large-instruct",
            "input": query
        }
        print(payload)
        async with self.session.post(self.embedding_server_url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()

        embedding = result["data"][0]["embedding"]
        embedded_query = np.array(embedding, dtype=np.float32).reshape(1, -1)
        return await self.search(embedded_query, limit)


    async def search(self, embedded_query, limit, filters=None):
        hits = await self.searcher.search(embedded_query, limit * 100)
        ids, scores = zip(*hits)
        # retrieved_articles = [self.all_articles[id.rsplit("_chunk", maxsplit=1)[0]] for id in ids]
        retrieved_articles = [self.all_articles[id] for id in ids if id in self.all_articles]
        if filters:
            filtered_articles = [article for article in retrieved_articles if filtered(article, filters)]
            return scores, filtered_articles[:limit]
        else:
            return scores, retrieved_articles[:limit]

    async def rerank_retrieve(self, input: dict, limit: int = 50, follow_up=True):
        queries = input["reformulated_queries"]
        all_results = {}
        articleid2article = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Retrieving resources for {queries}")
        for query in queries:
            query = f"Instruct: {self.task}\nQuery: {query}"
            _, articles = await self.get_docs(query, limit=limit)
            for art in articles:
                articleid2article[art.id] = art
            search_results = await self.cross_select_top_sentences(articles, query, limit=int(40 / len(queries)))
            all_results[query] = search_results

        if len(queries) > 1:
            reranked_result = await reciprocal_rank_fusion(all_results)
            return [], [articleid2article[_id] for _id in reranked_result.keys()]
        else:
            return [], list(all_results[queries[0]].keys())

    async def cross_select_top_sentences(self, articles, query, limit=50):
        sentences = [art.text for art in articles]
        features = self.cross_tokenizer(
            [query] * len(sentences),
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            scores = self.cross_model(**features).logits.flatten().cpu()

        # Get indices of the top sentences sorted by cosine similarity
        top_indices = np.argsort(-scores)[:limit]
        # print("Num top indices: ", len(top_indices))

        # Collect the top sentences and their respective cosine scores
        top_article_ids_with_scores = {articles[i].id: scores[i] for i in top_indices}

        return top_article_ids_with_scores


async def reciprocal_rank_fusion(search_results_dict, k=100):
    fused_scores = {}

    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + k)

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results


def filtered(article, filters):
    passed_filters = 0
    for filter_key, filter_values in filters:
        if set([filter_values]).intersection(set([article.filter_key])):
            passed_filters += 1
    return passed_filters == len(filters.keys())


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
