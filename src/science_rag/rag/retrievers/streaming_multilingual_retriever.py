#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`science_rag.rag.retrievers.streaming_multilingual_retriever` - embedding_retriever

============
EmbeddingRetriever
============

EmbeddingRetriever retrieves relevant references based on the messages from the chat sent.
There is no underlying database and EmbeddingRetriever returns an dummy document.

example of usage:
    from science_rag.rag.retrievers.streaming_multilingual_retriever import EmbeddingRetriever
    e_retriever = EmbeddingRetriever()
    messages = messages = ["Hej", "Er der noget om biblioteker?"]
    refs = e_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
"""

import logging
from science_rag.rag.rag import Retriever, Reference
from science_rag.tools import KNNSearch
from science_rag.tools.message_history import clean_sources_from_messages

# from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn.functional as F
import json

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index"
MODEL_PATH = "/data/mitCFU-models/multilingual-e5-large"
CROSS_MODEL_PATH = "/data/mitCFU-models/ms-marco-MiniLM-L-6-v2"


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        model_path=MODEL_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        cross_model_path=CROSS_MODEL_PATH,
        jed_document_path=None,
    ):
        # We should not use GPU for such small models, since they will take up the whole k8s GPU regardless of their size
        self.device = "cpu"
        # embedding model for faiss index
        self.model = AutoModel.from_pretrained(model_path, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=self.device)
        self.model.to(self.device)
        self.model.eval()
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
        metadata = doc.get("metadata", None)
        subjects = []
        creators_person = []
        creators_publisher = []
        audience_subject = []
        material_types_empty = []
        languages = []
        series_titles = []

        article_link_fallback = f"{metadata.get('Title', 'No_Link')}.pdf"
        pdf_title_fallback = metadata.get("Title", "No_Title").replace(".pdf", "")

        return Reference(
            id=str(doc_id),
            article_headline=metadata.get("Title", pdf_title_fallback),
            article_link=metadata.get("URL", article_link_fallback),
            score=0.0,
            text=text,
            chunk="Not chunked",
            keywords=subjects,
            creators=creators_person + creators_publisher,
            audience=audience_subject,
            # materialtypes=material_types_general + material_types_specific,
            materialtypes=material_types_empty,
            publicationdate=doc.get("firstPublicationDate") or "",
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

    # https://huggingface.co/intfloat/multilingual-e5-large
    async def get_docs(self, query: str, limit: int = 3):
        model_device = next(self.model.parameters()).device
        batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(model_device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embedded_query = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy().astype(np.float32)
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
