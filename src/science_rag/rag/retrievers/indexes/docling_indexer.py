from science_rag.tools.generic_parser import GenericParser
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from science_rag.tools.knn_searcher import KNNSearch
from science_rag.tools.embedder import Embedder
from science_rag.rag.retrievers.indexes.multilinguale5 import index_paragraph_docs_GPU_batches
from transformers import AutoTokenizer, AutoModel
import random
import logging
import os
import json
import requests

logger = logging.getLogger(__name__)

EMBEDDING_SERVER_URL = "http://ai-p301:5009/v1/embeddings"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

# Temporary map to show some examples of why links work/don't work
WEBPDF_MAP = {
    "zoo-aarsberetning-2024.pdf": "https://content.zoo.dk/media/fk1nlps0/zoo-aarsberetning-2024.pdf",
    "aarsberetning-2023.pdf": "https://content.zoo.dk/media/whqkk2vs/aarsberetning-2023.pdf",
    "FORVALTNING AF DYREBESTAND.pdf": "https://www.zoo.dk/om-zoo/dyrene-i-zoo/forvaltning-af-dyrebestanden",
    "Computational Thinking integreret i matematikundervisningen.pdf": "https://doi.org/10.5281/zenodo.19255073",
    "Fight the Bite.pdf": "https://undervisning.life.dk/fb",
}

def get_science_rag_document_paths(path_to_folder):
    # temporary list of documents to ignore
    documents_to_ignore = {"Samling af datakilder til RAG.docx",
                           "links til kilder.docx",
                           "Webhenvisning fra zoo.docx",
                           "speciale-henvisninger.docx"}

    science_rag_doc_paths = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            if file in documents_to_ignore:
                continue
            if file.endswith('.pdf'):
                science_rag_doc_paths.append(os.path.join(root, file))
    return science_rag_doc_paths

def get_docling_chunks(input_file):
    # parse document
    converter = DocumentConverter()
    doc = converter.convert(input_file).document

    # chunk document
    chunker = HybridChunker()
    chunks = [chunk for chunk in chunker.chunk(dl_doc=doc)]

    # convert to expected json format
    jedish_docs = []
    for i, chunk in enumerate(chunks):
        source = WEBPDF_MAP.get(chunk.meta.origin.filename, chunk.meta.origin.filename)
        jedish_json = {
            f"{source}_side{chunk.meta.doc_items[0].prov[0].page_no}_chunk{i}": {
                "abstract": chunker.contextualize(chunk=chunk)
            }
        }
        jedish_docs.append(jedish_json)

    return jedish_docs

def validate_abstract(document):
    id, doc = next(iter(document.items()))
    abstract = doc.get("abstract")
    # before appending, check if the text is non-string type
    if not isinstance(abstract, str):
        logger.debug(
            f"Abstract is not a string: {abstract} in doc {doc} with id {doc['id']}"
        )
        return False

    if len(abstract) < 150:
        return False

    return True

def embed_chunks(chunks: list[dict], batch_size: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Embed all chunks via the embedding server, returning (embeddings, labels)."""
    all_embeddings = []
    labels = []
    batch_texts = []
    batch_labels = []

    def flush_batch():
        response = requests.post(
            EMBEDDING_SERVER_URL,
            json={"input": batch_texts, "model": EMBEDDING_MODEL},
        )
        response.raise_for_status()
        data = response.json()["data"]
        batch_embeddings = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        all_embeddings.extend(batch_embeddings)
        labels.extend(batch_labels)
        batch_texts.clear()
        batch_labels.clear()

    for doc in tqdm(chunks, desc="Embedding chunks"):
        for _id in doc:
            if not validate_abstract(doc):
                continue
            abstract = doc[str(_id)].get("abstract")
            batch_texts.append(abstract)
            batch_labels.append(_id)

            if len(batch_texts) >= batch_size:
                flush_batch()

    if batch_texts:
        flush_batch()

    return np.array(all_embeddings, dtype=np.float32), np.array(labels)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_science_rag_folder",
        metavar="path-to-science-rag-folder",
        help="path to folder containing science rag documents",
        type=str,
    )
    parser.add_argument(
        "document_index_file_path",
        metavar="document-index-file-path",
        help="path to save the indexed chunks",
        type=str)
    parser.add_argument(
        "faiss_db_directory",
        metavar="faiss-db-directory",
        help="path to directory to save faiss index and embeddings",
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        metavar="batch-size",
        help="batch size for embedding chunks",
        default=50,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Getting paths to science rag documents from {args.path_to_science_rag_folder}")
    science_rag_doc_paths = get_science_rag_document_paths(args.path_to_science_rag_folder)
    science_rag_chunks = []

    logger.info(f"Reading and chunking {len(science_rag_doc_paths)} science rag documents")
    for file_path in science_rag_doc_paths:
        science_rag_chunks.extend(get_docling_chunks(file_path))

    logger.info(f"Saving {len(science_rag_chunks)} science rag chunks to {args.document_index_file_path}")
    with open(args.document_index_file_path, "w") as f:
        json.dump(science_rag_chunks, f)

    logger.info(f"Embedding {len(science_rag_chunks)} chunks via {EMBEDDING_SERVER_URL}")
    embeddings, labels = embed_chunks(science_rag_chunks, batch_size=int(args.batch_size))

    logger.info(f"Saving FAISS index to {args.faiss_db_directory}")
    os.makedirs(args.faiss_db_directory, exist_ok=True)
    db = KNNSearch.build(embeddings, labels)
    db.save(
        index_path=args.faiss_db_directory + "/embeddings",
        labels_path=args.faiss_db_directory + "/labels",
    )
    logger.info(f"FAISS db saved to {args.faiss_db_directory}")


if __name__ == "__main__":
    main()
