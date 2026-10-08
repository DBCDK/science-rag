import argparse
import json
import logging
import os

import pandas as pd
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from science_rag.preprocessing.astra_df_to_chunked_docs import astra_df_to_docling_chunks
from science_rag.preprocessing.astra_preprocessor import AstraPreprocessor
from science_rag.preprocessing.link_map import chunk_url, extra_metadata, load_link_map, nfc
from science_rag.rag.retrievers.indexes.multilinguale5 import index_paragraph_docs_GPU_batches
from science_rag.tools.embedder import OpenAIEmbedder

# Importing standard config for Astra csv's ---> which columns use for abstract and metadata
# all columns not listed: used in abstact.
# aktiviteter_metadata_cols: use for metadata (but exclude from abstract).
# exclude_cols or exclude_col_if_contains: exclude altogether, based on exact column name or if the column contains a certain string.
from science_rag.preprocessing.astra_csv_cols_config import (
    AKTIVITETER_METADATA_COLS,
    AKTIVITETER_EXCLUDE_COLS,
    AKTIVITETER_EXCLUDE_COL_IF_CONTAINS,
    FORLOB_METADATA_COLS,
    FORLOB_EXCLUDE_COLS,
    FORLOB_EXCLUDE_COL_IF_CONTAINS,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
GLYPHGATE_API_KEY = os.environ.get("GLYPHGATE_API_KEY")


def get_science_rag_document_paths(path_to_folder):
    science_rag_doc_paths = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            if file.endswith(".pdf"):
                science_rag_doc_paths.append(os.path.join(root, file))
    return science_rag_doc_paths


def get_docling_chunks(input_file, link_map, converter, chunker):
    doc = converter.convert(input_file).document
    chunks = [chunk for chunk in chunker.chunk(dl_doc=doc)]

    # convert to expected json format
    science_rag_docs = []
    for i, chunk in enumerate(chunks):
        filename = nfc(chunk.meta.origin.filename)
        row = link_map.get(filename, {})
        source = row.get("url") or filename
        page_no = chunk.meta.doc_items[0].prov[0].page_no
        # the id is built from the filename, since several documents can share the same url
        document_chunk = {
            f"{filename}_side{page_no}_chunk{i}": {
                "abstract": chunker.contextualize(chunk=chunk),
                "metadata": {
                    "URL": chunk_url(source, page_no),
                    "Title": row.get("title") or filename.replace(".pdf", ""),
                }
                | extra_metadata(row),
            }
        }
        science_rag_docs.append(document_chunk)

    return science_rag_docs


def get_remote_embedder(endpoint_url, model):
    if not GLYPHGATE_API_KEY:
        logger.warning(f"The env-var GLYPHGATE_API_KEY is not set, calling {endpoint_url} without an api key")
    return OpenAIEmbedder(base_url=endpoint_url, model=model, api_key=GLYPHGATE_API_KEY)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_science_rag_folder",
        metavar="path-to-science-rag-folder",
        help="path to folder containing science rag documents",
        type=str,
    )
    parser.add_argument(
        "document_index_file_path", metavar="document-index-file-path", help="path to save the indexed chunks", type=str
    )
    parser.add_argument(
        "faiss_db_directory",
        metavar="faiss-db-directory",
        help="path to directory to save faiss index and embeddings",
        type=str,
    )
    parser.add_argument(
        "--link-map",
        type=str,
        required=True,
        help="path to csv with sources and metadata for the documents, keyed on filename",
    )
    parser.add_argument(
        "--batch-size",
        metavar="batch-size",
        help="batch size for embedding chunks",
        default=50,
    )
    parser.add_argument(
        "--aktiviteter-csv", type=str, required=False, help="(Optional) Path to the Aktiviteter CSV file."
    )
    parser.add_argument("--forlob-csv", type=str, required=False, help="(Optional) Path to the Forløb CSV file.")
    parser.add_argument(
        "--embedding-endpoint",
        type=str,
        required=False,
        default=None,
        help="(Optional) base url (`.../v1`) of an OpenAI-compatible embeddings endpoint. If not set, embeds locally on GPU/CPU.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="model name sent to --embedding-endpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Getting paths to science rag documents from {args.path_to_science_rag_folder}")
    science_rag_doc_paths = get_science_rag_document_paths(args.path_to_science_rag_folder)

    # a science_rag_chunk must - in order (to work with index_paragraph_docs_GPU_batches - contain:
    # ID as the key to a dictionary with at least "abstract" as a key, and the value of "abstract"
    # should be the text to embed
    science_rag_chunks = []

    link_map = load_link_map(args.link_map)
    missing = [p for p in science_rag_doc_paths if nfc(os.path.basename(p)) not in link_map]
    for file_path in missing:
        logger.warning(f"{file_path} is not in the link map, using filename as source")

    logger.info(f"Reading and chunking {len(science_rag_doc_paths)} science rag documents")
    converter = DocumentConverter()
    chunker = HybridChunker()
    for file_path in science_rag_doc_paths:
        science_rag_chunks.extend(get_docling_chunks(file_path, link_map, converter, chunker))

    # Initializing preprocessor only if we have to preprocess Astra CSV files
    if args.aktiviteter_csv or args.forlob_csv:
        preprocessor = AstraPreprocessor()

    # Reading and preprocessing activities (aktiviteter)
    if args.aktiviteter_csv:
        aktiviteter_df = pd.read_csv(args.aktiviteter_csv, sep=",", encoding="utf-8")
        aktiviteter_list_of_jedish_docs = astra_df_to_docling_chunks(
            aktiviteter_df,
            preprocessor=preprocessor,
            metadata_cols=AKTIVITETER_METADATA_COLS,
            exclude_cols=AKTIVITETER_EXCLUDE_COLS,
            exclude_col_if_contains=AKTIVITETER_EXCLUDE_COL_IF_CONTAINS,
        )

        science_rag_chunks.extend(aktiviteter_list_of_jedish_docs)

    # Reading and preprocessing courses (forløb)
    if args.forlob_csv:
        forlob_df = pd.read_csv(args.forlob_csv, sep=",", encoding="utf-8")
        forlob_list_of_jedish_docs = astra_df_to_docling_chunks(
            forlob_df,
            preprocessor=preprocessor,
            metadata_cols=FORLOB_METADATA_COLS,
            exclude_cols=FORLOB_EXCLUDE_COLS,
            exclude_col_if_contains=FORLOB_EXCLUDE_COL_IF_CONTAINS,
        )

        science_rag_chunks.extend(forlob_list_of_jedish_docs)

    logger.info(f"Saving {len(science_rag_chunks)} science rag chunks")
    with open(args.document_index_file_path, "w") as f:
        json.dump(science_rag_chunks, f)

    logger.info(
        f"Embedding {len(science_rag_chunks)} science rag chunks and saving FAISS db to {args.faiss_db_directory}"
    )
    embedder = None
    if args.embedding_endpoint:
        embedder = get_remote_embedder(args.embedding_endpoint, args.embedding_model)
    index_paragraph_docs_GPU_batches(
        path_to_index_file=args.document_index_file_path,
        path=args.faiss_db_directory,
        batch_size=args.batch_size,
        embedder=embedder,
    )


if __name__ == "__main__":
    main()
