#!/usr/bin/env python
"""
:mod:`science_rag.preprocessing.link_map` -- sources and metadata for science rag documents

========
Link map
========

The link map is a csv with one row per document, keyed on ``filename``. ``url`` and ``title``
become ``URL`` and ``Title`` in the chunk metadata, the remaining columns are added under
capitalised Danish keys, but only when the link map has a value for that document.

Astra pages (aktiviteter/forløb) are not files, so their urls come from a separate json map from
page title to url, loaded with ``load_astra_links``.
"""

import json
import unicodedata
from urllib.parse import urlparse

import pandas as pd

__all__ = ["load_link_map", "load_astra_links", "extra_metadata", "chunk_url", "nfc"]

# link map column -> metadata key
EXTRA_METADATA_COLS = {
    "afsender": "Afsender",
    "fag": "Fag",
    "klassetrin": "Klassetrin",
    "forloeb": "Forløb",
    "indskoling": "Indskoling",
    "mellemtrin": "Mellemtrin",
    "udskoling": "Udskoling",
    "laerervejledning": "Lærervejledning",
}
BOOL_COLS = {"indskoling", "mellemtrin", "udskoling", "laerervejledning"}


def nfc(s: str) -> str:
    """Normalize to NFC, so filenames from the filesystem and the csv compare equal."""
    return unicodedata.normalize("NFC", s)


def load_link_map(path) -> dict[str, dict[str, str]]:
    """Load the link map csv as {filename: row}, with empty strings for missing values."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {nfc(row["filename"]): row for row in df.to_dict("records")}


def load_astra_links(path) -> dict[str, str]:
    """Load the Astra links json as {title: url}."""
    with open(path, encoding="utf-8") as f:
        return {nfc(title): url for title, url in json.load(f).items()}


def extra_metadata(row: dict[str, str]) -> dict:
    """Metadata from the non-empty extra columns of a link map row."""
    metadata = {}
    for col, key in EXTRA_METADATA_COLS.items():
        value = row.get(col, "")
        if value == "":
            continue
        metadata[key] = value == "True" if col in BOOL_COLS else value
    return metadata


def chunk_url(source: str, page_no: int) -> str:
    """Link to a chunk; only pdf links get a ``#page=N`` anchor."""
    if urlparse(source).path.lower().endswith(".pdf"):
        return f"{source}#page={page_no}"
    return source
