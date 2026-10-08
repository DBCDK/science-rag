import pandas as pd
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from science_rag.preprocessing.astra_preprocessor import AstraPreprocessor
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def _df_to_json_safe_dict(value):
    """
    Small helper function to make sure metadata is JSON serializable
    and does not contain e.g. NaNs or Pandas-specific types.
    """
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


# Astra school levels -> grade ranges, in the "4.-6." format used by the link map
SCHOOL_LEVEL_GRADES = {"Indskoling": (0, 3), "Mellemtrin": (4, 6), "Udskoling": (7, 9)}

# The Astra exports have no URL column, but astra.dk is WordPress and redirects ?p=<post ID> to the page
ASTRA_URL_TEMPLATE = "https://astra.dk/?p={id}"


def _astra_url(post_id) -> str | None:
    """2623 / 2623.0 / '2623' -> 'https://astra.dk/?p=2623'; missing or non-numeric IDs give None."""
    try:
        return ASTRA_URL_TEMPLATE.format(id=int(float(post_id)))
    except (TypeError, ValueError):
        return None


def _astra_fag(fag: str) -> str | None:
    """'Grundskole>Biologi|Grundskole>Fysik/Kemi' -> 'Biologi, Fysik/Kemi'; level-only entries are dropped."""
    subjects = [entry.split(">", 1)[1] for entry in fag.split("|") if ">" in entry]
    return ", ".join(dict.fromkeys(subjects)) or None


def _astra_klassetrin(levels: list[str]) -> str | None:
    """['Mellemtrin', 'Udskoling', 'EUD'] -> '4.-9., EUD'; adjacent school levels are merged into one range."""
    ranges = sorted(SCHOOL_LEVEL_GRADES[level] for level in levels if level in SCHOOL_LEVEL_GRADES)
    merged = []
    for first, last in ranges:
        if merged and first == merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    parts = [f"{first}.-{last}." for first, last in merged]
    parts += [level for level in levels if level not in SCHOOL_LEVEL_GRADES]
    return ", ".join(parts) or None


def harmonize_astra_metadata(metadata: dict, afsender: str = "Astra") -> dict:
    """
    Align Astra metadata with the metadata the link map gives the pdf documents (see link_map.py):
    adds Afsender, adds URL from the WordPress post ID if missing, rewrites Fag and Klassetrin to the
    link map format, and adds the Indskoling/Mellemtrin/Udskoling booleans from Klassetrin.
    Empty values are left out.
    """
    metadata = {**metadata, "Afsender": afsender}
    if not metadata.get("URL"):
        metadata["URL"] = _astra_url(metadata.get("ID"))
    if isinstance(metadata.get("Fag"), str):
        metadata["Fag"] = _astra_fag(metadata["Fag"])
    if isinstance(metadata.get("Klassetrin"), str):
        levels = [level.strip() for level in metadata["Klassetrin"].split("|") if level.strip()]
        metadata["Klassetrin"] = _astra_klassetrin(levels)
        for level in SCHOOL_LEVEL_GRADES:
            metadata[level] = level in levels
    return {k: v for k, v in metadata.items() if v is not None}


def astra_df_to_docling_chunks(
    df: pd.DataFrame,
    preprocessor: AstraPreprocessor,
    metadata_cols: list[str],
    exclude_cols: list[str],
    exclude_col_if_contains: list[str],
):
    """
    Convert a DataFrame to Docling chunks in a jedish-compatible format. This format is a list of
    dictionaries, where each dictionary has a chunk ID_chunk_idx as keys and values are dictionaries
    containing 'abstract' (text to be embedded) and 'metadata' - including at least the Title and URL
    in this metadata is important for downstream tasks.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be processed (should contain Title and URL column).
        preprocessor (AstraPreprocessor): An instance of AstraPreprocessor or similar to preprocess the text.
        metadata_cols (list[str]): List of column names to keep as metadata and not use in abstract.
        exclude_cols (list[str]): List of column names to exclude from the abstract.
        exclude_col_if_contains (list[str]): List of substrings; any column containing these are excluded from abstract.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary has a chunk ID as key and a
        value that is another dictionary with 'abstract' (text for embedding) and 'metadata'.
    """
    df_raw = preprocessor.concatenate_page_content_raw(
        df,
        metadata_cols=metadata_cols,
        exclude_cols=exclude_cols,
        exclude_col_if_contains=exclude_col_if_contains,
    )
    preprocessed_data = df_raw.copy()
    preprocessed_data["page_content"] = preprocessed_data["page_content_raw"].apply(preprocessor.preprocess)

    # Chunking logic needs to be added here
    converter = DocumentConverter()
    chunker = HybridChunker()

    jedish_docs = []
    for row_idx, row in preprocessed_data.iterrows():
        page_content = str(row["page_content"] or "").strip()

        if not page_content:
            continue

        if row["metadata"] is None or not isinstance(row["metadata"], dict):
            logger.warning(f"Row {row_idx} has invalid metadata: {row['metadata']}. Skipping.")
            continue

        # Base metadata is all metadata columns (see astra_csv_cols_config.py).
        # Should contain at least URL and Title, which are used downstream during generation.
        base_metadata = harmonize_astra_metadata({k: _df_to_json_safe_dict(v) for k, v in row["metadata"].items()})
        doc_name = base_metadata.get("Title", "No_Title").replace(" ", "_").replace(",", "")
        doc = converter.convert_string(content=page_content, format=InputFormat.MD, name=doc_name).document

        chunks = [chunk for chunk in chunker.chunk(dl_doc=doc)]
        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": chunk_idx,
                "chunk_id": f"{doc_name}_chunk{chunk_idx}",
                "docling_doc_name": doc_name,
            }

            jedish_json = {
                f"{doc_name}_chunk{chunk_idx}": {
                    "abstract": chunker.contextualize(chunk=chunk),
                    "metadata": chunk_metadata,
                }
            }
            jedish_docs.append(jedish_json)

    return jedish_docs
