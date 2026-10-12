import json
import unicodedata
from types import SimpleNamespace

import numpy as np
import pytest

from science_rag.preprocessing.link_map import chunk_url, extra_metadata, load_link_map
from science_rag.rag.retrievers.indexes.docling_indexer import get_docling_chunks
from science_rag.rag.retrievers.indexes.multilinguale5 import index_paragraph_docs_GPU_batches
from science_rag.tools import KNNSearch

LINK_MAP_CSV = """filename,relative_path,url,title,forloeb,afsender,fag,klassetrin,indskoling,mellemtrin,udskoling,laerervejledning
Styr-Lyden.pdf,NVHus/Styr lyden/Styr-Lyden.pdf,https://nvhus.dk/styr-lyden/,,Styr lyden,Naturvidenskabernes Hus,,,False,True,False,True
Styr-Lyden-Undersoegelse-1.pdf,NVHus/Styr lyden/Styr-Lyden-Undersoegelse-1.pdf,https://nvhus.dk/styr-lyden/,,Styr lyden,Naturvidenskabernes Hus,,,False,True,False,True
Det blå guld - Elevbog.pdf,Ole Haubo/Det blå guld - Elevbog.pdf,https://haubo.net/Det%20blå%20guld%20-%20Elevbog.pdf,Det blå guld - elevbog,,Ole Haubo,NT,7.-9.,,,,
metodekit_A4.pdf,Experimentarium/metodekit_A4.pdf,,,,Experimentarium,,,,,,
"""


@pytest.fixture
def link_map(tmp_path):
    path = tmp_path / "link_map.csv"
    path.write_text(LINK_MAP_CSV, encoding="utf-8")
    return load_link_map(path)


class FakeChunker:
    """Two chunks per document, on page 1 and 2"""

    def chunk(self, dl_doc):
        for page_no in (1, 2):
            yield SimpleNamespace(
                text=f"{dl_doc.name} side {page_no}",
                meta=SimpleNamespace(
                    origin=SimpleNamespace(filename=dl_doc.name),
                    doc_items=[SimpleNamespace(prov=[SimpleNamespace(page_no=page_no)])],
                ),
            )

    def contextualize(self, chunk):
        return chunk.text


class FakeConverter:
    def convert(self, path):
        return SimpleNamespace(document=SimpleNamespace(name=path.rsplit("/", 1)[-1]))


def chunks_for(filename, link_map):
    return get_docling_chunks(f"some/dir/{filename}", link_map, FakeConverter(), FakeChunker())


def test_chunk_url_only_anchors_pdf_links():
    assert chunk_url("https://haubo.net/a%20b.pdf", 3) == "https://haubo.net/a%20b.pdf#page=3"
    assert chunk_url("https://nvhus.dk/styr-lyden/", 3) == "https://nvhus.dk/styr-lyden/"
    assert chunk_url("https://undervisning.life.dk/fb", 3) == "https://undervisning.life.dk/fb"
    assert chunk_url("https://example.dk/doc.PDF?x=1", 2) == "https://example.dk/doc.PDF?x=1#page=2"


def test_extra_metadata_skips_empty_values_and_parses_bools(link_map):
    assert extra_metadata(link_map["Styr-Lyden.pdf"]) == {
        "Afsender": "Naturvidenskabernes Hus",
        "Forløb": "Styr lyden",
        "Indskoling": False,
        "Mellemtrin": True,
        "Udskoling": False,
        "Lærervejledning": True,
    }
    assert extra_metadata(link_map["Det blå guld - Elevbog.pdf"]) == {
        "Afsender": "Ole Haubo",
        "Fag": "NT",
        "Klassetrin": "7.-9.",
    }


def test_pdf_source_gets_page_anchor_and_title(link_map):
    docs = chunks_for("Det blå guld - Elevbog.pdf", link_map)
    metadata = [next(iter(d.values()))["metadata"] for d in docs]
    assert [m["URL"] for m in metadata] == [
        "https://haubo.net/Det%20blå%20guld%20-%20Elevbog.pdf#page=1",
        "https://haubo.net/Det%20blå%20guld%20-%20Elevbog.pdf#page=2",
    ]
    assert metadata[0]["Title"] == "Det blå guld - elevbog"


def test_web_source_has_no_page_anchor_and_falls_back_to_filename_title(link_map):
    docs = chunks_for("Styr-Lyden.pdf", link_map)
    metadata = next(iter(docs[0].values()))["metadata"]
    assert metadata["URL"] == "https://nvhus.dk/styr-lyden/"
    assert metadata["Title"] == "Styr-Lyden"


@pytest.mark.parametrize("filename", ["metodekit_A4.pdf", "not-in-link-map.pdf"])
def test_missing_url_falls_back_to_filename(link_map, filename):
    docs = chunks_for(filename, link_map)
    metadata = next(iter(docs[0].values()))["metadata"]
    assert metadata["URL"] == f"{filename}#page=1"
    assert metadata["Title"] == filename.replace(".pdf", "")


def test_lookup_is_unicode_normalized(link_map):
    decomposed = unicodedata.normalize("NFD", "Det blå guld - Elevbog.pdf")
    metadata = next(iter(chunks_for(decomposed, link_map)[0].values()))["metadata"]
    assert metadata["Title"] == "Det blå guld - elevbog"


def test_ids_are_unique_when_documents_share_url(link_map):
    docs = chunks_for("Styr-Lyden.pdf", link_map) + chunks_for("Styr-Lyden-Undersoegelse-1.pdf", link_map)
    ids = [next(iter(d)) for d in docs]
    assert len(set(ids)) == len(ids) == 4
    assert ids[0] == "Styr-Lyden.pdf_side1_chunk0"


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_index_uses_given_embedder(tmp_path):
    docs = [{f"doc{i}": {"abstract": "x" * (150 + i)}} for i in range(3)]
    index_file = tmp_path / "chunks.json"
    index_file.write_text(json.dumps(docs))

    index_paragraph_docs_GPU_batches(
        path=str(tmp_path), path_to_index_file=str(index_file), batch_size=2, embedder=FakeEmbedder()
    )

    searcher = KNNSearch.load(str(tmp_path / "embeddings"), str(tmp_path / "labels.npy"))
    assert list(np.load(tmp_path / "labels.npy")) == ["doc0", "doc1", "doc2"]
    assert searcher.index.ntotal == 3
