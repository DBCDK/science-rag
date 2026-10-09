import json

import pytest

from science_rag.preprocessing.astra_df_to_chunked_docs import harmonize_astra_metadata
from science_rag.preprocessing.link_map import load_astra_links

ASTRA_LINKS = {"Foretag en hjertedissektion": "https://astra.dk/aktiviteter/foretag-en-hjertedissektion/"}


@pytest.mark.parametrize(
    "fag, expected",
    [
        ("Grundskole>Biologi", "Biologi"),
        ("Grundskole>Biologi|Grundskole>Fysik/Kemi|Grundskole>Natur/teknologi", "Biologi, Fysik/Kemi, Natur/teknologi"),
        ("EUD>Biologi|Grundskole", "Biologi"),
        ("Grundskole>Fysik|Gymnasiale uddannelser>Fysik", "Fysik"),
    ],
)
def test_fag(fag, expected):
    assert harmonize_astra_metadata({"Fag": fag})["Fag"] == expected


def test_level_only_fag_is_dropped():
    assert "Fag" not in harmonize_astra_metadata({"Fag": "Grundskole"})


@pytest.mark.parametrize(
    "klassetrin, expected, booleans",
    [
        ("Mellemtrin", "4.-6.", (False, True, False)),
        ("Mellemtrin|Udskoling", "4.-9.", (False, True, True)),
        ("Indskoling|Mellemtrin|Udskoling", "0.-9.", (True, True, True)),
        ("Indskoling|Udskoling", "0.-3., 7.-9.", (True, False, True)),
        ("EUD|Udskoling", "7.-9., EUD", (False, False, True)),
        ("Gymnasium", "Gymnasium", (False, False, False)),
    ],
)
def test_klassetrin(klassetrin, expected, booleans):
    metadata = harmonize_astra_metadata({"Klassetrin": klassetrin})
    assert metadata["Klassetrin"] == expected
    assert (metadata["Indskoling"], metadata["Mellemtrin"], metadata["Udskoling"]) == booleans


def test_afsender_and_missing_values():
    metadata = harmonize_astra_metadata(
        {"Title": "Bakterier", "URL": "https://astra.dk/x", "Fag": None, "Klassetrin": None}
    )
    assert metadata == {"Title": "Bakterier", "URL": "https://astra.dk/x", "Afsender": "Astra"}


@pytest.mark.parametrize("post_id", [2623, 2623.0, "2623"])
def test_url_from_id(post_id):
    assert harmonize_astra_metadata({"ID": post_id})["URL"] == "https://astra.dk/?p=2623"


@pytest.mark.parametrize("post_id", [None, "", "abc"])
def test_url_left_out_without_valid_id(post_id):
    assert "URL" not in harmonize_astra_metadata({"ID": post_id})


def test_existing_url_is_kept():
    assert harmonize_astra_metadata({"ID": 2623, "URL": "https://astra.dk/x"})["URL"] == "https://astra.dk/x"


def test_url_from_astra_links():
    metadata = harmonize_astra_metadata({"ID": 2636, "Title": "Foretag en hjertedissektion"}, astra_links=ASTRA_LINKS)
    assert metadata["URL"] == "https://astra.dk/aktiviteter/foretag-en-hjertedissektion/"


def test_astra_links_takes_precedence_over_existing_url():
    metadata = harmonize_astra_metadata(
        {"Title": "Foretag en hjertedissektion", "URL": "https://astra.dk/x"}, astra_links=ASTRA_LINKS
    )
    assert metadata["URL"] == "https://astra.dk/aktiviteter/foretag-en-hjertedissektion/"


def test_title_not_in_astra_links_falls_back_to_id():
    metadata = harmonize_astra_metadata({"ID": 2623, "Title": "Ukendt"}, astra_links=ASTRA_LINKS)
    assert metadata["URL"] == "https://astra.dk/?p=2623"


def test_astra_links_lookup_is_unicode_normalized():
    decomposed = "Ga\u030adefulde verden - halvleder"  # "å" as "a" + combining ring
    links = {"G\u00e5defulde verden - halvleder": "https://astra.dk/forlob/gaadefulde-verden-halvleder/"}
    metadata = harmonize_astra_metadata({"Title": decomposed}, astra_links=links)
    assert metadata["URL"] == "https://astra.dk/forlob/gaadefulde-verden-halvleder/"


def test_load_astra_links(tmp_path):
    path = tmp_path / "astra_links.json"
    path.write_text(json.dumps(ASTRA_LINKS, ensure_ascii=False), encoding="utf-8")
    assert load_astra_links(path) == ASTRA_LINKS
