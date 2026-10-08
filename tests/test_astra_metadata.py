import pytest

from science_rag.preprocessing.astra_df_to_chunked_docs import harmonize_astra_metadata


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
