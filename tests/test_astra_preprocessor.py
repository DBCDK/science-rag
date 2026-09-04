import re
import pandas as pd

from science_rag.preprocessing.astra_preprocessor import AstraPreprocessor


URL_PATTERN = re.compile(
    r"""
    (
        https?://\S+
        |
        www\.\S+
        |
        \S+\.(?:pdf|jpg|jpeg|png|gif|webp|docx?|xlsx?|pptx?)\b
        |
        wp-content/uploads/\S+
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

HTML_PATTERN = re.compile(
    r"""
    (
        <[^>]+>
        |
        &nbsp;
        |
        &amp;
        |
        &lt;
        |
        &gt;
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def contains_url_pattern(value: str) -> bool:
    """
    Return True if the value still contains URL-like or file-link patterns.
    """

    if not isinstance(value, str):
        return False

    return bool(URL_PATTERN.search(value))


def contains_html_pattern(value: str) -> bool:
    """
    Return True if the value still contains HTML tags or common HTML entities.
    """

    if not isinstance(value, str):
        return False

    return bool(HTML_PATTERN.search(value))


def validate_page_content_value(value) -> dict:
    """
    Validate a single processed page_content value.

    Returns a dictionary with boolean checks.
    """

    return {
        "is_string": isinstance(value, str),
        "is_not_empty": isinstance(value, str) and bool(value.strip()),
        "contains_url_pattern": contains_url_pattern(value),
        "contains_html_pattern": contains_html_pattern(value),
    }


def validate_page_content_series(page_content: pd.Series) -> pd.DataFrame:
    """
    Validate an entire pandas Series containing processed page_content.

    Returns a dataframe where each row shows validation results.
    """

    validation_rows = []

    for index, value in page_content.items():
        checks = validate_page_content_value(value)
        checks["index"] = index
        validation_rows.append(checks)

    result = pd.DataFrame(validation_rows).set_index("index")

    result["passed"] = (
        result["is_string"]
        & result["is_not_empty"]
        & ~result["contains_url_pattern"]
        & ~result["contains_html_pattern"]
    )

    return result


def print_validation_summary(validation_result: pd.DataFrame) -> None:
    """
    Print a compact summary of validation results.
    """

    total_rows = len(validation_result)
    passed_rows = validation_result["passed"].sum()
    failed_rows = total_rows - passed_rows

    print(f"Total rows: {total_rows}")
    print(f"Passed rows: {passed_rows}")
    print(f"Failed rows: {failed_rows}")

    if failed_rows > 0:
        print("\nFailure counts:")
        print(validation_result.loc[~validation_result["passed"]].sum(numeric_only=True))


def get_failed_page_content_rows(
    df: pd.DataFrame,
    page_content_col: str = "page_content",
) -> pd.DataFrame:
    """
    Return rows where processed page_content fails validation.
    """

    validation_result = validate_page_content_series(df[page_content_col])
    failed_indices = validation_result.index[~validation_result["passed"]]

    return df.loc[failed_indices].copy()


def check_single_example(raw_text: str) -> None:
    """
    Quick manual test for a single raw text value.
    """

    preprocessor = AstraPreprocessor()
    cleaned = preprocessor.preprocess(raw_text)
    checks = validate_page_content_value(cleaned)

    print("Cleaned text:")
    print(cleaned)
    print("\nValidation checks:")
    print(checks)

    assert checks["is_string"]
    assert checks["is_not_empty"]
    assert not checks["contains_url_pattern"]
    assert not checks["contains_html_pattern"]


def check_dataframe_page_content(
    df: pd.DataFrame,
    raw_col: str = "page_content_raw",
    clean_col: str = "page_content",
) -> pd.DataFrame:
    """
    Preprocess and validate a dataframe.
    It is assumed that the dataframe has a column with raw page content
    and it will create a new column with your preferred name for the
    cleaned content (default is "page_content").
    """

    preprocessor = AstraPreprocessor()
    df[clean_col] = df[raw_col].apply(preprocessor.preprocess)

    validation_result = validate_page_content_series(df[clean_col])
    print_validation_summary(validation_result)

    return validation_result
