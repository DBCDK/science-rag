import re
import html
import logging
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class AstraPreprocessor:
    """
    Preprocessing for cleaning and normalizing raw page content from Astra
    .csv exports, which may include:

    - HTML fragments
    - widgets/layout noise
    - URLs and file links
    - Duplicated text from activities or headings
    - Extra whitespaces

    Workflow should be:
    1. Concatenate relevant columns into 'page_content_raw' and 'metadata'.
    2. Apply `preprocess` to clean and normalize the text into 'page_content'.
    3. Validate cleaned 'page_content' using 'test_astra_preprocessor.py' functions.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_layout_noise: bool = True,
        deduplicate: bool = True,
        normalize_whitespace: bool = True,
    ):
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_layout_noise = remove_layout_noise
        self.deduplicate = deduplicate
        self.normalize_whitespace = normalize_whitespace

        # Layout patterns we want to remove
        self.layout_noise_patterns = [
            r"\bwidget-one-column\b",
            r"\bwidget-two-columns\b",
            r"\bwidget-three-columns\b",
            r"\bwidget\b",
            r"\ball-tabs\b",
            r"\belementor[-\w]*\b",
            r"\belementor-widget-container\b",
            r"\belementor-text-editor\b",
            r"\belementor-clearfix\b",
            r"\belementor-heading-title\b",
            r"\belementor-size-default\b",
            r"\bpanel-pane\b",
            r"\bpane-entity-field\b",
            r"\bpane-node-field-image\b",
            r"\bpane-node-body\b",
            r"\bbox-content\b",
            r"\bbox editor\b",
            r"\bdata-id\b",
            r"\bdata-element_type\b",
            r"\bdata-widget_type\b",
            r"\bheading\.default\b",
            r"\btext-editor\.default\b",
        ]

        self.generic_noise_patterns = [
            r"\blink\b",
            r"\bdownload\b",
            r"\bHent undervisningsforløbet\b",
            r"\bLæs mere om modellen her\b",
            r"\bLæs mere om modellen\b",
        ]

    # ---------------------------------------
    # --- Extract raw content + metadata ----
    # ---------------------------------------
    def concatenate_page_content_raw(
        self,
        df,
        metadata_cols: list[str] = None,
        exclude_cols: list[str] = None,
        exclude_col_if_contains: list[str] = None,
    ):
        """
        Concatenates the content of all string type columns in the DataFrame into a single column 'page_content_raw',
        excluding columns specified in `exclude_cols` or columns containing any substring in `exclude_col_if_contains`.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        metadata_cols (list[str]): List of column names to add to the metadata dictionary for each row.
        exclude_cols (list[str]): List of column names to exclude from concatenation.
        exclude_col_if_contains (list[str]): List of substrings; columns containing any of these substrings will be excluded.

        Returns:
        pd.DataFrame: The DataFrame with the new 'page_content_raw' and 'metadata' columns.
        """
        metadata_cols = metadata_cols or []
        exclude_cols = exclude_cols or []
        exclude_col_if_contains = exclude_col_if_contains or []

        missing_cols = [col for col in metadata_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Metadata columns {missing_cols} are not in the csv and are left out of the metadata")
            metadata_cols = [col for col in metadata_cols if col in df.columns]

        df = df.copy()
        df["page_content_raw"] = ""
        df["metadata"] = None
        for index, row in df.iterrows():
            page_content_parts = []
            for col in df.columns:
                if col in exclude_cols or col in metadata_cols:
                    continue
                if any(substring in col for substring in exclude_col_if_contains):
                    continue
                if isinstance(row[col], str):
                    page_content_parts.append(row[col])

            df.at[index, "page_content_raw"] = " ".join(page_content_parts)
            df.at[index, "metadata"] = {col: row[col] for col in metadata_cols}
        return df

    # ---------------------------------------
    # ----- Main preprocessing function -----
    # ---------------------------------------
    def preprocess(self, page_content_raw: Optional[str]) -> str:
        """
        Args:
            page_content_raw: The raw concatenated text/HTML from Astra .csv exports.

        Returns:
            page_content: The preprocessed/cleaned text.
        """

        if not isinstance(page_content_raw, str):
            return ""

        text = page_content_raw

        # 1. Remove HTML tags and decode entities
        text = self._decode_html_entities(text)
        if self.remove_html:
            text = self._remove_html(text)

        # 2. Remove URLs and file links
        if self.remove_urls:
            text = self._remove_urls(text)
        text = self._clean_pipe_artifacts(text)

        # 3. Remove layout and generic noise patterns
        if self.remove_layout_noise:
            text = self._remove_layout_noise(text)

        # 4. Normalizing spacing and repititions
        text = self._normalize_common_artifacts(text)
        text = self._clean_5e_heading_repetitions(text)
        text = self._remove_repeated_phrases(text)
        if self.deduplicate:
            text = self._deduplicate_chunks(text)
        text = self._fix_spacing_around_punctuation(text)
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        return text

    # ---------------------------------------
    # --------- Helper functions ------------
    # ---------------------------------------
    def _decode_html_entities(self, text: str) -> str:
        text = html.unescape(text)
        text = text.replace("\xa0", " ")
        text = text.replace("&nbsp;", " ")
        return text

    def _remove_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        return soup.get_text(separator=" ")

    def _remove_urls(self, text: str) -> str:
        # Full URLs
        text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"www\.\S+", " ", text, flags=re.IGNORECASE)

        # File names or URL fragments that may remain
        text = re.sub(
            r"\b\S+\.(?:pdf|jpg|jpeg|png|gif|webp|docx?|xlsx?|pptx?)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Common leftover Astra upload path fragments
        text = re.sub(
            r"\bwp-content/uploads/\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        return text

    def _clean_pipe_artifacts(self, text: str) -> str:
        # Converts "Title|https://...|" into cleaner text after URLs are removed.
        return text.replace("|", " ")

    def _remove_layout_noise(self, text: str) -> str:
        for pattern in self.layout_noise_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        for pattern in self.generic_noise_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        return text

    def _normalize_common_artifacts(self, text: str) -> str:
        # CO 2 -> CO2
        text = re.sub(r"\bCO\s+2\b", "CO2", text, flags=re.IGNORECASE)

        # 70 % -> 70%
        text = re.sub(r"\s+%", "%", text)

        # Remove isolated zeroes, often broken link labels
        text = re.sub(r"\b0\b", " ", text)

        return self._normalize_whitespace(text)

    def _clean_5e_heading_repetitions(self, text: str) -> str:
        """
        Simplify repeated 5E heading patterns.

        Examples:
        - "Forklar 5E-modellen: Forklar Forklar" -> "Forklar"
        - "Undersøg 5E-modellen: undersøg Undersøg" -> "Undersøg"
        """

        phase_variants = [
            "Engagér",
            "Engager",
            "Undersøg",
            "Forklar",
            "Evaluer",
            "Udvid",
            "Udvid og bearbejd",
        ]

        for phase in phase_variants:
            escaped = re.escape(phase)

            text = re.sub(
                rf"\b{escaped}\b\s+5E-modellen:?\s*[^.!?]{{0,50}}?\b{escaped}\b",
                phase,
                text,
                flags=re.IGNORECASE,
            )

        text = re.sub(
            r"\b(Engagér|Engager|Undersøg|Forklar|Evaluer|Udvid og bearbejd|Udvid)"
            r"(?:\s+\1\b)+",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

        return text

    def _remove_repeated_phrases(self, text: str, max_words: int = 8) -> str:
        """
        Remove adjacent duplicated phrases.

        Examples:
        - "Myrejagten Myrejagten" -> "Myrejagten"
        - "Dissektion af insekter Dissektion af insekter" -> "Dissektion af insekter"
        """

        word = r"[A-Za-zÆØÅæøå0-9][A-Za-zÆØÅæøå0-9’'.,:-]*"

        for n_words in range(max_words, 0, -1):
            pattern = rf"\b((?:{word}\s+){{{n_words - 1}}}{word})\s+\1\b"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        return text

    def _deduplicate_chunks(self, text: str) -> str:
        """
        Deduplicate sentence-like or section-like chunks while preserving order.
        """

        section_starters = [
            "Engagér",
            "Engager",
            "Undersøg",
            "Forklar",
            "Evaluer",
            "Udvid",
            "Udvid og bearbejd",
            "Faglige mål",
            "Faglige pointer",
            "Kompetenceområde",
            "Kompetenceområder",
            "Kort om forløbet",
            "Forløbet er struktureret efter 5E-modellen",
            "Flere artikler om området",
            "Interessante links",
            "Aktivitet:",
        ]

        starter_pattern = "|".join(re.escape(s) for s in section_starters)

        parts = re.split(
            rf"(?<=[.!?])\s+|(?=\b(?:{starter_pattern})\b)",
            text,
            flags=re.IGNORECASE,
        )

        seen = set()
        cleaned_parts = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            key = part.lower()
            key = re.sub(r"\s+", " ", key)
            key = re.sub(r"[^\wæøåÆØÅ0-9 ]+", "", key).strip()

            if len(key) < 3:
                continue

            if key not in seen:
                seen.add(key)
                cleaned_parts.append(part)

        return " ".join(cleaned_parts)

    def _fix_spacing_around_punctuation(self, text: str) -> str:
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.!?])([A-ZÆØÅ])", r"\1 \2", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
