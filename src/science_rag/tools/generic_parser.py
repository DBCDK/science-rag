import requests
import tempfile
import os
import logging
from urllib.parse import urlparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_unstructured.document_loaders import UnstructuredLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericParser:
    def __init__(self, allowed_domains=None):
        if allowed_domains is None:
            allowed_domains = {"mitcfu.dk", "dbc.dk"}
        self.allowed_domains = allowed_domains

    @staticmethod
    def extract_metadata(JED_dict: dict, _: int) -> dict:
        try:
            title = JED_dict.get("titles", {}).get("main", [None])[0]
            material_type = JED_dict.get("materialTypes", [{}])[0].get("general", {}).get("display")
            publication_date = (
                JED_dict.get("manifestations", {})
                .get("all", [{}])[0]
                .get("edition", {})
                .get("publicationDateForRanking")
            )
            persons = JED_dict.get("manifestations", {}).get("all", [{}])[0].get("contributors", {}).get("persons", [])
            person_names = set()
            for person in persons:
                if person.get("firstName") or person.get("lastName"):
                    person_names.add(f"{person.get('firstName', '')} {person.get('lastName', '')}")

            metadata_dict = {
                "mainTitle": title,
                "materialTypes": material_type,
                "publicationDateForRanking": publication_date,
                "contributors": list(person_names),
            }

            return metadata_dict

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    def is_safe_pdf_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        if (
            parsed_url.scheme == "https"
            and parsed_url.path.endswith(".pdf")
            and parsed_url.hostname in self.allowed_domains
        ):
            return True
        else:
            return False

    def _load_pdf_from_url(self, url) -> list:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type:
                logger.info(
                    f"Rejected {url} since Content-Type in response.headers does not contain 'pdf'. Actual type: {content_type}"
                )
                return []

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            loader = PyPDFLoader(file_path=tmp_path)
            documents = loader.load()
            os.remove(tmp_path)

            for doc in documents:
                doc.metadata["source"] = url

            return documents

        except Exception as e:
            logger.info(f"Failed to load from {url}: {e}")
            return []

    def _load_pdf_from_path(self, path) -> list:
        try:
            loader = PyPDFLoader(file_path=path)
            documents = loader.load()
            return documents

        except Exception as e:
            logger.info(f"Failed to load from {path}: {e}")
            return []

    def _load_txt_from_path(self, path) -> list:
        try:
            loader = TextLoader(file_path=path)
            documents = loader.load()
            return documents

        except Exception as e:
            logger.info(f"Failed to load from {path}: {e}")
            return []

    def _load_JED_from_path(self, JED_path) -> list:
        try:
            filename = os.path.basename(JED_path)
            filename_without_json = os.path.splitext(filename)[0]
            jq_schema_and_filekey = f'.["{filename_without_json}"]'  # .abstract | join(" ")'
            loader = JSONLoader(
                file_path=JED_path,
                jq_schema=jq_schema_and_filekey,
                content_key='.abstract | join(" ")',
                is_content_key_jq_parsable=True,
                metadata_func=self.extract_metadata,
                text_content=False,
            )
            documents = loader.load()

            file_url = f"https://mitcfu.dk/MaterialeInfo/?faust={filename_without_json}"
            for doc in documents:
                doc.metadata["source"] = file_url

            return documents

        except Exception as e:
            logger.info(f"Failed to load from {JED_path}: {e}")
            return []

    def _load_generic_from_path(self, path) -> list:
        try:
            loader = UnstructuredLoader(file_path=path)
            documents = loader.load()
            return documents

        except Exception as e:
            logger.info(f"Failed to load from {path}: {e}")
            return []

    # dispatching to URL or PATH
    def dispatch(self, url_or_path: str):
        # if the item is an URL
        if url_or_path.startswith("http"):
            if self.is_safe_pdf_url(url_or_path):
                documents = self._load_pdf_from_url(url_or_path)
                return documents
            else:
                return []

        # if the item is a path, we check if it is a pdf, txt, or json. If none of these, we use the unstructured/generic loader.
        elif os.path.exists(url_or_path):
            if url_or_path.lower().endswith(".pdf"):
                return self._load_pdf_from_path(url_or_path)
            elif url_or_path.lower().endswith(".txt"):
                return self._load_txt_from_path(url_or_path)
            elif url_or_path.lower().endswith(".json"):
                return self._load_JED_from_path(url_or_path)
            else:
                return self._load_generic_from_path(url_or_path)

        else:
            logger.info(f"Invalid or missing URL/path: {url_or_path}")
            return []

    def parse_all(self, urls_or_paths_list: list[str]):
        all_docs = []
        for url_or_path in urls_or_paths_list:
            docs = self.dispatch(url_or_path)
            all_docs.extend(docs)
        return all_docs


# example usage. Give a list of urls or paths and use the GenericParser to get a list of Document objects.
# Document objects can be used for embedding and reference purposes in RAG solutions.

if __name__ == "__main__":
    inputs = [
        "https://mitcfu.dk/pv/TV0000129447.pdf",
        "https://mitcfu.dk/pv/TV0000129500.pdf",
        "./TV0000129447.pdf",
        "facebok.com/definetely-not-a-phishing-link",
        "./TV0000129447.json",
    ]

    generic_parser = GenericParser()
    all_documents = generic_parser.parse_all(inputs)
