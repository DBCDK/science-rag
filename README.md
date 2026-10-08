# Science-RAG

RAG-solution for PDFs from CFU. The repository uses the same RAG-structure as MitCFU-RAG, but instead of getting data from
MitCFU Marc entries, documents are indexed via `docling`'s `DocumentConverter` (see
`rag/retrievers/indexes/docling_indexer.py`), optionally enriched with Astra CSV metadata. From the resulting chunks we can
use the metadata and page_content to create FAISS indexes for similarity search. For now, we simply extract page_content and
create a JED-like structure to enable the rest of the MitCFU-pipeline to do the hard work.

`tools/generic_parser.py` additionally provides a `GenericParser` utility that loads common filetypes (such as .pdf, .txt,
.json, with a fallback for other types) via LangChain Documentloaders into a list of LangChain `Document` objects. It's a
building block for custom loading and is not currently wired into the docling-based indexing pipeline described below.

If you want to index your own documents using the current pipeline, see the description below in "So you want to index your
own documents ..."

## So you want to index your own documents ...
On your local machine:
1. Place all your documents (pdfs, text files, etc.) in a directory of your choice (I have mine locally under git/science-rag/data)
2. Make a directory where you want to store your embeddings/FAISS index such as `output_embedding_dir`
3. To parse the documents and create the faiss index, run

`python src/science_rag/rag/retrievers/indexes/docling_indexer.py path/to/all/documents  name_of_output_chunk_document.json output_embedding_dir/ --link-map path/to/science_rag_link_map.csv`

`--link-map` is a csv with one row per document, keyed on `filename`, that gives each document its source and metadata
(see `preprocessing/link_map.py`). `url` and `title` become `URL` and `Title` in the chunk metadata (`#page=N` is only
added to pdf links), and the columns `afsender`, `fag`, `klassetrin`, `forloeb`, `indskoling`, `mellemtrin`, `udskoling`
and `laerervejledning` are added when they have a value. Documents without a row or url fall back to their filename.

you can optionally include Astra .csv files (to be preprocessed, chunked and indexed together with the documents) using the following flags: 

`python src/science_rag/rag/retrievers/indexes/docling_indexer.py path/to/all/documents  name_of_output_chunk_document.json output_embedding_dir/ --link-map path/to/science_rag_link_map.csv --aktiviteter-csv path/to/aktiviteter.csv --forlob-csv path/to/forlob.csv`

The Astra exports have no url column, so each aktivitet/forløb gets `URL` = `https://astra.dk/?p=<ID>`, built from the
`ID` column (the WordPress post ID, which astra.dk redirects to the page). This happens in `harmonize_astra_metadata` in
`preprocessing/astra_df_to_chunked_docs.py`, which also adds `Afsender` and rewrites `Fag`/`Klassetrin` to the link map
format. Rows without a valid `ID` get no `URL`.

By default the chunks are embedded locally with `multilingual-e5-large-instruct`. To embed with a remote
Glyphgate embeddings endpoint instead, add `--embedding-endpoint http://glyph-gate-1-0.ai-prod.svc.cloud.dbc.dk/v1` (a full
`.../v1/embeddings` url works too). The api key, if the endpoint needs one, is read from the environment variable
`GLYPHGATE_API_KEY`, and `--embedding-model` sets the model name sent to the endpoint (default
`intfloat/multilingual-e5-large-instruct`). The index must be embedded with the same model the service uses for queries.

4. When starting the RAG service, point to the location of the json file list and the FAISS index (as well as embedding/validator models).
5. You can start the service using: 

`streaming-service-science-rag /data/mitCFU-models/multilingual-e5-large output_embedding_dir --article_index_path name_of_output_chunk_document --validator-model-path /data/mitCFU-models/ms-marco-MiniLM-L-6-v2 
--verbose --port 5011`

The service is FastAPI/uvicorn-based; once it's running, interactive API docs are available at
`/docs` and `/redoc` (e.g. `http://localhost:5011/docs`).

6. You can then start Streamlit using:
streamlit run src/science_rag/streamlit_ui.py --server.port 8111
7. Hooray! You should now be able to query your own documents using RAG.

If you have further questions about the process, ask rani or nily for their notebook example for the PDFs from CFU.

## How to start the service with Docker
Build the docker image from the Dockerfile:

`docker build -t $USER/science-rag:test -f Dockerfile .`

The build downloads the validator model and FAISS index/document-chunks bundle from Artifactory
via `MODEL_PATH`/`FAISS_PATH`/`INDEX_PATH` build args (see `Dockerfile`); these resolve from
`ARTIFACTORY_URL`/`AI_PRODUCTION`/`AI_DOCKER_LAYERS`, which CI (the `buildImage()` step in
`Jenkinsfile`) sets automatically. Building locally requires those set in your environment too.
The image's `uv sync` also installs the `dbc` dependency group (`dbc_pyutils`), which a plain
local `uv sync`/`pytest` does not — see "Optional `dbc_pyutils` integration" below.

The embedding model itself is **not** baked into the image — `/data/science-rag-1-0` inside the
container is expected to be a symlink/mount to the `multilingual-e5-large-instruct` model
(on k8s this comes from a volume mount). Run the image with that path mounted:

`docker run -v /path/to/multilingual-e5-large-instruct:/data/science-rag-1-0 -p 5011:5000 -it $USER/science-rag`

`--rm` ensures the docker container is closed down properly after use.

If you have started the service on your local machine you can reach it via this url:
`http://localhost:<PORT_NUMBER>`

or locally via this url:
`localhost:<PORT_NUMBER>`

## Optional `dbc_pyutils` integration
The service works fully without any internal DBC package installed — a plain `uv sync`/`pip
install -e .` gives you the whole RAG pipeline and API. `dbc_pyutils` (installed via `uv sync
--group dbc`, DBC network access required) unlocks a few extras used on DBC cluster
hardware/CI, gated automatically at startup:

- **`/status`**: without `dbc_pyutils`, a bare `{"status": "ok"}`. With it, an enriched payload
  with build/git/version info, an instance id, memory usage, and query statistics.
- **`/metrics`**: omitted entirely without `dbc_pyutils`. With it, a Prometheus-format scrape
  endpoint.
- **Logging**: without `dbc_pyutils`, plain stdlib logging (`logging.basicConfig`). With it,
  structured JSON logs. Nothing is configurable via an env var either way.

## Create faiss embeddings (alternate indexing path)
`create-faiss-index` is a simpler, direct embed-and-index CLI (`rag/retrievers/indexes/multilinguale5.py`),
separate from the `docling_indexer.py` pipeline described above — it does not do PDF/CSV parsing,
it expects documents already extracted into a folder. Run `pip install -e .` and
`conda install -c conda-forge faiss` first, same as above, then:

`create-faiss-index --path-to-db path/to/faiss-db --path-to-folder path/to/documents --path-to-index-file path/to/index-file --batch-size 10 --create-new-index-extract`

This indexes the documents in `--path-to-folder` and saves the FAISS index and labels under
`--path-to-db`.

## How to run tests for this project
### Unit tests
Run `pytest` from the repo root (`tests/`, configured via `pyproject.toml`'s
`[tool.pytest.ini_options]`).

### Validation tests
None exist today. The `evaluate`/`evaluate-retrieval`/`compare-retrievers` tools that used to
serve this role were removed as dead code (they imported nonexistent packages and never ran) —
this is open work, not a regression.

### Performance tests
None exist today — same status as validation tests above.

### Sanity checks before Merge Request
- `pytest`
- `ruff check`
- `pip-audit`
- CI (`Jenkinsfile`) already runs the test step automatically on every push.

## Artifacts built in this project
A Docker image built from `Dockerfile`, deployed as `science-rag-1-0` (see `Jenkinsfile`'s
`SCIENCE_RAG_1_0_VERSION` gitops variable, set for both `ai-staging` and `ai-prod`).

## Related Jenkins jobs on is.dbc.dk
The pipeline defined in this repo's `Jenkinsfile`: runs tests, builds the Docker image, and on
`main` rolls out `deployment/science-rag-1-0` to `ai-staging` then `ai-prod`.

## Related artifacts from Artifactory
- Validator model: `mitcfu-rag/ms-marco-MiniLM-L-6-v2.tgz` (shared with MitCFU-RAG)
- FAISS index: `science-rag/science_rag_delivery_two_index.tgz`
- Document chunks: `science-rag/science_rag_delivery_two_document_chunks.json`

(paths relative to the Artifactory roots configured via `ARTIFACTORY_URL`/`AI_PRODUCTION`/
`AI_DOCKER_LAYERS` — see `Dockerfile`)

## Related repositories
MitCFU-RAG (mitcfu-rag) — this repo shares its RAG pipeline structure and validator model.

## Production version of the service
_TODO_: link the live production URL here.

## Documentation on confluence
_TODO_: link the Confluence page(s) for this project here.
