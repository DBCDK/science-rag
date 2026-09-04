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
On ai-p301:
1. Place all your documents (pdfs, text files, etc.) in a directory of your choice (I have mine locally under git/science-rag/data)
2. Make a directory where you want to store your embeddings/FAISS index such as `output_embedding_dir`
3. To parse the documents and create the faiss index, run

`python src/science_rag/rag/retrievers/indexes/docling_indexer.py path/to/all/documents  name_of_output_chunk_document.json output_embedding_dir/`

you can optionally include Astra .csv files (to be preprocessed, chunked and indexed together with the documents) using the following flags: 

`python src/science_rag/rag/retrievers/indexes/docling_indexer.py path/to/all/documents  name_of_output_chunk_document.json output_embedding_dir/ --aktiviteter-csv path/to/aktiviteter.csv --forlob-csv path/to/forlob.csv`

4. When starting the RAG service, point to the location of the json file list and the FAISS index (as well as embedding/validator models).
5. You can start the service using: 

`streaming-service-science-rag /data/mitCFU-models/multilingual-e5-large output_embedding_dir --article_index_path name_of_output_chunk_document --validator-model-path /data/mitCFU-models/ms-marco-MiniLM-L-6-v2 
--verbose --port 5011`

6. You can then start Streamlit using:
streamlit run src/science_rag/streamlit_ui.py --server.port 8111
7. Hooray! You should now be able to query your own documents using RAG.

If you have further questions about the process, ask rani or nily for their notebook example for the PDFs from CFU.

## NOTE: The rest of the documentation is from MitCFU-RAG.
Many of commands should be analogous, but see the guide above for greater clarity.

## How to start the service from the command line
The easiest way to test is to start two services: the streaming service and the streamlitui service
Before starting them, make sure you have done the following:

Checkout the project on `ai-p301`. This is currently the only place where the files/models are
Run `pip install -e .` from the root of the project on
Run `conda install -c conda-forge faiss` to install faiss

### Starting the streaming service
Start the service with the following parameters
`streaming-service-mitcfu /data/mitCFU-models/multilingual-e5-large /data/mitCFU/faiss-indexes/mitcfu_faiss_index_with_pedagogical_note/ -p 5011 --article_index_path /data/mitCFU/faiss-indexes/mitcfu_faiss_index_file_with_pedagogical_note`

### Starting the streamlitui service
Start the service with the following parameters.
`streamlit run src/science_rag/streamlit_ui.py --server.port 8111`
NOTE: If you did not run your streaming service on port 5011, you will have to manually change the endpoint by editing
the variable `STREAMING_ENDPOINT` in `streamlit_ui.py`

## How to start the service with DOCKER
Build the docker image from the Dockerfile:

`docker build -t $USER/<my-service-name>:test -f Dockerfile .`

Run the docker image:

`docker run -v /data/mitCFU-models/multilingual-e5-large:/data/mitcfu-rag-1-0 -p 5011:5000 -it $USER/mitcfu-rag`

`-e LOG_FORMAT=text` gives you log output in text instead of json

`--rm` ensures the docker container is closed down properly after use

If you for example have started the service on the server ai-p301 you can reach the service via this url
`http://ai-p301:<PORT_NUMBER>`

or locally via this url: 

`localhost:<PORT_NUMBER>`

## Create faiss embeddings
### How to run <my_command>
Run `pip install -e .` from the root of the project.
Run `conda install -c conda-forge faiss`

First run `touch mitcfu-index-file-path`
Then run `create-faiss-index --path_to_db mitcfu-faiss-db --path_to_folder /data/mitcfu-rag/jed-docs --path_to_index_file mitcfu-index-file-path --batch_size 10 --create_new_index_extract`

This will start the indexing of the documents in the folder "--path_to_folder" and save the FAISS index and labels in the path specified after "--path_to_db".
"""

## How to run tests for this project
### Unit tests 

### Validation tests

### Performance tests

### Sanity checks before Merge Request


## Artifacts built in this project


## Related Jenkins jobs on is.dbc.dk


## Related artifacts from Artifactory


## Related repositories
MitCFU-RAG (mitcfu-rag).


## Production version of the service
 
 
## Documentation on confluence
