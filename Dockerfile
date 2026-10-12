FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3:latest

RUN apt-get update && apt-get install -y --no-install-recommends wget

ARG INDEX_FILEPATH=science_rag_delivery_three
ARG DOCUMENT_CHUNK_FILEPATH=science_rag_delivery_three_document_chunks.json
ARG MODEL_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/mitcfu-rag/ms-marco-MiniLM-L-6-v2.tgz
ARG FAISS_PATH=${ARTIFACTORY_URL}/${AI_DOCKER_LAYERS}/science-rag/${INDEX_FILEPATH}.tgz
ARG INDEX_PATH=${ARTIFACTORY_URL}/${AI_DOCKER_LAYERS}/science-rag/${DOCUMENT_CHUNK_FILEPATH}

RUN useradd -m python
USER python
WORKDIR /home/python

COPY --chown=python src src
COPY --chown=python pyproject.toml pyproject.toml
COPY --chown=python uv.lock uv.lock

RUN wget -nv --no-check-certificate ${MODEL_PATH} -O ms-marco-MiniLM-L-6-v2.tgz && \
    wget -nv --no-check-certificate ${FAISS_PATH} -O ${INDEX_FILEPATH}.tgz && \
    wget -nv --no-check-certificate ${INDEX_PATH} -O ${DOCUMENT_CHUNK_FILEPATH} && \
    tar -xzvf ms-marco-MiniLM-L-6-v2.tgz && \
    tar -xzvf ${INDEX_FILEPATH}.tgz && \
    rm ms-marco-MiniLM-L-6-v2.tgz && \
    rm ${INDEX_FILEPATH}.tgz && \
    ln -s ${INDEX_FILEPATH} faiss_index && \
    ln -s ${DOCUMENT_CHUNK_FILEPATH} document_chunks.json && \
    uv sync --no-dev --frozen --group dbc

# Ensure uv env is on path
ENV PATH="/home/python/.venv/bin:$PATH"
ENV SCIENCE_RAG_VLLM_MODEL="google/gemma-4-26B-A4B-it"
# /data/science-rag-1-0 is a symlink to the model (multilingual-e5-large-instruct) on the k8s volume mount
# faiss_index and document_chunks.json are symlinks to ${INDEX_FILEPATH} / ${DOCUMENT_CHUNK_FILEPATH}
CMD ["streaming-service-science-rag", "/data/science-rag-1-0", "faiss_index", "--article_index_path", "document_chunks.json", "--validator-model-path", "ms-marco-MiniLM-L-6-v2", "--port", "5000"]
EXPOSE 5000
