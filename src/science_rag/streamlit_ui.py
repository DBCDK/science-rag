import torch

torch.classes.__path__ = []  # type: ignore

import os
import time
import random
import json
import streamlit as st
import requests

from science_rag.tools.llm_formatting import gen_wrapper, select_model_function, GEMMA_3_12B, MIXTRAL_8X7B

# from fakta_chat.config import RAG

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_img_path = os.path.join(current_dir, "faktalink_icon.png")
STREAMING_PORT = os.environ.get("STREAMING_PORT", "5009")
STREAMING_ENDPOINTS = {
    "tgi": f"http://ai-p301:{STREAMING_PORT}",
    "vllm": f"http://ai-p301:{STREAMING_PORT}/v1/chat/completions",
}
STREAMING_BACKEND = os.environ.get("SCIENCE_RAG_UI_STREAM_BACKEND", "tgi").lower()
if STREAMING_BACKEND not in STREAMING_ENDPOINTS:
    STREAMING_BACKEND = "tgi"
STREAMING_ENDPOINT = STREAMING_ENDPOINTS[STREAMING_BACKEND]


version = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if not st.session_state:
    chat_history = None


def clear_chat_history():
    st.session_state.messages = []
    global chat_history
    chat_history = None


def chat():
    st.session_state.messages = chat_history


def stream_tokens(response, model_name):
    model_function = select_model_function(model_name)
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        payload = line.strip()
        if payload.startswith("data:"):
            payload = payload[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
            yield model_function(obj)
        except json.JSONDecodeError:
            continue



st.sidebar.button("New Chat", on_click=clear_chat_history)

st.image(relative_img_path, width=150)


greeting = "Hej 👋 Jeg er MitCFU-RAG og jeg kan hjælpe dig med at finde information om materialer\
              fra MitCFU. Men indtil videre er jeg vist stadig mest en kopi af FaktaChat. \
              \n\nHvad kan jeg hjælpe dig med?"

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "greeting", "content": greeting}]

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="🤓"):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# React to user input
if prompt := st.chat_input("Indsæt dit spørgmål her ..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🤓"):
        fillers = [
            "Lad mig finde relevant information i mitcfu ...",
            "Lad mig se...",
            "Et øjeblik...",
            "Vent lige...",
            "Hmm, lad mig finde noget...",
        ]
        with st.spinner(random.choice(fillers)):
            references = []
            payload = {"messages": st.session_state.messages}
            if STREAMING_BACKEND == "vllm":
                payload["stream"] = True

            response_stream = requests.post(
                STREAMING_ENDPOINT,
                json=payload,
                stream=True,
            )
            response_stream.raise_for_status()
            response = st.write_stream(stream_tokens(response_stream, GEMMA_3_12B))

            st.session_state.messages.append({"role": "assistant", "content": response})
