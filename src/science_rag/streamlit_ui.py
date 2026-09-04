import torch

torch.classes.__path__ = []  # type: ignore

import os
import random
import streamlit as st
from openai import OpenAI

from science_rag.config import DEFAULT_MODEL

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_img_path = os.path.join(current_dir, "faktalink_icon.png")
STREAMING_ENDPOINTS = {
    "vllm": "http://ai-p301:5009/v1/chat/completions",
}
STREAMING_BACKEND = os.environ.get("SCIENCE_RAG_UI_STREAM_BACKEND", "vllm").lower()
if STREAMING_BACKEND not in STREAMING_ENDPOINTS:
    STREAMING_BACKEND = "vllm"
STREAMING_ENDPOINT = STREAMING_ENDPOINTS[STREAMING_BACKEND]


def _base_url(url: str) -> str:
    """openai.OpenAI wants the `.../v1` base and appends `chat/completions` itself."""
    return url.removesuffix("/chat/completions")


client = OpenAI(base_url=_base_url(STREAMING_ENDPOINT), api_key="unused")

version = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if not st.session_state:
    chat_history = None


def clear_chat_history():
    st.session_state.messages = []
    global chat_history
    chat_history = None


def chat():
    st.session_state.messages = chat_history


st.sidebar.button("New Chat", on_click=clear_chat_history)

st.image(relative_img_path, width=150)


greeting = "Hej 👋 Jeg er Science-RAG og jeg kan hjælpe dig med at finde information om materialer\
              fra MitCFU og andre kilder, der omhandler naturvidenskab. \
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
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=st.session_state.messages,
                stream=True,
            )
            response = st.write_stream(
                chunk.choices[0].delta.content
                for chunk in stream
                if chunk.choices and chunk.choices[0].delta.content
            )

            st.session_state.messages.append({"role": "assistant", "content": response})
