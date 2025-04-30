from dotenv import load_dotenv
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

import streamlit as st

# This code send query to vactor db with context data

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    print("RAG...")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    pinecone_index = pc.Index("llamaindex-documentation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager(handlers=[llama_debug])

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = get_index()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True
    )

st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",
    page_icon="W ",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with LlamaIndex docs")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex open source python library?",
        }
    ]


if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            responce = st.session_state.chat_engine.chat(message=prompt)
            st.write(responce.response)
            nodes = [node for node in responce.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source node {i+1}: csore= {node.score}")
                    st.write(node.text)
            message = {"role": "assistant", "content": responce.response}
            st.session_state.messages.append(message)
