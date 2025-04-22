from dotenv import load_dotenv
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManage

# This code send query to vactor db with context data

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if __name__ == "__main__":
    print("RAG...")

    pinecone_index = pc.Index("llamaindex-documentation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager(handlers=[llama_debug])

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )

    query = "What is Llamaindex query engine?"
    query_engine = index.as_query_engine()
    responce = query_engine.query(query)

    print(responce)

    print("File was run successfully....")
