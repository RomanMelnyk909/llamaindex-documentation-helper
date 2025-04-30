from dotenv import load_dotenv
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if __name__ == "__main__":
    print("RAG...")

    pinecone_index = pc.Index("llanaindex-resume-pdf-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )

    query = "Write short summary about all this files"
    query_engine = index.as_query_engine()
    responce = query_engine.query(query)

    print(responce)

    print("File was run successfully....")