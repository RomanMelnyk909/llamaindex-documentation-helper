from dotenv import load_dotenv
import os

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from pathlib import Path

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if __name__ == "__main__":
    input_dir = Path("./resume-pdf/")

    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
    )

    all_docs = []
    for docs in reader.iter_data():
        all_docs.extend(docs)

    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=500, chunk_overlap=20
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = "llanaindex-resume-pdf-helper"
    pinecone_index = pc.Index("llanaindex-resume-pdf-helper")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=all_docs,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"{os.environ['PINECONE_ENVIRONMENT']}")

    pass

    # for pdf_file in pdf_files:
    #   for file in pdf_files:
    #     documents.append(file.name)
    # print(documents)
