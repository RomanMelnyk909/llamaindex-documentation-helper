from dotenv import load_dotenv
import os

from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from llama_index.readers.file import UnstructuredReader

from pathlib import Path

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# This code reed data from documents, divides it to small chunks, makes embeddings from this chunks an push it to vector db


if __name__ == "__main__":
    input_dir = Path("./llamaindex-docs/")

    loader = UnstructuredReader()

    html_files = input_dir.glob("*.html")

    documents = []
    for html_file in html_files:
        document = loader.load_data(unstructured_kwargs={"filename": str(html_file)})
        documents.extend(document)

    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=500, chunk_overlap=20
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = "llamaindex-documentation-helper"
    pinecone_index = pc.Index("llamaindex-documentation-helper")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"{os.environ['PINECONE_ENVIRONMENT']}")

    pass
