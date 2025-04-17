from dotenv import load_dotenv
import os

from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone

from llama_index.readers.file import UnstructuredReader

from pathlib import Path

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


if __name__ == "__main__":
    input_dir = Path("./docs/")

    # UnstructuredReader = download_loader("UnstructuredReader")
    # dir_reader = SimpleDirectoryReader(
    #     input_dir=str(input_dir), file_extractor={".html": UnstructuredReader}
    # )

    loader = UnstructuredReader()

    html_files = input_dir.glob("*.html")

    documents = []
    for html_file in html_files:
        document = loader.load_data(unstructured_kwargs={"filename": str(html_file)})
        documents.extend(document)

  # ********************************************************************
    # documents = dir_reader.load_data()
    # node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    # embeded_model = OpenAIEmbedding(
    #     model="text-embedding-ada-002", embed_batch_size=100
    # )
    # service_context = ServiceContext.from_defaults(
    #     llm=llm, embeded_model=embeded_model, node_parser=node_parser
    # )

    # index_name = "llamaindex-documentation-helper"
    # pinecone_index = pinecone.Index(index_name=index_name)

    # vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # index = VectorStoreIndex.from_documents(
    #     documents=documents,
    #     storage_context=storage_context,
    #     service_context=service_context,
    #     show_progress=True,
    # )
        
  # ********************************************************************
        
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=500, chunk_overlap=20
    )

    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

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
