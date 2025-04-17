from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import download_loader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


if __name__ == '__main__':
  print(f"{os.environ['PINECONE_ENVIRONMENT']}")
  