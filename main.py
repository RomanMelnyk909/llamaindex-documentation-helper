from dotenv import load_dotenv
import os
from pinecone import Pinecone


load_dotenv()

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

if __name__ == "__main__":
  print('RAG...')