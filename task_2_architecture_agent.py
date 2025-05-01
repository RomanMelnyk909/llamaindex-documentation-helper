import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4", temperature=0.2)
Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)


def analyze_codebase(path: str):
    print("Reading codebase...")
    documents = SimpleDirectoryReader(input_dir=path, recursive=True).load_data()

    print("Building index...")
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    print("Generating documentation...")
    response = query_engine.query(
        """
        Generate a complete system architecture documentation in markdown format. 
        Follow this structure:
        
        ## Introduction
        
        ## Purpose & Scope
        
        ## System Overview
        
        ### Key Functionalities
        (List each feature with its name and a 2-3 sentence description.)
        
        ### Technology Stack
        - Programming Languages
        - Frameworks/Libraries
        - Databases
        - Message Queues
        - CI/CD Tools
        
        ### Codebase Structure
        - Repository Organization
        - Key Packages
        - Key Modules & Entry Points (e.g. API routes, CLI, etc.)

        Keep the output clean and markdown-formatted.
        """
    )

    return response.response


if __name__ == "__main__":
    markdown_doc = analyze_codebase("./repo")

    # with open("architecture.md", "w", encoding="utf-8") as f:
    #     f.write(markdown_doc)

    print("✅ Architecture doc saved to architecture.md")
