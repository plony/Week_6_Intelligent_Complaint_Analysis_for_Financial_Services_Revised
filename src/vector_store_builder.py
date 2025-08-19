# src/vector_store_builder.py

import os
import sys
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add the project's root directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_model import EmbeddingModel
from src.preprocessing import filter_and_clean_data
from src.vector_store_manager import VectorStoreManager


def build_vector_store():
    """
    Orchestrates the process of loading, cleaning, chunking, embedding,
    and indexing the complaint data.
    """
    # Define file paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, "data", "complaints.csv")
    vector_store_dir = os.path.join(project_root, "vector_store")
    
    # Check if a vector store already exists
    if os.path.exists(vector_store_dir):
        print("Vector store already exists. Skipping build process.")
        return

    # Load and clean the raw data
    try:
        raw_df = pd.read_csv(raw_data_path)
        filtered_df = filter_and_clean_data(raw_df)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # Take a smaller sample to prevent MemoryError
    # Adjust the number as needed based on your system's RAM
    sample_size = 15000
    if len(filtered_df) > sample_size:
        filtered_df = filtered_df.sample(n=sample_size, random_state=42)
        print(f"Dataset sampled down to {len(filtered_df)} records.")

    # Load DataFrame into LangChain documents
    loader = DataFrameLoader(
        filtered_df, page_content_column="cleaned_narrative"
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents for processing.")

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    # Initialize embedding model and vector store manager
    embedding_model = EmbeddingModel()
    vector_store_manager = VectorStoreManager(embedding_model)

    # Build and save the vector store
    vector_store_manager.build_and_save_index(docs, vector_store_dir)


if __name__ == "__main__":
    build_vector_store()