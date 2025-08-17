# src/vector_store_builder.py
# src/vector_store_builder.py

import os
import sys
import pandas as pd

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_complaints_data
from src.preprocessing import filter_and_clean_data
from src.embedding_model import EmbeddingModel
from src.vector_store_manager import VectorStoreManager

if __name__ == "__main__":
    raw_data_path = "data/complaints.csv"

    # 1. Load and preprocess the data
    print("--- STEP 1: LOADING AND PREPROCESSING DATA ---")
    df = load_complaints_data(raw_data_path)
    if df is None:
        sys.exit("Data loading failed. Exiting.")

    cleaned_df = filter_and_clean_data(df)
    if cleaned_df is None:
        sys.exit("Data preprocessing failed. Exiting.")

    # Save the cleaned dataframe for future use
    output_path = "data/filtered_complaints.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nCleaned and filtered dataset saved to {output_path}")

    # 2. Initialize embedding model and vector store manager
    print("\n--- STEP 2: BUILDING THE VECTOR STORE ---")
    embedding_model = EmbeddingModel()
    vector_store_manager = VectorStoreManager(embedding_model)

    # 3. Create the FAISS index
    vector_store_manager.create_index(cleaned_df)
    print("\nVector store building process complete.")
