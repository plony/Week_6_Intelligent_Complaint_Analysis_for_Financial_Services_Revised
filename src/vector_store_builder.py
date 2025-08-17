# src/vector_store_builder.py
import pandas as pd
import os
import sys
from tqdm import tqdm

# Correct import paths for LangChain modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Add the project root to the system path to allow for absolute imports
# This is a robust way to handle imports in project-based scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom modules
from src.data_loader import load_complaints_data
from src.preprocessing import filter_and_clean_data

def build_vector_store(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Orchestrates the data loading, preprocessing, chunking, embedding, and indexing process.
    """
    # Load the raw data using the data_loader module
    df = load_complaints_data(file_path)
    if df is None:
        return
        
    # Preprocess the data using the preprocessing module
    cleaned_df = filter_and_clean_data(df)
    if cleaned_df is None:
        return
    
    # Save the cleaned dataframe for future use and as a deliverable
    output_path = 'data/filtered_complaints.csv'
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned and filtered dataset saved to {output_path}")

    # --- 1. Text Chunking ---
    print(f"Creating chunks with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    metadata = []
    
    for index, row in tqdm(cleaned_df.iterrows(), total=cleaned_df.shape[0], desc="Creating documents from narratives"):
        doc_content = row['cleaned_narrative']
        meta_data = {
            'product': row['Product'],
            'company': row['Company'],
            'date': row['Date received'],
            'complaint_id': row['Complaint ID']
        }
        
        # Split the document into chunks
        chunks = text_splitter.split_text(doc_content)
        
        # Add each chunk with its metadata
        for chunk in chunks:
            documents.append(chunk)
            metadata.append(meta_data)
            
    # --- 2. Embedding Model Selection ---
    print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- 3. Embedding and Indexing ---
    print("Creating and saving the FAISS vector store...")
    db = FAISS.from_texts(texts=documents, embedding=embedding_model, metadatas=metadata)

    # --- 4. Save the vector store ---
    save_path = "vector_store"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    db.save_local(save_path, index_name="faiss_index")
    print(f"Vector store saved to {save_path}/faiss_index.bin")
    
if __name__ == "__main__":
    raw_data_path = 'data/complaints.csv'
    build_vector_store(raw_data_path)