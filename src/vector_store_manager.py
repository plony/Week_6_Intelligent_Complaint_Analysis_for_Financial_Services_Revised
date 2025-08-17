# src/vector_store_manager.py

# src/vector_store_manager.py

import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embedding_model import EmbeddingModel
from typing import List, Dict, Any

class VectorStoreManager:
    """
    Manages the creation, loading, and querying of the FAISS vector store.
    """
    def __init__(self, embedding_model: EmbeddingModel, db_path: str = "vector_store"):
        """
        Initializes the manager with an embedding model and database path.

        Args:
            embedding_model (EmbeddingModel): An instance of the embedding model.
            db_path (str): The directory where the FAISS index is stored.
        """
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.faiss_db = None
        
    def create_index(self, df: pd.DataFrame):
        """
        Creates and saves a new FAISS index from a DataFrame of complaints.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the 'cleaned_narrative' column.
        """
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        documents = []
        metadata = []
        for _, row in df.iterrows():
            chunks = text_splitter.split_text(row['cleaned_narrative'])
            for chunk in chunks:
                documents.append(chunk)
                meta_data = {
                    'product': row['Product'],
                    'company': row['Company'],
                    'complaint_id': row['Complaint ID']
                }
                metadata.append(meta_data)
        
        self.faiss_db = FAISS.from_texts(
            texts=documents, 
            embedding=self.embedding_model.model, 
            metadatas=metadata
        )
        self.faiss_db.save_local(self.db_path, index_name="faiss_index")
        print(f"FAISS index created and saved to '{os.path.join(self.db_path, 'faiss_index.faiss')}'")
        
    def load_index(self):
        """
        Loads an existing FAISS index from disk.
        """
        try:
            # FAISS.load_local automatically looks for the .faiss and .pkl files
            # given the folder path and index name.
            self.faiss_db = FAISS.load_local(
                folder_path=self.db_path, 
                embeddings=self.embedding_model.model,
                index_name="faiss_index",
                allow_dangerous_deserialization=True
            )
            print("FAISS index loaded successfully.")
            return True
        except Exception as e:
            print(f"Error: FAISS index could not be loaded. Details: {e}")
            print("Please run 'python src/vector_store_builder.py' first.")
            return False

    def retrieve_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs a similarity search to retrieve relevant documents for a query.

        Args:
            query (str): The user's query string.
            k (int): The number of relevant documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of retrieved documents with their metadata.
        """
        if self.faiss_db is None:
            print("Error: Vector store is not loaded. Cannot perform retrieval.")
            return []
            
        docs = self.faiss_db.similarity_search(query, k=k)
        retrieved_docs = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
        return retrieved_docs