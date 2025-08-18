# src/vector_store_manager.py
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embedding_model import EmbeddingModel


class VectorStoreManager:
    """
    Manages the creation, saving, and loading of the FAISS vector store.
    """
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.faiss_db = None

    def build_and_save_index(
        self, documents: List[Document], save_dir: str
    ):
        """
        Builds a new FAISS vector index from a list of documents
        and saves it to disk.
        """
        print("Building vector index...")
        self.faiss_db = FAISS.from_documents(
            documents, self.embedding_model.model
        )
        print("Vector index built successfully.")

        print("Saving vector index to disk...")
        self.faiss_db.save_local(save_dir)
        print(f"Vector index saved to {save_dir}")

    def load_index(self, save_dir: str = "vector_store") -> bool:
        """
        Loads a pre-built FAISS vector index from disk.
        Returns True if successful, False otherwise.
        """
        if not os.path.exists(save_dir):
            print(f"Error: Vector store directory not found at {save_dir}")
            return False

        try:
            self.faiss_db = FAISS.load_local(
                save_dir, self.embedding_model.model,
                allow_dangerous_deserialization=True
            )
            print("FAISS index loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False

    def get_retriever(self, k: int = 5):
        """
        Returns a retriever instance from the loaded vector store.
        """
        if not self.faiss_db:
            print("Error: Vector store not loaded.")
            return None
        return self.faiss_db.as_retriever(search_kwargs={"k": k})
