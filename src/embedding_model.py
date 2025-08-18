# src/embedding_model.py

from langchain_community.embeddings import SentenceTransformerEmbeddings


class EmbeddingModel:
    """
    A class to manage the embedding model for the RAG pipeline.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.

        Args:
            model_name (str): The name of the Sentence-Transformer model to use.
        """
        self.model = SentenceTransformerEmbeddings(model_name=model_name)
        print(f"Embedding model '{model_name}' loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of text documents.

        Args:
            texts (list[str]): A list of text strings.

        Returns:
            list[list[float]]: A list of corresponding vector embeddings.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Generates an embedding for a single query string.

        Args:
            query (str): The query string.

        Returns:
            list[float]: The vector embedding for the query.
        """
        return self.model.embed_query(query)