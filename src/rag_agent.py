# src/rag_agent.py

# src/rag_agent.py

import os
import sys
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding_model import EmbeddingModel
from src.vector_store_manager import VectorStoreManager

class RAGChatbotAgent:
    """
    A Retrieval-Augmented Generation (RAG) agent for answering questions
    based on a knowledge base of customer complaints.
    """
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store_manager = VectorStoreManager(self.embedding_model)
        
        self.is_loaded = self.vector_store_manager.load_index()

        # Initialize the LLM generator from Hugging Face Hub
        # You need to set your Hugging Face API key as an environment variable
        # `os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."` or
        # in your terminal: `set HUGGINGFACEHUB_API_TOKEN=hf_...` (Windows)
        # or `export HUGGINGFACEHUB_API_TOKEN=hf_...` (macOS/Linux)
        repo_id = "google/flan-t5-xxl" # A powerful and general-purpose model
        self.llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5, max_length=512)

        # Create the robust prompt template
        self.prompt_template = """
        You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
        Use the following retrieved complaint excerpts to formulate your answer.
        If the context doesn't contain the answer, state that you don't have enough information.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])

        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.faiss_db.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt}
        )
        print("RAG agent initialized and linked to the LLM.")

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answers a user query by retrieving relevant documents and generating a response.
        Returns a dictionary containing the answer and the source documents.
        """
        if not self.is_loaded:
            return {"answer": "The vector store could not be loaded. Please run the builder script first.", "source_documents": []}

        # The RetrievalQA chain handles the full process of retrieval, context stuffing, and generation.
        result = self.qa_chain({"query": query})
        
        # We need to get the source documents manually for display
        retrieved_docs = self.vector_store_manager.faiss_db.as_retriever().get_relevant_documents(query)
        
        return {
            "answer": result["result"],
            "source_documents": retrieved_docs
        }