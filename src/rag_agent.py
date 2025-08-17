# src/rag_agent.py

import sys
import os
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI # Placeholder for a real LLM

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
        
        # Load the pre-built FAISS index
        self.is_loaded = self.vector_store_manager.load_index()
        
        # NOTE: This is a placeholder for a real LLM. 
        # For a real application, you would load a model here (e.g., from OpenAI, Hugging Face, etc.)
        # self.llm = OpenAI(api_key="YOUR_API_KEY") 
        print("RAG agent initialized. LLM placeholder ready.")

    def answer_question(self, query: str) -> str:
        """
        Answers a user query by retrieving relevant documents and generating a response.

        Args:
            query (str): The user's question.

        Returns:
            str: The synthesized answer from the RAG agent.
        """
        if not self.is_loaded:
            return "The vector store could not be loaded. Please run the builder script first."

        # 1. Retrieval: Find relevant documents
        print(f"\n--- Retrieving documents for query: '{query}' ---")
        retrieved_docs = self.vector_store_manager.retrieve_documents(query)

        if not retrieved_docs:
            return "No relevant complaints found for your query."

        # 2. Augmentation & Generation: Format documents and feed to a mock LLM
        print("\n--- Retrieved Documents (Context) ---")
        context_text = ""
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1} (Product: {doc['metadata']['product']}):")
            print(f"  {doc['text'][:150]}...") # Print a snippet
            context_text += f"Document {i+1}:\n{doc['text']}\n\n"

        # This is where a real LLM would generate the answer.
        # For this example, we'll just return a synthesized response.
        
        prompt_template = PromptTemplate(
            template="You are a helpful assistant for a financial services company. "
                     "Synthesize the following customer complaints to answer the question: {question}\n\n"
                     "Complaints:\n{context}\n\n"
                     "Synthesized Answer:",
            input_variables=["question", "context"]
        )

        formatted_prompt = prompt_template.format(question=query, context=context_text)
        
        # Simulating LLM response
        simulated_response = (
            "Based on the complaints provided, a major issue is customers reporting"
            " unexpected fees and difficulty with billing disputes."
        )

        return f"\n--- Synthesized Answer (Simulated LLM) ---\n{simulated_response}\n\n" \
               f"Original Contexts Retrieved:\n{context_text}"