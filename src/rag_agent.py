# src/rag_agent.py

import os
import sys
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

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
        # Using a model known to be available on the free inference API
        repo_id = "tiiuae/falcon-7b-instruct"
        self.llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5)
        
        # Create the robust prompt template
        self.prompt_template = """
        You are a financial analyst assistant for CrediTrust. Your task is to
        answer questions about customer complaints. Use the following retrieved
        complaint excerpts to formulate your answer. If the context doesn't
        contain the answer, state that you don't have enough information.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        self.prompt = PromptTemplate(template=self.prompt_template,
                                     input_variables=["context", "question"])

        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.faiss_db.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt}
        )
        print("RAG agent initialized and linked to the LLM.")
        print(f"Using LLM model: {repo_id}") # Added for confirmation

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answers a user query by retrieving relevant documents and generating
        a response. Returns a dictionary containing the answer and the source
        documents.
        """
        if not self.is_loaded:
            return {"answer": "The vector store could not be loaded. Please "
                              "run the builder script first.",
                    "source_documents": []}

        # The RetrievalQA chain handles the full process
        result = self.qa_chain({"query": query})

        # Get source documents manually for display
        retrieved_docs = self.vector_store_manager.faiss_db.as_retriever().\
            get_relevant_documents(query)
        
        return {
            "answer": result["result"],
            "source_documents": retrieved_docs
        }