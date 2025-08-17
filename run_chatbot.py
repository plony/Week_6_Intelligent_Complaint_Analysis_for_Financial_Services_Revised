# run_chatbot.py


import sys
import os

# Add the project's src directory to the system path to find modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_agent import RAGChatbotAgent

def main():
    """
    Main function to run the complaint analysis chatbot.
    """
    print("Initializing the CrediTrust Complaint Analysis Chatbot...")
    
    # --- Check for the existence of the correct FAISS file ---
    # The FAISS index is split into two files: .faiss and .pkl.
    # The .faiss file contains the actual vectors, and its existence is a good check.
    index_path = os.path.join("vector_store", "faiss_index.faiss")
    
    # This check ensures we have the correct file before trying to proceed.
    # It uses os.path.join for platform-independent path handling.
    if not os.path.exists(index_path):
        print(f"Error: Vector store has not been built or the file is missing.")
        print(f"Expected to find '{os.path.abspath(index_path)}'.")
        print("Please run 'python src/vector_store_builder.py' first.")
        sys.exit(1)
    
    # If the file exists, the script will proceed
    agent = RAGChatbotAgent()
    
    # Check if the vector store loaded successfully inside the agent
    if not agent.is_loaded:
        sys.exit()

    print("\nChatbot is ready. You can ask questions about customer complaints.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Thank you for using the chatbot. Goodbye!")
            break
        
        response = agent.answer_question(user_query)
        print(response)

if __name__ == "__main__":
    main()