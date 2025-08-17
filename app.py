# app.py

import gradio as gr
import os
import sys

# Add the project's src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_agent import RAGChatbotAgent

# --- Global Initialization (done once) ---
agent = RAGChatbotAgent()

# --- Core Chatbot Logic for Gradio Interface ---
def chatbot_response(question):
    """
    Generates a response to a user question using the RAG agent.
    
    Args:
        question (str): The user's input question.
        
    Returns:
        tuple: A tuple containing the AI-generated answer and formatted sources.
    """
    if not question:
        return "Please enter a question to get a response.", ""

    response = agent.answer_question(question)
    generated_answer = response["answer"]
    
    # Format the source documents for display
    source_markdown = ""
    if response["source_documents"]:
        source_markdown += "### Sources:\n"
        for i, doc in enumerate(response["source_documents"]):
            source_markdown += f"- **Source {i+1} (Product: {doc.metadata.get('product', 'N/A')}, Company: {doc.metadata.get('company', 'N/A')}):**\n"
            source_markdown += f"  > {doc.page_content}\n\n"
    
    return generated_answer, source_markdown

# --- Gradio Interface ---
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, label="Your Question"),
    outputs=[
        gr.Textbox(label="AI-Generated Answer", lines=5),
        gr.Markdown(label="Sources Used")
    ],
    title="CrediTrust Complaint Analysis Chatbot",
    description="Ask me questions about customer complaints and I'll provide an answer based on the provided data. "
                "The chatbot is powered by a RAG (Retrieval-Augmented Generation) pipeline.",
    live=False,
    flagging_mode="never"
)

# Launch the Gradio app
if __name__ == "__main__":
    # Ensure the vector store is built before running the app
    if not agent.is_loaded:
        print("Error: Vector store is not loaded. Please run 'python src/vector_store_builder.py' first.")
    else:
        iface.launch()