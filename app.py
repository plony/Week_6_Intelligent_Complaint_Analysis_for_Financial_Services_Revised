# app.py

import gradio as gr
from src.rag_agent import RAGChatbotAgent

# Initialize the RAG agent
agent = RAGChatbotAgent()


def chatbot_response(query):
    """
    Handles the user query, retrieves an answer from the RAG agent,
    and formats the output.
    """
    response = agent.answer_question(query)

    answer = response["answer"]
    source_docs = response["source_documents"]

    source_text = ""
    if source_docs:
        source_text = "### Sources\n"
        for i, doc in enumerate(source_docs):
            source_text += f"**Source {i+1}:** {doc.page_content}\n"
    
    return answer, source_text


# Create the Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analysis Chatbot") as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Analysis Chatbot
        Ask me questions about customer complaints and I'll provide an answer
        based on the provided data. The chatbot is powered by a RAG
        (Retrieval-Augmented Generation) pipeline.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What are the most common complaints "
                            "about student loans?",
                lines=5
            )
            submit_button = gr.Button("Submit", variant="primary")
            clear_button = gr.Button("Clear")

        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="AI-Generated Answer",
                interactive=False,
                lines=10
            )

    sources_output = gr.Markdown(label="Sources")

    submit_button.click(
        fn=chatbot_response,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )
    clear_button.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch()