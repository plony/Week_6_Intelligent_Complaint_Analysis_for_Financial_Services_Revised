# test_connection.py

import os
from huggingface_hub import InferenceClient

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not HUGGINGFACEHUB_API_TOKEN:
    print("HUGGINGFACEHUB_API_TOKEN is not set.")
    exit()

# Create a client instance with the API token and a different model
client = InferenceClient(
    model="google/flan-t5-large", # Changed model
    token=HUGGINGFACEHUB_API_TOKEN
)

try:
    print("Attempting to generate a response...")
    response = client.text_generation(
        "What is a financial complaint?",
        max_new_tokens=50
    )
    print("Connection successful!")
    print("Generated Response:")
    print(response)
except Exception as e:
    print(f"Connection failed. Error: {e}")