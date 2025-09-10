import os
from dotenv import load_dotenv
import gradio as gr

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_google_genai import GoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Google API
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

# Load and parse documents
file_path = '/workspace/Work/UI_Demo_Gradio/data/ragintro.txt'
if not os.path.exists(file_path):
    raise ValueError(f"File {file_path} does not exist. Please check the path.")

reader = SimpleDirectoryReader(input_files=[file_path])
documents = reader.load_data()

parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)
all_documents = parser.get_nodes_from_documents(documents)

# Use Hugging Face embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector index
index = VectorStoreIndex.from_documents(documents=all_documents, embed_model=embed_model)
print(f"Vectors Store ID: {index.index_id}")

# Create query engine from index
query_engine = index.as_query_engine(similarity_top_k=3)

# Function to handle user queries
def chatbot(user_input, history=[]):
    # Retrieve relevant nodes using the query engine
    response = query_engine.query(user_input)
    
    # The result's response text
    answer = response.response
    
    # Update history
    history.append((user_input, answer))
    return history, history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📘 RAG Chatbot using Google API")
    chatbot_component = gr.Chatbot()
    message = gr.Textbox(placeholder="Type your message here...")
    state = gr.State([])

    message.submit(chatbot, inputs=[message, state], outputs=[chatbot_component, state])
    message.submit(lambda: "", inputs=[], outputs=[message])  # Clear input after submit

if __name__ == "__main__":
    demo.launch(share=True)
