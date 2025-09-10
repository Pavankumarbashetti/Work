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

# Google API Key
# google_api_key = os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("Google API key not found in environment variables.")

# Initialize the Google Generative AI model
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

# Load and parse documents
reader = SimpleDirectoryReader(input_files=['/workspace/Work/UI_Demo_Gradio/data/ragintro.txt'])
documents = reader.load_data()

parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)
all_documents = parser.get_nodes_from_documents(documents)


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Create the vector index
index = VectorStoreIndex.from_documents(documents=all_documents,embed_model=embed_model)
print(f"Vectors Store ID: {index.index_id}")

# Set up the chat engine
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    llm=llm,
    system_prompt=(
        "You are a chatbot that can answer questions based on the contents of a provided PDF document. "
        "If the question is unrelated or the answer is not found in the document, reply with 'I don't know.'"
    ),
)

# Gradio chatbot function
def chatbot(user_input, history=[]):
    response = chat_engine.chat(user_input)
    history.append((user_input, response))
    return history, history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 📘 RAG - Chatbot")
    chatbot_component = gr.Chatbot()
    message = gr.Textbox(placeholder="Type your message here...")
    state = gr.State([])

    message.submit(chatbot, inputs=[message, state], outputs=[chatbot_component, state])
    message.submit(lambda: "", inputs=[], outputs=[message])  # Clear input after submit

if __name__ == "__main__":
    demo.launch(share=True)
