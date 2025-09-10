import os
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# -------------------------------
# Environment setup
# -------------------------------
DATA_PATH = "./data"
CHROMA_DIR = "./chroma_db"
RAG_FILE = "ragintro.txt"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not set. Please define it in the environment.")

# -------------------------------
# Load and split the ragintro file
# -------------------------------
def load_txt_as_docs(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_text(content)
    docs = [Document(page_content=chunk, metadata={"source": file_path}) for chunk in splits]
    return docs

# -------------------------------
# Initialize or load Chroma vectorstore
# -------------------------------
def load_chroma():
    file_path = os.path.join(DATA_PATH, RAG_FILE)
    docs = load_txt_as_docs(file_path)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
        vectorstore.persist()
    return vectorstore

# -------------------------------
# Setup vectorstore and LLM
# -------------------------------
vectorstore = load_chroma()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# -------------------------------
# Define chatbot logic
# -------------------------------
def chatbot(history, user_message):
    if history is None:
        history = []

    # Retrieve relevant context
    similar_docs = retriever.get_relevant_documents(user_message)
    
    context = "\n\n".join([doc.page_content for doc in similar_docs])
    # print(context)
    # Prepare prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a RAG knowledge chatbot. Use context and answer"),
        ("human",
         "Context:\n{context}\n\nUser Message:\n{user_message}")
    ])

    query = prompt_template.format_messages(context=context, user_message=user_message)
    response = llm.invoke(query)

    # Append user message and response as a tuple
    history.append((user_message, response.content))

    return history, history

# -------------------------------
# Create Gradio interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI Chatbot")
    chatbot_interface = gr.Chatbot(label="Chat with RAG Knowledge Bot")
    state = gr.State([])  # Holds conversation history
    user_input = gr.Textbox(label="Type your message here...", placeholder="Ask something about RAG...")

    user_input.submit(chatbot, inputs=[state, user_input], outputs=[chatbot_interface, state])

if __name__ == "__main__":
    demo.launch(share=True)
