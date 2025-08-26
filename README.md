# Rag-Based-Chatbot
Developed a RAG-based chatbot integrating multiple modern AI technologies. The project utilized MongoDB as the backend for storing and retrieving chat history, ensuring persistence and context-aware conversations. For the frontend, I built an interactive and user-friendly interface using Streamlit. Tavily API key also integrated .


import os
import sys
import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# MongoDB Setup
mongo_client = MongoClient(mongodb_uri)
db = mongo_client["chatbot"]
chat_collection = db["chat_history"]

# Tavily Setup
tavily_client = TavilyClient(api_key=tavily_api_key)

# Streamlit UI
st.set_page_config(
    page_title="ChatBot by Rana Muhammad Abdullah", layout="wide")
st.title("Chatbot (ReAct Agent Enabled)")

# Sidebar - Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Load PDF and build FAISS Vectorstore


def load_vectorstore(uploaded_file):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

# Format PDF documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Tavily search


def search_tavily(query):
    search_result = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_answer=True,
        max_results=5
    )
    return search_result.get("answer", "") or "No relevant info found on web."

# Save to MongoDB


def save_chat_to_mongo(user_input, response, context, pdf_name):
    chat_collection.insert_one({
        "timestamp": datetime.utcnow(),
        "pdf_name": pdf_name,
        "user_input": user_input,
        "response": response,
        "context": context,
    })


# LLM + RAG setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=google_api_key,
    temperature=0.7,
    max_output_tokens=512
)

chat_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
)

output_parser = StrOutputParser()
rag_chain = chat_prompt | llm | output_parser

# Main App
if uploaded_file:
    vectorstore = load_vectorstore(uploaded_file)
    retriever = vectorstore.as_retriever()

    # Define tools
    def run_rag(query):
        docs = retriever.get_relevant_documents(query)
        context = format_docs(docs)
        return rag_chain.invoke({"context": context, "question": query})

    def run_tavily(query):
        return search_tavily(query)

    tools = [
        Tool(
            name="RAGTool",
            func=run_rag,
            description="Use this to answer questions about the uploaded PDF or internal documents."
        ),
        Tool(
            name="WebSearchTool",
            func=run_tavily,
            description="Use this when the question needs current or external web-based information."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Show history in sidebar
    st.sidebar.markdown("### Previous Chats")
    history = chat_collection.find(
        {"pdf_name": uploaded_file.name}).sort("timestamp", -1).limit(5)
    for chat in history:
        st.sidebar.markdown(f"**Q:** {chat['user_input']}")
        st.sidebar.markdown(f"**A:** {chat['response']}")

    # Chat input
    query = st.chat_input("Ask a question...")

    if query:
        with st.spinner("Thinking..."):
            response = agent.run(query)

            # Display
            st.markdown("**Answer:**")
            st.success(response)

            # Save history
            save_chat_to_mongo(
                query, response, "[Handled by ReAct Agent]", uploaded_file.name)

else:
    st.info("Please upload a PDF to begin chatting.")
