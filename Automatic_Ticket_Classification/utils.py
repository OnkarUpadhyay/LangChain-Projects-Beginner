import streamlit as st
import os
import PyPDF2
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

# ==========================================
# 1. CACHED RESOURCES (MASSIVE SPEED BOOST)
# ==========================================
@st.cache_resource
def get_embeddings():
    """Loads the embedding model only once into memory."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'local_files_only': True})

@st.cache_resource
def get_faiss_index(_embeddings, persist_dir="./faiss_index"):
    """Loads the FAISS index only once into memory."""
    try:
        return FAISS.load_local(persist_dir, _embeddings, allow_dangerous_deserialization=True)
    except:
        return None

@st.cache_resource
def get_svm_model(model_path='modelsvm.pk1'):
    """Loads the trained SVM model only once into memory."""
    try:
        return joblib.load(model_path)
    except:
        return None

# ==========================================
# 2. USER & RAG FUNCTIONS
# ==========================================
def predict_department(user_query, embeddings, svm_model):
    """Converts user query to embedding and predicts department."""
    query_embedding = embeddings.embed_query(user_query)
    result = svm_model.predict([query_embedding])
    return result[0]

def get_similar_docs(index, query, k=3):
    return index.similarity_search(query, k=k)

def get_answer(docs, user_input):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    
    context = "\n".join([doc.page_content for doc in docs])
    system_message = SystemMessage(
        content=f"You are a helpful corporate assistant. Use the following policy context to answer the user's question accurately.\n\nContext:\n{context}"
    )
    human_message = HumanMessage(content=user_input)
    
    response = llm.invoke([system_message, human_message])
    return response.content

# ==========================================
# 3. ADMIN & DATA PROCESSING FUNCTIONS
# ==========================================
def read_pdf_data(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50) 
    return text_splitter.create_documents(text_splitter.split_text(text))

def push_to_faiss(chunks_data, embeddings, persist_dir="./faiss_index"):
    vector_store = FAISS.from_documents(chunks_data, embeddings)
    vector_store.save_local(persist_dir)
    return vector_store