import os
from langchain_community.document_loaders import SitemapLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import asyncio

# Function to fetch data from website (handles both Sitemaps and standard URLs)
def get_website_data(url):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        pass 
    else:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

    if url.lower().endswith('.xml'):
        loader = SitemapLoader(url)
    else:
        loader = WebBaseLoader(url)
    
    docs = loader.load()
    return docs

# Function to split data into smaller chunks
def split_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

# Function to create local embeddings instance
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# Function to push data to local FAISS index
def push_to_faiss(chunks_data, embeddings, persist_dir="./faiss_index"):
    # Generate the FAISS vector store
    vector_store = FAISS.from_documents(chunks_data, embeddings)
    # Save it locally to persist the data
    vector_store.save_local(persist_dir)
    return vector_store

# Function to build the pure LCEL Conversational Chain
def get_conversational_chain(vector_store, groq_api_key):
    llm = ChatGroq( 
        model_name="llama-3.3-70b-versatile",
        api_key=groq_api_key, 
        temperature=0.3
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    system_prompt = (
        "You are an expert, friendly customer support assistant for a website. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, say that you cannot find the answer on the website. "
        "Always be polite, concise, and professional.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The Pure LCEL RAG Pipeline
    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever
        )
        | RunnablePassthrough.assign(
            answer=(
                {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]}
                | prompt
                | llm
                | StrOutputParser()
            )
        )
    )
    
    return rag_chain