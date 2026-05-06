import streamlit as st
from utils import get_website_data, split_data, create_embeddings, push_to_faiss, get_conversational_chain
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Modern Page Configuration
st.set_page_config(page_title="Site-Bot AI", page_icon="🌐", layout="wide")

# Custom CSS for Modern UI/UX
st.markdown("""
<style>
    /* Main background and font styling */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up the top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        border: 1px solid #f0f0f0;
    }

    /* Primary buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #4F46E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'Groq_API_Key' not in st.session_state:
    st.session_state['Groq_API_Key'] = ''

# -------- SIDEBAR CONFIGURATION --------
with st.sidebar:
    st.title("🌐 Site-Bot Setup")
    st.markdown("Configure your AI agent here.")
    st.divider()
    
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    st.session_state['Groq_API_Key'] = groq_api_key if groq_api_key else os.getenv("GROQ_API_KEY")

    st.markdown("### 🔗 Target Website")
    target_url = st.text_input("Enter Webpage or Sitemap URL", placeholder="https://example.com/sitemap.xml")
    
    load_button = st.button("Ingest Website Data", key="load_button")

    if load_button:
        if st.session_state['Groq_API_Key'] == "":
            st.error("Please provide your Groq API Key first.")
        elif target_url == "":
            st.error("Please provide a target URL.")
        else:
            with st.status("Building Knowledge Base...", expanded=True) as status:
                st.write("📥 Fetching website data...")
                site_data = get_website_data(target_url)
                
                st.write("✂️ Splitting data into chunks...")
                chunks_data = split_data(site_data)
                
                st.write("🧠 Generating local embeddings...")
                embeddings = create_embeddings()
                
                st.write("💾 Saving to local FAISS database...")
                # UPDATE THIS LINE:
                st.session_state.vector_store = push_to_faiss(chunks_data, embeddings)
                
                status.update(label="Knowledge Base Ready!", state="complete", expanded=False)
            st.success("App is ready for chat!")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -------- MAIN CHAT INTERFACE --------
st.title('💬 Support Chatbot')
st.caption("Ask me anything about the ingested website. I'll provide answers and links.")

# Render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    
    # Prevent chat if DB isn't ready
    if st.session_state.vector_store is None or st.session_state['Groq_API_Key'] == "":
        st.warning("⚠️ Please configure the API Key and Ingest Website Data in the sidebar first.")
    else:
        # 1. Add user message to UI
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching website..."):
                # Get the RAG chain
                chain = get_conversational_chain(st.session_state.vector_store, st.session_state['Groq_API_Key'])
                
                # Execute retrieval and generation
                response = chain.invoke({"input": prompt})
                answer = response["answer"]
                source_documents = response["context"]

                # Format the sources as clickable links or distinct text blocks
                source_links = set()
                for doc in source_documents:
                    if 'source' in doc.metadata:
                        source_links.add(doc.metadata['source'])
                
                # Combine answer with sources
                full_response = answer
                if source_links:
                    full_response += "\n\n**🔗 Learn more:**\n"
                    for link in source_links:
                        full_response += f"- [{link}]({link})\n"
                
                # Display to UI
                st.markdown(full_response)
        
        # 3. Save assistant message to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})