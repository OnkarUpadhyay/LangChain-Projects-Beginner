import streamlit as st
import PyPDF2
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import List, Optional

# -- Load environment variables from .env file (if exists) ---
load_dotenv()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Page Configuration ---
st.set_page_config(page_title="AI Contextual Quiz Engine", page_icon="📚", layout="wide")

# --- Defining the structured output format options ---
class Option(BaseModel):
    label: str = Field(description="The lowercase letter for the option: 'a', 'b', 'c', or 'd'")
    text: str = Field(description="The text of the multiple choice option")

class QuestionItem(BaseModel):
    question: str = Field(description="The actual question based on the context")
    options: Optional[List[Option]] = Field(default=None, description="Exactly 4 options if MCQ. Null if subjective.")
    answer: str = Field(description="The correct answer (e.g., 'c. lord' for MCQ, or full text for Subjective)")

class Quiz(BaseModel):
    questions: List[QuestionItem] = Field(description="The list of generated questions")


# --- Initialize Free Local Embedding Model (Runs without an API key) ---
# Using a standard embedding model that works well for English academic text.
@st.cache_resource 
def load_embeddings():
    with st.spinner("Loading local embedding model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # THIS IS THE MAGIC LINE: Force it to never ping the internet
            model_kwargs={'local_files_only': True} 
        )
    return embeddings

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def process_document_into_vector_store(text, embeddings):
    """Implementation of my design flow: TEXT -> CHUNKS -> EMBEDDINGS -> VECTOR STORE"""

    # 1. READ/CHUNK (TEXT CHUNKS)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 300,
        length_function = len,
    )

    # The output document is structured list of 'TEXT CHUNK 1', 'TEXT CHUNK 2', etc.
    documents = text_splitter.create_documents([text])

    # 2. EMBED & STORE
    # FAISS takes the Text Chunks, passes them through the Embeddings model, 
    # and creates an efficient, searchable database in memory.
    with st.spinner("Building Vector Store from document chunks.."):
        vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def generate_pdf_document(text_content):
    """Generates a downloadable PDF file from the generated text"""
    # Ensure raw byte content is correct
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 12)

        # We need to clean the text of special characters that FPDF cannot handle.
        #cleaned_text = text_content.replace(u'\u201c', '"').replace(u'\u201d', '"').replace(u'\u2018', "'").replace(u'\u2019', "'")
        cleaned_text = text_content

        # Split text into lines to ensure proper multi-cell wrapping
        lines = cleaned_text.split('\n')
        for line in lines:
            pdf.multi_cell(0, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'))
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"Error creating PDF download: {e}")
        return b""

def create_prompt_template():
    """Builds the prompt based on user configurations."""
    template = """
    You are an expert academic evaluator. Using ONLY the provided context retrieved from a book/document, 
    generate a set of questions and answers based on the user's precise preferences.
    
    Strict Rule: If the context does not contain enough direct information to generate an accurate question that meets all criteria, state that clearly for that item.
    
    Context (Retrieved via Semantic Search):
    {context}
    
    Configuration:
    - Difficulty Level: {difficulty}
    - Question Type: {q_type}
    - Output Format: {layout}
    
    Instructions:
    1. Base every question on specific information in the provided context.
    2. If Difficulty is "Advanced," prioritize analytical or critical thinking questions.
    3. If Output Format is "Questions and Answers Together," output the question immediately followed by its answer.
    4. If Output Format is "Questions and Answers Separated," list ALL questions first, followed by a distinct "Answers" section.
    5. CRITICAL MCQ RULE: For Single-Correct MCQs, there must be EXACTLY ONE correct option. The other three options MUST BE COMPLETELY FALSE based on the context. Do NOT use other true facts from the text as incorrect options.
    
    Output Requirements:
    - You must map your generated questions strictly to the required schema.

    Generate the content now:
    """
    return PromptTemplate(
        input_variables=["context", "difficulty", "q_type", "layout"],
        template=template
    )


def format_quiz_to_markdown(quiz_obj: Quiz, layout: str) -> str:
    """Takes the structured Pydantic object and formats it using standard Python."""
    final_text = ""
    
    if layout == "Questions & Answers Together":
        for i, q in enumerate(quiz_obj.questions, 1):
            final_text += f"**Q{i}: {q.question}**\n"
            if q.options:
                for opt in q.options:
                    final_text += f"{opt.label}. {opt.text}\n"
            final_text += f"\n*Answer: {q.answer}*\n\n---\n\n"
            
    elif layout == "Questions & Answers Separated":
        final_text += "### Questions\n\n"
        for i, q in enumerate(quiz_obj.questions, 1):
            final_text += f"**Q{i}: {q.question}**\n"
            if q.options:
                for opt in q.options:
                    final_text += f"{opt.label}. {opt.text}\n"
            final_text += "\n"
            
        final_text += "---\n### Answers\n\n"
        for i, q in enumerate(quiz_obj.questions, 1):
             final_text += f"**A{i}:** {q.answer}\n\n"
             
    return final_text

# --- App UI and Logic ---
def main():
    st.title("📚 AI Contextual Quiz Engine")
    st.caption("A production-level MCQ & Subjective Question Creator. Upload your context and specify requirements.")

    # Load Embeddings Model
    embeddings = load_embeddings()

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key_input = st.text_input("Enter Groq API Key (Optional)", type="password", 
                                      help="Leave blank to use the system default GROQ_API_KEY environment variable.")
        api_key = api_key_input if api_key_input else os.getenv("GROQ_API_KEY")

        uploaded_file = st.file_uploader("Upload Book/Document (PDF)", type=["pdf"])
        if uploaded_file and not st.session_state.document_loaded:
            with st.spinner("Extracting text and running RAG pipeline..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    st.session_state.vector_store = process_document_into_vector_store(extracted_text, embeddings)
                    st.session_state.document_loaded = True
                    st.success("Document RAG database built successfully!")
                else:
                    st.error("Could not extract meaningful text from this PDF.")

        if st.session_state.document_loaded:
            if st.button("Clear loaded document"):
                st.session_state.vector_store = None
                st.session_state.document_loaded = False
                st.rerun()

        st.divider()

        difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
        q_type = st.selectbox("Question Type", ["Single-Correct MCQ", "Multi-Correct MCQ", "Subjective"])
        layout = st.radio("Output Layout", ["Questions & Answers Together", "Questions & Answers Separated"])
        
        generate_btn = st.button("Generate Assessment", use_container_width=True)

    # --- Main Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle Generation Logic
    if generate_btn:
        if not st.session_state.document_loaded:
            st.error("Please upload a PDF document and wait for the RAG pipeline to build the database.")
        elif not api_key:
            st.error("Please provide an API key or ensure the system default GROQ_API_KEY is set in your environment.")
        else:
            user_meta_query = f"Create a {difficulty.lower()} level assessment with {q_type} questions, outputting formatted as '{layout}'."
            st.session_state.messages.append({"role": "user", "content": user_meta_query})
            with st.chat_message("user"):
                st.markdown(user_meta_query)

            # LLM Execution
            with st.chat_message("assistant"):
                with st.spinner("Performing Semantic Search & generating questions..."):
                    try:
                        # 1. Semantic Search
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                        relevant_documents = retriever.invoke(user_meta_query)
                        structured_context = "\n\n".join([doc.page_content for doc in relevant_documents])
                        
                        # 2. Setup LLM with Pydantic Structured Output
                        llm = ChatGroq(temperature=0.0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
                        structured_llm = llm.with_structured_output(Quiz)
                        
                        # 3. Chain Setup
                        prompt_tmpl = create_prompt_template()
                        chain = prompt_tmpl | structured_llm
                        
                        # 4. Invoke Chain (Returns Pydantic Quiz Object)
                        quiz_object = chain.invoke({
                            "context": structured_context,
                            "difficulty": difficulty,
                            "q_type": q_type,
                            "layout": layout
                        })
                        
                        # 5. Format to Markdown and Display
                        result_text = format_quiz_to_markdown(quiz_object, layout)
                        st.markdown(result_text)
                        
                        st.session_state.messages.append({"role": "assistant", "content": result_text})
                        
                        # 6. PDF Generation
                        pdf_bytes = generate_pdf_document(result_text)
                        if pdf_bytes:
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes,
                                file_name=f"{q_type}_{difficulty}_assessment.pdf",
                                mime="application/pdf"
                            )
                        
                    except Exception as e:
                        st.error(f"An error occurred during semantic retrieval or generation: {e}")

if __name__ == "__main__":
    main()