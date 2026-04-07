#import os
import streamlit as st
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
#from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- SESSION STATE INITIALIZATION ---
if 'history_store' not in st.session_state:
    st.session_state['history_store'] = {} # Stores chat history per session
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''

st.set_page_config(page_title="Chat GPT Clone", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h1>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("😎 Setting")
st.session_state['API_Key'] = st.sidebar.text_input("What's your API key?", type="password")

# Helper function to get history for the current session
def get_session_history(session_id: str):
    if session_id not in st.session_state['history_store']:
        st.session_state['history_store'][session_id] = InMemoryChatMessageHistory()
    return st.session_state['history_store'][session_id]

# Summarize button logic
if st.sidebar.button("Summarise the conversation"):
    history = get_session_history("default_session")
    if history.messages:
        # Simple summary: list all previous exchanges
        summary_text = "\n".join([f"{type(m).__name__}: {m.content}" for m in history.messages])
        st.sidebar.write(f"Nice chatting with you! ❤️\n\nFull History:\n{summary_text}")
    else:
        st.sidebar.write("No conversation to summarize yet!")

# --- CORE LOGIC ---
def get_response(user_input, api_key):
    # 1. Setup the Model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.5, 
        groq_api_key=api_key
    )

    # 2. Setup the Prompt 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 3. Create the Chain using LCEL
    chain = prompt | llm

    # 4. Wrap with History Management
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 5. Invoke the chain
    config = {"configurable": {"session_id": "default_session"}}
    response = with_history.invoke({"input": user_input}, config=config)
    
    return response.content

# --- UI LAYOUT ---
response_container = st.container()
input_container = st.container()

with input_container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button and st.session_state['API_Key']:
            # Get Response
            model_response = get_response(user_input, st.session_state['API_Key'])
            
            # Store for UI display
            st.session_state['messages'].append({"role": "user", "content": user_input})
            st.session_state['messages'].append({"role": "ai", "content": model_response})

            # Display Chat
            with response_container:
                for i, msg in enumerate(st.session_state['messages']):
                    if msg["role"] == "user":
                        message(msg["content"], is_user=True, key=f"{i}_user")
                    else:
                        message(msg["content"], key=f"{i}_ai")
        elif submit_button and not st.session_state['API_Key']:
            st.error("Please enter your API Key in the sidebar.")