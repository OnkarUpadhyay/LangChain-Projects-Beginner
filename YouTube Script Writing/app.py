import os
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import base64
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="YouTube Script Writing Tool", page_icon="✍🏼", layout="wide")

# Applying Styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #b1f288;
    color:#FFFFFF;
    }
</style>""", unsafe_allow_html=True)


def get_local_gif(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    return f"data:image/gif;base64,{encoded}"

gif_source = get_local_gif('./cat-pixel.gif')

html_code = f"""
<div style="display: flex; align-items: center; gap: 15px;">
    <img src="{gif_source}" style="width: 90px;">
    <h1 style="margin: 0; padding: 0;">YouTube Script Writing Tool</h1>
</div>
<hr style="margin-top: 10px;">
"""

st.markdown(html_code, unsafe_allow_html=True)

# Creating Session State Variable
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] =''

# Function to generate video script
def generate_script(prompt,video_length,creativity):
    
    # Template for generating 'Title'
    title_template = PromptTemplate(
        input_variables = ['subject'], 
        template='Please come up with a title for a YouTube video on the {subject}.'
        )

    # Template for generating 'Video Script' using search engine
    script_template = PromptTemplate(
        input_variables = ['title', 'DuckDuckGo_Search','duration'], 
        template='Create a script for a YouTube video based on this title for me. TITLE: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search} '
    )

    #Setting up OpenAI LLM
    llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature=creativity
    ) 
    
    #Creating chain for 'Title' & 'Video Script'
    title_chain = title_template | llm | StrOutputParser()  
    script_chain = script_template | llm | StrOutputParser()

    search = DuckDuckGoSearchRun()

    # Executing the chains we created for 'Title'
    title = title_chain.invoke({"subject": prompt})

    # Executing the chains created for 'Video Script' by the help of search engine 'DuckDuckGo'
    search_result = search.invoke(prompt) 
    script = script_chain.invoke({
        "title": title, 
        "DuckDuckGo_Search": search_result, 
        "duration": video_length
    })

    # Returning the output
    return search_result,title,script

# Sidebar to capture the API key
with st.sidebar:
    st.title("🔑 API Key")
    api_key_input = st.text_input("Enter your API Key:", type="password", help = "Get your API key from https://groq.com/ and paste it here to use the tool.")
    st.session_state['API_Key'] = api_key_input if api_key_input else os.getenv("GROQ_API_KEY")

    st.image('./youtube.png', width=200, use_container_width=True)

# Captures User Inputs
prompt = st.text_input('Please provide the topic of the video',key="prompt")
video_length = st.text_input('Expected Video Length 🕒 (in minutes)',key="video_length")  
creativity = st.slider('Creativity limit ✨ - (0 LOW || 1 HIGH)', 0.0, 1.0, 0.2,step=0.1)

submit = st.button("Generate Script for me")

if submit:
    
    if st.session_state['API_Key']:
        with st.spinner("🪄 Generating your script..."):
            search_result,title,script = generate_script(prompt,video_length,creativity)
        #Let's generate the script
        st.success('Hope you like this script ❤️')

        #Display Title
        st.subheader("Title:🔥")
        st.write(title)

        #Display Video Script
        st.subheader("Your Video Script:📝")
        st.write(script)

        #Display Search Engine Result
        st.subheader("Check Out - DuckDuckGo Search:🔍")
        with st.expander('Show me 👀'): 
            st.info(search_result)
    else:
        st.error("Ooopssss!!! Please provide API key.....")