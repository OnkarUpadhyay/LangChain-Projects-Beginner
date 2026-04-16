from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import  streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tabulate
# Load environment variables from .env file
load_dotenv()

# Set the GROQ_API_KEY environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def query_agent(data, query):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(data)
    # Initialize the ChatGroq model
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    
    # Create a pandas dataframe agent
    # Note: LangChain recently added a security feature requiring developers to explicitly opt-in to agents that execute arbitrary Python code under the hood.

    agent = create_pandas_dataframe_agent(model, df, verbose=True, allow_dangerous_code=True) 
    
    # Get the response from the agent
    response = agent.invoke(query)
    
    return response

# ----- Streamlit app -----
st.title("CSV Data Analysis Tool")
st.header("Please upload your CSV file here:")

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

query = st.text_area("Enter your query for the CSV data:")
if st.button("Submit Query"):
    if uploaded_file is not None and query:
        with st.spinner("Processing your query..."):
            result = query_agent(uploaded_file, query)
            st.subheader("Response:")
            st.write(result)
    else:
        st.warning("Please upload a CSV file and enter a query before submitting.")
