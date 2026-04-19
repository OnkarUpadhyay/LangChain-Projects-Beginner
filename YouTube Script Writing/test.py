from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search_result = search.invoke("What is Python programming language?")
print(search_result)