import os
import requests
from bs4 import BeautifulSoup
from crewai_tools import SerperDevTool
from langchain.tools import tool
from dotenv import load_dotenv
from urllib3.exceptions import InsecureRequestWarning

load_dotenv()
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Primary tool.
google_search = SerperDevTool()

# Secondary tool.
@tool("duckduckgo_search")
def duckduckgo_search(query: str):
    """
    Search the web using DuckDuckGo. 
    Use this as a fallback if the primary search tool fails.
    """
    url = "https://duckduckgo.com/html/"
    params = {'q': query}
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.post(url, data=params, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        # Use BeautifulSoup to get CLEAN text only.
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # DuckDuckGo Lite result snippets are usually in 'result__snippet' classes.
        for snippet in soup.find_all('a', class_='result__snippet')[:3]:
            results.append(snippet.get_text())

        clean_text = "\n".join(results)
        
        if not clean_text:
            # Fallback if the specific class changed.
            return soup.get_text()[:500] 
            
        return f"Found results for {query}:\n{clean_text}"
            
    except Exception as e:
        return f"Search failed: {str(e)}"