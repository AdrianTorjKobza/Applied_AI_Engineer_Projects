from crewai import Agent
from langchain_community.llms import Ollama
from tools import google_search, duckduckgo_search

# Initialize Local LLM via Ollama.
# Ensure you have run: ollama pull llama3.
ollama_llama3 = Ollama(model="llama3")

def create_agents():
    logistics_expert = Agent(
        role='Logistics Expert',
        goal='Find the most cost-effective and convenient travel options.',
        backstory='You are a master of travel hacking and logistics, specialized in finding real-time data.',
        tools=[google_search],
        llm=ollama_llama3,
        verbose=True
    )

    weather_specialist = Agent(
        role='Weather Specialist',
        goal='Provide weather forecasts.',
        backstory="""You are highly resilient. If your primary Google search tool returns 
        an error or is blocked, you immediately switch to DuckDuckGo to ensure 
        no interruption in service.""",
        tools=[google_search, duckduckgo_search], # Multiple tools for resilience. Duck Duck Go is the fallback plan.
        llm=ollama_llama3,
        max_iter=5,
        verbose=True
    )
    
    return logistics_expert, weather_specialist