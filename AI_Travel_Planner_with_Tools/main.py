import os
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import create_agents
from tasks import create_tasks

load_dotenv()

def run_planner(destination):
    # 1. Setup Agents.
    logistics_agent, weather_agent = create_agents()
    
    # 2. Setup Tasks.
    tasks = create_tasks(logistics_agent, weather_agent, destination)
    
    # 3. Initialize Crew.
    travel_crew = Crew(
        agents=[logistics_agent, weather_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    # 4. Execute.
    print(f"### Starting Travel Plan for: {destination} ###")
    result = travel_crew.kickoff()
    return result

if __name__ == "__main__":
    dest = input("Enter your destination: ")
    final_itinerary = run_planner(dest)
    print("\n\n########################")
    print("## FINAL JSON OUTPUT ##")
    print("########################\n")
    print(final_itinerary)