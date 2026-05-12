from crewai import Task
from pydantic import BaseModel
from typing import List

# Define the structured output schema
class FlightInfo(BaseModel):
    airline: str
    price: str
    departure: str

class DayPlan(BaseModel):
    day: int
    forecast: str
    suggested_activities: List[str]

class Itinerary(BaseModel):
    destination: str
    total_estimated_cost: str
    flights: List[FlightInfo]
    itinerary: List[DayPlan]

def create_tasks(logistics_agent, weather_agent, destination):
    flight_task = Task(
        description=f"Search for 3 real-time flight options to {destination} for next month. Include airline and price.",
        expected_output="A list of 3 flight options with prices and airlines.",
        agent=logistics_agent
    )

    weather_task = Task(
        description=f"Get a 3-day weather forecast for {destination}. If the primary search tool fails, use the backup tool.",
        expected_output="A detailed 3-day weather summary.",
        agent=weather_agent
    )

    output_task = Task(
        description=f"Compile all data into a final JSON travel plan for {destination}.",
        expected_output="A structured JSON object following the Itinerary schema.",
        agent=logistics_agent,
        context=[flight_task, weather_task],
        output_json=Itinerary
    )
    
    return [flight_task, weather_task, output_task]