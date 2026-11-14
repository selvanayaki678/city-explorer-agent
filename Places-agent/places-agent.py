"""
Places Agent - finds REAL places (via Foursquare Places API) in a given
area, based on a traveler's interests. Geocodes the area first (Nominatim,
free, no key), then searches Foursquare around those coordinates.

This is the first agent in the City Explorer multi-agent A2A project.

Environment variables required:
    FSQ_API_KEY - your Foursquare Places API key

Local test:
    python places_agent.py

AgentCore deploy:
    agentcore configure -e places_agent.py
    agentcore launch
    agentcore invoke '{"area": "Jayanagar, Bengaluru", "interests": "cafe, temple"}'
"""

import os
import requests
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp

FSQ_API_KEY = os.environ.get("FSQ_API_KEY")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FSQ_SEARCH_URL = "https://places-api.foursquare.com/places/search"


# ---------------------------------------------------------------------------
# Tool: find_places
# Wraps the two-step geocode -> Foursquare search flow already validated
# in test_geocode_then_search.py, exposed as a Strands tool the agent's
# LLM can call.
# ---------------------------------------------------------------------------
@tool
def find_places(area: str, interest: str = "") -> str:
    """Find real, current places near a given area using live location data.

    Use this tool whenever the user asks about places to visit, eat, or do
    something in a specific city, neighborhood, or address. Do NOT answer
    from memory - always call this tool to get accurate, current results.

    Args:
        area: A city, neighborhood, or address (e.g. "Jayanagar, Bengaluru",
              "Koramangala, Bengaluru", "Electronic City, Bengaluru").
        interest: What kind of place to look for (e.g. "cafe", "temple",
                  "museum", "park", "restaurant", "biryani"). Leave empty
                  to get top nearby places of any type.

    Returns:
        A string listing up to 5 matching places with name, category,
        address, and distance from the area center. Returns an error
        message if the area could not be found or the API call failed.
    """
    if not FSQ_API_KEY:
        return "Error: FSQ_API_KEY is not set. Cannot search for places."

    # --- Step 1: geocode the area to lat/lng ---
    try:
        geo_resp = requests.get(
            NOMINATIM_URL,
            params={"q": area, "format": "json", "limit": 1},
            headers={"User-Agent": "city-explorer-agent/1.0"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_results = geo_resp.json()
    except requests.RequestException as e:
        return f"Error geocoding '{area}': {e}"

    if not geo_results:
        return f"Could not find a location matching '{area}'. Try a more specific name."

    lat = geo_results[0]["lat"]
    lon = geo_results[0]["lon"]
    resolved_name = geo_results[0]["display_name"]

    # --- Step 2: search Foursquare Places around those coordinates ---
    search_params = {
        "ll": f"{lat},{lon}",
        "radius": 2000,
        "limit": 5,
        "fields": "fsq_place_id,name,categories,location,distance",
    }
    if interest:
        search_params["query"] = interest

    try:
        search_resp = requests.get(
            FSQ_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {FSQ_API_KEY}",
                "X-Places-Api-Version": "2025-06-17",
                "accept": "application/json",
            },
            params=search_params,
            timeout=10,
        )
    except requests.RequestException as e:
        return f"Error searching places: {e}"

    if search_resp.status_code != 200:
        return f"Places search failed (status {search_resp.status_code}): {search_resp.text}"

    results = search_resp.json().get("results", [])
    if not results:
        label = interest if interest else "places"
        return f"No {label} found near {resolved_name}."

    # --- Format results into a readable block for the LLM to work with ---
    lines = [f"Places near {resolved_name}:"]
    for place in results:
        name = place.get("name", "Unknown")
        category = place.get("categories", [{}])[0].get("name", "N/A")
        address = place.get("location", {}).get("address", "address not listed")
        distance = place.get("distance", "N/A")
        lines.append(f"- {name} ({category}) - {address} - {distance}m away")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Places Agent, a specialist travel assistant.

Given an area and a traveler's interests, use the find_places tool to look up
REAL, current places - never invent places from memory. Always call the tool
at least once per request.

If the user names multiple interests, call the tool once per interest so you
get relevant results for each.

Present the results as a concise, friendly list: for each place, give the
name, a one-line note on why it fits, and roughly how far it is. Keep the
whole answer short - this output may be merged with other agents' output
by an orchestrator, so skip greetings and sign-offs.
"""

agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[find_places],
)

# ---------------------------------------------------------------------------
# AgentCore Runtime wiring
# ---------------------------------------------------------------------------
app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    """
    Entry point AgentCore Runtime calls on each invocation.
    Expected payload: {"area": "Jayanagar, Bengaluru", "interests": "cafe, temple"}
    """
    area = payload.get("area", "")
    interests = payload.get("interests", "")

    if not area:
        return {"error": "Please provide an 'area' in the payload."}

    prompt = f"Find places to visit in {area}."
    if interests:
        prompt += f" I'm interested in: {interests}."

    result = agent(prompt)
    return {"result": str(result)}


if __name__ == "__main__":
    # Local smoke test - runs without any AWS deployment.
    # Requires FSQ_API_KEY to be set in the environment.
    test_payload = {"area": "Jayanagar, Bengaluru", "interests": "cafe, temple"}
    print(invoke(test_payload))