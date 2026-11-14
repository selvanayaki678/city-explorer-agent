"""
Places Agent - finds REAL places (Foursquare Places API) near an area.
Geocodes the area first (Nominatim, free, no key), then searches Foursquare.

Env var required:
    FSQ_API_KEY  (local: .env.local in the project root; deployed: agentcore.json envVars)

Local:   agentcore dev
Deploy:  agentcore deploy
Invoke:  agentcore invoke '{"area": "Jayanagar, Bengaluru", "interests": "cafe, temple"}'
"""

import os

import requests
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FSQ_SEARCH_URL = "https://places-api.foursquare.com/places/search"


@tool
def find_places(area: str, interest: str = "") -> str:
    """Find real, current places near a given area using live location data.

    Use this tool whenever the user asks about places to visit, eat, or do in a
    specific city, neighborhood, or address. Do NOT answer from memory.

    Args:
        area: A city, neighborhood, or address (e.g. "Jayanagar, Bengaluru").
        interest: Kind of place (e.g. "cafe", "temple", "museum", "park",
                  "restaurant"). Leave empty for top nearby places of any type.

    Returns:
        Up to 5 matching places with name, category, address and distance,
        or an error message if the area wasn't found or the API call failed.
    """
    # Read at call time, so it works regardless of when env vars get loaded
    api_key = os.environ.get("FSQ_API_KEY")
    if not api_key:
        return "Error: FSQ_API_KEY is not set. Cannot search for places."

    # Step 1: geocode the area to lat/lng
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

    # Step 2: search Foursquare around those coordinates
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
                "Authorization": f"Bearer {api_key}",
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
        return f"No {interest or 'places'} found near {resolved_name}."

    lines = [f"Places near {resolved_name}:"]
    for place in results:
        name = place.get("name", "Unknown")
        category = (place.get("categories") or [{}])[0].get("name", "N/A")
        address = (place.get("location") or {}).get("address", "address not listed")
        distance = place.get("distance", "N/A")
        lines.append(f"- {name} ({category}) - {address} - {distance}m away")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a Places Agent, a specialist travel assistant.

Given an area and a traveler's interests, use the find_places tool to look up
REAL, current places - never invent places from memory. Always call the tool
at least once per request.

If the user names multiple interests, call the tool once per interest.

Present results as a concise, friendly list: for each place give the name, a
one-line note on why it fits, and roughly how far it is. Keep it short - this
output may be merged with other agents' output by an orchestrator, so skip
greetings and sign-offs.
"""

# Explicit model so we control which Claude model + region is used.
# Override with the BEDROCK_MODEL_ID env var; make sure the model is enabled
# for your account in this region.
model = BedrockModel(
    model_id=os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[find_places],
)

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    """Accepts either payload shape:

    1. {"prompt": "cafes and temples in Jayanagar"}
       -> what `agentcore invoke "..."` and the dev chat UI send
    2. {"area": "Jayanagar, Bengaluru", "interests": "cafe, temple"}
       -> structured form, handy when another agent (the orchestrator) calls this one
    """
    prompt = payload.get("prompt", "")

    if not prompt:
        area = payload.get("area", "")
        interests = payload.get("interests", "")

        if not area:
            return {"error": "Send either 'prompt' or 'area' (+ optional 'interests')."}

        prompt = f"Find places to visit in {area}."
        if interests:
            prompt += f" I'm interested in: {interests}."

    result = agent(prompt)
    return {"result": str(result)}


if __name__ == "__main__":
    # Starts the HTTP server (port 8080) - required for AgentCore Runtime
    app.run()