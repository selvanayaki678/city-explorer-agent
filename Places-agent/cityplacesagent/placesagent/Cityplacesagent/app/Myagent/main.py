"""
Places Agent - an A2A (Agent-to-Agent) server on Amazon Bedrock AgentCore Runtime.

Live place data plus computation that a plain LLM can't do.
Data:   Nominatim (geocoding, free) + Foursquare Places API (live places).
Skills: find_places, plan_walking_route, meet_in_the_middle, compare_areas
        (each tool is advertised as a skill on the Agent Card).

A2A contract: port 9000, Agent Card at /.well-known/agent-card.json, JSON-RPC at /.

Env vars:
    FSQ_API_KEY            required. Local: agentcore\\.env.local. Deployed: agentcore.json envVars.
    NOMINATIM_USER_AGENT   optional. Identify your app per Nominatim's usage policy,
                           e.g. "city-explorer-agent/1.0 (https://github.com/you/repo)"
    BEDROCK_MODEL_ID       optional. Defaults to the Haiku 4.5 profile below.
    AWS_REGION             optional. Defaults to us-west-2.
    A2A_HOST               optional. Set to 0.0.0.0 only if the deployed runtime can't reach the
                           server (serve_a2a binds 127.0.0.1 unless it detects a container).

Local:  agentcore dev   (needs "protocol": "A2A" in agentcore.json)
Deploy: agentcore deploy
"""

import math
import os
import threading
import time
from functools import lru_cache
from urllib.parse import quote

import requests
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from bedrock_agentcore.runtime import serve_a2a
from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a.executor import StrandsA2AExecutor

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FSQ_SEARCH_URL = "https://places-api.foursquare.com/places/search"
USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "city-explorer-agent/1.0")
FSQ_FIELDS = "fsq_place_id,name,categories,location,distance,latitude,longitude"
WALK_METERS_PER_MIN = 5000 / 60  # ~5 km/h


class PlacesError(Exception):
    """A problem the agent should report to the user in plain words."""


# ---------------------------------------------------------------------------
# Geocoding (Nominatim): cached, and limited to 1 request/second per its policy
# ---------------------------------------------------------------------------
_nominatim_lock = threading.Lock()
_last_nominatim_call = -math.inf


@lru_cache(maxsize=256)
def geocode(area: str):
    """Return (lat, lon, display_name) for a free-text place name."""
    global _last_nominatim_call
    with _nominatim_lock:
        wait = 1.1 - (time.monotonic() - _last_nominatim_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": area, "format": "json", "limit": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
        except requests.RequestException as e:
            raise PlacesError(f"Error geocoding '{area}': {e}")
        finally:
            _last_nominatim_call = time.monotonic()

    if not results:
        raise PlacesError(f"Could not find a location matching '{area}'. Try a more specific name.")
    first = results[0]
    return float(first["lat"]), float(first["lon"]), first["display_name"]


# ---------------------------------------------------------------------------
# Foursquare search
# ---------------------------------------------------------------------------
def search_foursquare(lat, lon, query="", radius=2000, limit=5):
    api_key = os.environ.get("FSQ_API_KEY")
    if not api_key:
        raise PlacesError("FSQ_API_KEY is not set. Cannot search for places.")

    params = {"ll": f"{lat},{lon}", "radius": int(radius), "limit": limit, "fields": FSQ_FIELDS}
    if query:
        params["query"] = query

    try:
        resp = requests.get(
            FSQ_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Places-Api-Version": "2025-06-17",
                "accept": "application/json",
            },
            params=params,
            timeout=10,
        )
    except requests.RequestException as e:
        raise PlacesError(f"Error searching places: {e}")

    if resp.status_code != 200:
        raise PlacesError(f"Places search failed (status {resp.status_code}): {resp.text[:200]}")
    return resp.json().get("results", [])


def _coords(place):
    lat, lon = place.get("latitude"), place.get("longitude")
    if lat is None or lon is None:
        main = (place.get("geocodes") or {}).get("main") or {}
        lat, lon = main.get("latitude"), main.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _summarize(place):
    return {
        "id": place.get("fsq_place_id") or place.get("name"),
        "name": place.get("name", "Unknown"),
        "category": (place.get("categories") or [{}])[0].get("name", "N/A"),
        "address": (place.get("location") or {}).get("address", "address not listed"),
        "distance_m": place.get("distance"),
        "coords": _coords(place),
    }


# ---------------------------------------------------------------------------
# Geometry and formatting helpers (plain code: this is what an LLM can't do reliably)
# ---------------------------------------------------------------------------
def haversine_m(a, b):
    """Straight-line distance in meters between two (lat, lon) points."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def midpoint(a, b):
    """Geographic midpoint of two (lat, lon) points."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    bx = math.cos(la2) * math.cos(lo2 - lo1)
    by = math.cos(la2) * math.sin(lo2 - lo1)
    lat3 = math.atan2(math.sin(la1) + math.sin(la2), math.sqrt((math.cos(la1) + bx) ** 2 + by**2))
    lon3 = lo1 + math.atan2(by, math.cos(la1) + bx)
    return math.degrees(lat3), (math.degrees(lon3) + 540) % 360 - 180


def fmt_dist(meters):
    return f"{round(meters)} m" if meters < 1000 else f"{meters / 1000:.1f} km"


def walk_min(meters):
    return max(1, round(meters / WALK_METERS_PER_MIN))


def map_link(coords):
    return f"https://www.google.com/maps/search/?api=1&query={coords[0]},{coords[1]}"


def route_link(origin, stops):
    """Google Maps walking directions: origin -> stops -> last stop."""
    pts = [f"{lat},{lon}" for lat, lon in stops]
    link = (
        "https://www.google.com/maps/dir/?api=1&travelmode=walking"
        f"&origin={origin[0]},{origin[1]}&destination={pts[-1]}"
    )
    if len(pts) > 1:
        link += "&waypoints=" + quote("|".join(pts[:-1]), safe=",")
    return link


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def find_places(area: str, interest: str = "") -> str:
    """Find real, current places near an area, with distance and a map link.

    Use for a simple "what's near X" request about ONE kind of place.

    Args:
        area: City, neighborhood or address (e.g. "Jayanagar, Bengaluru").
        interest: Kind of place (e.g. "cafe", "temple", "museum", "park").
                  Leave empty for top nearby places of any type.

    Returns:
        Up to 5 places with category, address, distance from the area center and
        a map link, or an error message.
    """
    try:
        lat, lon, resolved = geocode(area)
        results = search_foursquare(lat, lon, interest)
    except PlacesError as e:
        return f"Error: {e}"

    if not results:
        return f"No {interest or 'places'} found near {resolved}."

    lines = [f"Places near {resolved}:"]
    for p in map(_summarize, results):
        link = f" | map: {map_link(p['coords'])}" if p["coords"] else ""
        lines.append(f"- {p['name']} ({p['category']}) - {p['address']} - {p['distance_m']} m from center{link}")
    return "\n".join(lines)


@tool
def plan_walking_route(area: str, interests: list[str], keep_order: bool = False) -> str:
    """Build a multi-stop walking plan: pick one real place per interest and order the stops.

    Use when the user wants a plan, route, itinerary or "evening" covering SEVERAL kinds
    of places (e.g. cafe, then temple, then dinner). Distances are computed, not guessed.

    Args:
        area: Where to start (neighborhood, address or landmark).
        interests: Kinds of places to include, one stop each (2 to 5), e.g. ["cafe", "temple", "restaurant"].
        keep_order: True if the interests are meant to be visited in the given order
                    (breakfast, then sightseeing, then dinner). False lets the tool
                    pick the order that minimizes walking.

    Returns:
        Ordered stops with leg distances, straight-line walking times, totals and a
        Google Maps walking-route link, or an error message.
    """
    interests = [i for i in (interests or []) if i and i.strip()][:5]
    if len(interests) < 2:
        return "Error: give at least two interests to plan a route (use find_places for one)."

    try:
        lat, lon, resolved = geocode(area)
        candidates, missing = {}, []
        for interest in interests:
            found = [c for c in map(_summarize, search_foursquare(lat, lon, interest, radius=1500, limit=5)) if c["coords"]]
            if found:
                candidates[interest] = found
            else:
                missing.append(interest)
    except PlacesError as e:
        return f"Error: {e}"

    if not candidates:
        return f"No matching places found near {resolved} for: {', '.join(interests)}."

    start = (lat, lon)
    current, used, stops = start, set(), []
    remaining = [i for i in interests if i in candidates]

    while remaining:
        def best_for(interest):
            options = [c for c in candidates[interest] if c["id"] not in used]
            return min(options, key=lambda c: haversine_m(current, c["coords"])) if options else None

        if keep_order:
            interest = remaining[0]
        else:
            scored = [(haversine_m(current, b["coords"]), i) for i in remaining if (b := best_for(i))]
            if not scored:
                break
            interest = min(scored)[1]

        remaining.remove(interest)
        pick = best_for(interest)
        if pick is None:
            missing.append(interest)
            continue
        leg = haversine_m(current, pick["coords"])
        stops.append((interest, pick, leg))
        used.add(pick["id"])
        current = pick["coords"]

    if not stops:
        return f"Could not build a route near {resolved}."

    total = sum(leg for _, _, leg in stops)
    lines = [f"Walking plan from {resolved} (straight-line estimates; real streets are usually longer):"]
    for n, (interest, p, leg) in enumerate(stops, 1):
        lines.append(
            f"{n}. [{interest}] {p['name']} ({p['category']}) - {p['address']} - "
            f"{fmt_dist(leg)}, ~{walk_min(leg)} min walk - map: {map_link(p['coords'])}"
        )
    lines.append(f"Total walking: about {fmt_dist(total)}, ~{walk_min(total)} min (not counting time at stops).")
    lines.append("Full route in Google Maps: " + route_link(start, [p["coords"] for _, p, _ in stops]))
    if missing:
        lines.append("No match found for: " + ", ".join(missing))
    return "\n".join(lines)


@tool
def meet_in_the_middle(area_a: str, area_b: str, interest: str = "cafe") -> str:
    """Find places that are fair for two people starting in different areas.

    Use when two people (or groups) in different places want to meet. The tool
    computes the geographic midpoint and each person's distance to every spot.

    Args:
        area_a: First person's area, neighborhood or address.
        area_b: Second person's area, neighborhood or address.
        interest: Kind of place to meet at (default "cafe").

    Returns:
        Up to 5 spots near the midpoint with each person's straight-line distance
        and a map link, or an error message.
    """
    try:
        a_lat, a_lon, a_name = geocode(area_a)
        b_lat, b_lon, b_name = geocode(area_b)
        a, b = (a_lat, a_lon), (b_lat, b_lon)
        apart = haversine_m(a, b)
        mid = midpoint(a, b)
        radius = min(max(apart / 4, 1000), 5000)
        results = search_foursquare(mid[0], mid[1], interest, radius=radius, limit=10)
    except PlacesError as e:
        return f"Error: {e}"

    rows = []
    for p in map(_summarize, results):
        if not p["coords"]:
            continue
        da, db = haversine_m(a, p["coords"]), haversine_m(b, p["coords"])
        rows.append((max(da, db), da, db, p))
    rows.sort(key=lambda r: r[0])

    if not rows:
        return f"No {interest} found near the midpoint between {area_a} and {area_b}."

    lines = [f"{area_a} and {area_b} are about {fmt_dist(apart)} apart (straight line). Spots near the midpoint:"]
    if apart > 40000:
        lines.append("Note: these areas are far apart, so a midpoint meeting may not be practical.")
    for _, da, db, p in rows[:5]:
        lines.append(
            f"- {p['name']} ({p['category']}) - {p['address']} - "
            f"{area_a}: {fmt_dist(da)}, {area_b}: {fmt_dist(db)} - map: {map_link(p['coords'])}"
        )
    return "\n".join(lines)


@tool
def compare_areas(areas: list[str], interest: str) -> str:
    """Compare how many places of one kind exist in each of several areas, using live data.

    Use for "which neighborhood has more cafes / parks / temples" questions.

    Args:
        areas: 2 to 4 areas or neighborhoods to compare.
        interest: Kind of place to count (e.g. "cafe", "park", "temple").

    Returns:
        Live counts within 1.5 km of each area's center (capped at 50 per area)
        with a few example names, or an error message.
    """
    areas = [a for a in (areas or []) if a and a.strip()][:4]
    if len(areas) < 2 or not interest:
        return "Error: give at least two areas and one kind of place to compare."

    lines = [f"Number of '{interest}' places within 1.5 km of each area's center (counts stop at 50):"]
    try:
        for area in areas:
            lat, lon, _ = geocode(area)
            found = search_foursquare(lat, lon, interest, radius=1500, limit=50)
            count = f"{len(found)}+" if len(found) >= 50 else str(len(found))
            examples = ", ".join(p.get("name", "?") for p in found[:3]) or "none found"
            lines.append(f"- {area}: {count} (e.g. {examples})")
    except PlacesError as e:
        return f"Error: {e}"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Places Agent: a specialist that answers location questions using
live data tools. Never answer from memory.

Tool choice:
- find_places: one kind of place near one area.
- plan_walking_route: a plan, route or outing with several kinds of places.
- meet_in_the_middle: two people in different areas who want somewhere fair to meet.
- compare_areas: which area has more of something.

Rules:
- Use only places, addresses, distances and counts that appear in tool output. Never invent them.
- Distances are straight-line estimates: say "about" and mention it briefly when giving walking times.
- Include the map link for every place you recommend, and the route link for plans.
- If a tool returns an error or no results, say so plainly and suggest a fix, such as a more specific area name.
- Keep answers short. No greetings, sign-offs or emojis, because another agent may merge your output.
"""

model = BedrockModel(
    model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    region_name=os.getenv("AWS_REGION", "us-west-2"),
)


agent = Agent(
    name="places-agent",
    description=(
        "Finds real places near an area, plans multi-stop walking routes, finds fair meeting "
        "spots for two people, and compares areas, using live Foursquare data."
    ),
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[find_places, plan_walking_route, meet_in_the_middle, compare_areas],
    callback_handler=None,  # don't print streamed text to the console (breaks on Windows emojis)
)


# ---------------------------------------------------------------------------
# Agent Card: what other agents discover at /.well-known/agent-card.json.
# serve_a2a() would otherwise publish one generic "main" skill, so we declare
# one skill per tool. Other agents (and their LLMs) choose us based on these.
# ---------------------------------------------------------------------------
def build_card() -> AgentCard:
    return AgentCard(
        name="places-agent",
        description=(
            "Specialist for real-world place questions. Uses live Foursquare data and geometry "
            "to find places, plan walking routes, pick fair meeting spots and compare areas."
        ),
        url="http://localhost:9000/",  # replaced by serve_a2a with the runtime URL when deployed
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="find_places",
                name="Find places",
                description="Real places of one kind near an area, with distance and map links.",
                tags=["places", "search", "local"],
                examples=["cafes in Jayanagar, Bengaluru", "temples near Indiranagar"],
            ),
            AgentSkill(
                id="plan_walking_route",
                name="Plan a walking route",
                description="Pick one real place per interest, order the stops, and compute leg distances, walking times and a map route.",
                tags=["itinerary", "route", "walking"],
                examples=["Plan an evening in Indiranagar: cafe, temple, then dinner"],
            ),
            AgentSkill(
                id="meet_in_the_middle",
                name="Meet in the middle",
                description="Fair meeting spots for two people in different areas, with each person's distance.",
                tags=["meetup", "midpoint", "distance"],
                examples=["Where can friends from Jayanagar and Whitefield meet for coffee?"],
            ),
            AgentSkill(
                id="compare_areas",
                name="Compare areas",
                description="Live counts of a kind of place across 2 to 4 areas.",
                tags=["analytics", "compare", "neighborhoods"],
                examples=["Which has more cafes, Indiranagar or Koramangala?"],
            ),
        ],
    )


if __name__ == "__main__":
    # A2A server on port 9000 (fixed by the AgentCore Runtime contract).
    serve_a2a(StrandsA2AExecutor(agent), build_card(), host=os.getenv("A2A_HOST"))
