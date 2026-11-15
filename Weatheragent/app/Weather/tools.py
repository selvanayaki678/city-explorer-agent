import math, os, threading, time
from functools import lru_cache

import requests
from langchain_core.tools import tool

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "city-explorer-conditions-agent/1.0")


class ConditionsError(Exception):
    """A problem the agent should report in plain words."""


_lock = threading.Lock()
_last_call = -math.inf


@lru_cache(maxsize=256)
def geocode(area: str):
    """Same approach as the Places Agent: cached, 1 request/second."""
    global _last_call
    with _lock:
        wait = 1.1 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(NOMINATIM_URL,
                             params={"q": area, "format": "json", "limit": 1},
                             headers={"User-Agent": USER_AGENT}, timeout=10)
            r.raise_for_status()
            results = r.json()
        except requests.RequestException as e:
            raise ConditionsError(f"Error geocoding '{area}': {e}")
        finally:
            _last_call = time.monotonic()
    if not results:
        raise ConditionsError(f"Could not find a location matching '{area}'. Try a more specific name.")
    f = results[0]
    return float(f["lat"]), float(f["lon"]), f["display_name"]


def aqi_category(aqi):
    """Plain code, not LLM guesswork: US AQI bands."""
    for limit, label in [(50, "Good"), (100, "Moderate"),
                         (150, "Unhealthy for sensitive groups"),
                         (200, "Unhealthy"), (300, "Very unhealthy")]:
        if aqi <= limit:
            return label
    return "Hazardous"


@tool
def get_air_quality(area: str) -> str:
    """Current air quality for an area: US AQI with rating, PM2.5, PM10 and UV index.

    Args:
        area: City, neighborhood or address (e.g. "Indiranagar, Bengaluru").
    """
    try:
        lat, lon, resolved = geocode(area)
        r = requests.get(AQ_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "us_aqi,pm2_5,pm10,uv_index", "timezone": "auto"}, timeout=10)
        r.raise_for_status()
        c = r.json()["current"]
    except ConditionsError as e:
        return f"Error: {e}"
    except (requests.RequestException, KeyError) as e:
        return f"Error: air quality lookup failed: {e}"

    aqi = c.get("us_aqi")
    rating = f"{aqi} ({aqi_category(aqi)})" if aqi is not None else "not available"
    return (f"Air quality near {resolved} (as of {c.get('time')}): US AQI {rating}, "
            f"PM2.5 {c.get('pm2_5')} ug/m3, PM10 {c.get('pm10')} ug/m3, UV index {c.get('uv_index')}.")


@tool
def get_weather(area: str) -> str:
    """Current temperature plus today's rain chance, sunrise and sunset for an area.

    Args:
        area: City, neighborhood or address (e.g. "Indiranagar, Bengaluru").
    """
    try:
        lat, lon, resolved = geocode(area)
        r = requests.get(WX_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,precipitation",
            "daily": "sunrise,sunset,precipitation_probability_max",
            "forecast_days": 1, "timezone": "auto"}, timeout=10)
        r.raise_for_status()
        j = r.json()
        c, d = j["current"], j["daily"]
    except ConditionsError as e:
        return f"Error: {e}"
    except (requests.RequestException, KeyError) as e:
        return f"Error: weather lookup failed: {e}"

    return (f"Weather near {resolved}: {c.get('temperature_2m')} C now, "
            f"rain chance today up to {d['precipitation_probability_max'][0]}%, "
            f"sunrise {d['sunrise'][0][-5:]}, sunset {d['sunset'][0][-5:]} (local time).")