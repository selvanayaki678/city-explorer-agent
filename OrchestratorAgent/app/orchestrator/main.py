import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp

SYSTEM_PROMPT = """You are a City Explorer orchestrator. You answer by delegating to two specialist agents:
- ask_places_agent: things to do, places, routes, meeting spots, neighborhood comparisons.
- ask_conditions_agent: air quality, weather, rain chance, UV, sunrise and sunset.

Rules:
- If the user wants to go somewhere or asks what to do AND mentions air, weather or conditions, call BOTH
  agents in the same step. Do not answer from memory.
- Send both agents the same place string, with the city included (e.g. "Indiranagar, Bengaluru"),
  so they resolve to the same location.
- Merge the answers into one short reply: places first (keep every map link), then conditions,
  then one line saying whether it is a good time to go out, based only on the conditions data.
- Use only facts that appear in the agents' replies. Never invent places, numbers or links.
- If an agent returns an error, say which part is missing and why, and still give the other part.
- Keep the Open-Meteo credit that the Conditions Agent includes.
- No greetings, sign-offs or emojis."""

app = BedrockAgentCoreApp()
_graph = None


def get_graph():
    """Heavy imports and graph construction happen on first use, not at module load."""
    global _graph
    if _graph is None:
        from langchain_aws import ChatBedrockConverse
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
        import city_tools as t 

        @tool
        def ask_places_agent(question: str) -> str:
            """Ask the Places Agent about real places: things to do, cafes, temples, walking routes,
            meeting spots for two people, or comparing neighborhoods. Always include the city,
            e.g. 'Things to do in Indiranagar, Bengaluru'."""
            return t.places(question)

        @tool
        def ask_conditions_agent(question: str) -> str:
            """Ask the Conditions Agent for live air quality (AQI), temperature, rain chance, UV index,
            sunrise and sunset for an area. Always include the city,
            e.g. 'Air quality and weather in Indiranagar, Bengaluru'."""
            return t.conditions(question)

        llm = ChatBedrockConverse(
            model=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            region_name=os.getenv("AWS_REGION", "us-east-1"))
        _graph = create_react_agent(llm, [ask_places_agent, ask_conditions_agent], prompt=SYSTEM_PROMPT)
    return _graph


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@app.entrypoint
async def invoke(payload, context=None):
    p = payload or {}
    prompt = p.get("prompt") or p.get("message") or p.get("input")
    if not prompt:
        return {"error": 'Send a JSON body like {"prompt": "I want to go to Indiranagar, Bengaluru..."}'}
    result = await get_graph().ainvoke({"messages": [("user", prompt)]})
    return {"result": _text(result["messages"][-1].content)}


if __name__ == "__main__":
    app.run()