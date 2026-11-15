import os
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from bedrock_agentcore.runtime import serve_a2a
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent
from tools import get_air_quality, get_weather

SYSTEM_PROMPT = """You are a Conditions Agent: you report outdoor conditions using live data tools.
Never answer from memory. Use only numbers that appear in tool output.
Give the AQI with its rating, temperature, rain chance and sunset, then one line of advice.
End with: Data by Open-Meteo.com.
If a tool returns an error, say so plainly and suggest a more specific area name.
Keep answers short. No greetings, sign-offs or emojis, because another agent may merge your output."""

llm = ChatBedrockConverse(
    model=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    region_name=os.getenv("AWS_REGION", "us-west-2"))
graph = create_react_agent(llm, [get_air_quality, get_weather], prompt=SYSTEM_PROMPT)


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class ConditionsExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        result = await graph.ainvoke({"messages": [("user", context.get_user_input())]})
        await event_queue.enqueue_event(new_agent_text_message(_text(result["messages"][-1].content)))

    async def cancel(self, context, event_queue):
        raise NotImplementedError


def build_card() -> AgentCard:
    return AgentCard(
        name="conditions-agent",
        description="Live air quality, weather, UV and sunset for an area. Data by Open-Meteo.com",
        url="http://localhost:9000/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"], default_output_modes=["text"],
        skills=[
            AgentSkill(id="get_air_quality", name="Air quality",
                       description="Current US AQI with rating, PM2.5, PM10 and UV index for an area.",
                       tags=["air", "aqi", "pollution"],
                       examples=["How is the air quality in Indiranagar, Bengaluru?"]),
            AgentSkill(id="get_weather", name="Weather and sunset",
                       description="Current temperature, rain chance, sunrise and sunset for an area.",
                       tags=["weather", "rain", "sunset"],
                       examples=["Will it rain in Indiranagar today?"]),
        ])


if __name__ == "__main__":
    serve_a2a(ConditionsExecutor(), build_card(), host=os.getenv("A2A_HOST"))