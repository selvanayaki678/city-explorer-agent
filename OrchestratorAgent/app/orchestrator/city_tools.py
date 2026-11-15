import json, os, uuid

REGION = os.getenv("AGENTS_REGION", "us-east-1")
PLACES_ARN = os.getenv(
    "PLACES_AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:717914742106:runtime/Cityplacesagent_Placesagent-OBqzJF2sa2")
CONDITIONS_ARN = os.getenv(
    "CONDITIONS_AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:717914742106:runtime/Weatheragent_Weather-rlfVBpGGfG")

_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client("bedrock-agentcore", region_name=REGION,
                               config=Config(read_timeout=120, retries={"max_attempts": 2}))
    return _client


def _texts(parts):
    return "".join(p.get("text", "") for p in (parts or []) if p.get("kind") == "text")


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("data:") or "\ndata:" in raw:
        chunks = [l[5:].strip() for l in raw.splitlines() if l.startswith("data:")]
        raw = chunks[-1] if chunks else raw
    return json.loads(raw)


def _extract_text(msg: dict) -> str:
    if "error" in msg:
        raise RuntimeError(msg["error"].get("message", str(msg["error"])))
    result = msg.get("result", {})
    if result.get("kind") == "message":
        return _texts(result.get("parts"))
    out = [_texts(a.get("parts")) for a in result.get("artifacts") or []]
    status_msg = (result.get("status") or {}).get("message")
    if not any(out) and status_msg:
        out.append(_texts(status_msg.get("parts")))
    return "\n".join(t for t in out if t)


def call_a2a(arn: str, text: str) -> str:
    body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "message/send",
            "params": {"message": {"role": "user", "messageId": uuid.uuid4().hex,
                                   "parts": [{"kind": "text", "text": text}]}}}
    resp = _get_client().invoke_agent_runtime(
        agentRuntimeArn=arn, runtimeSessionId=str(uuid.uuid4()),
        payload=json.dumps(body).encode())
    raw = resp["response"].read().decode()
    return _extract_text(_parse(raw)) or f"Agent returned no text. Raw reply: {raw[:300]}"


def places(question: str) -> str:
    try:
        return call_a2a(PLACES_ARN, question)
    except Exception as e:
        return f"Error: Places Agent failed: {e}"


def conditions(question: str) -> str:
    try:
        return call_a2a(CONDITIONS_ARN, question)
    except Exception as e:
        return f"Error: Conditions Agent failed: {e}"