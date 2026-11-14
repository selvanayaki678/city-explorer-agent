"""
Ask the local Places Agent (A2A server) a question and print its answer.

Start the agent first in another terminal:   agentcore dev

Usage (from the project root):
    .\\app\\Myagent\\.venv\\Scripts\\python.exe ask.py "cafes in Jayanagar, Bengaluru"
    .\\app\\Myagent\\.venv\\Scripts\\python.exe ask.py "Plan an evening in Indiranagar: cafe, temple, then dinner"
    .\\app\\Myagent\\.venv\\Scripts\\python.exe ask.py --raw "cafes in Jayanagar"   (show the full A2A JSON)
"""

import argparse
import json
import sys
import uuid

import requests


def texts_from_parts(parts):
    return [p["text"] for p in (parts or []) if isinstance(p, dict) and p.get("kind") == "text" and "text" in p]


def extract_answer(result):
    """Pull the agent's answer out of an A2A result (a Message or a Task)."""
    if result.get("kind") == "message":
        return "\n".join(texts_from_parts(result.get("parts")))

    # Task: the answer normally sits in the artifacts, else in the status message
    texts = []
    for artifact in result.get("artifacts") or []:
        texts += texts_from_parts(artifact.get("parts"))
    if not texts:
        texts = texts_from_parts(((result.get("status") or {}).get("message") or {}).get("parts"))
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description="Ask the local A2A Places Agent a question.")
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:9000/")
    parser.add_argument("--raw", action="store_true", help="print the full JSON-RPC response")
    args = parser.parse_args()

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": args.question}],
            }
        },
    }

    print(f"Asking: {args.question}\n(waiting, this can take 10-60 seconds while the agent calls its tools)\n")
    try:
        resp = requests.post(args.url, json=payload, timeout=180)
    except requests.ConnectionError:
        sys.exit(f"Could not connect to {args.url}\nIs 'agentcore dev' running, and is the protocol A2A?")
    except requests.Timeout:
        sys.exit("Timed out after 180 seconds. Check the agentcore dev terminal for errors.")

    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    if args.raw:
        print(json.dumps(data, indent=2))
        return

    if "error" in data:
        sys.exit(f"The agent returned an error: {json.dumps(data['error'])}")

    result = data.get("result", {})
    state = (result.get("status") or {}).get("state")
    if state and state != "completed":
        print(f"[task state: {state}]")

    answer = extract_answer(result)
    if not answer:
        sys.exit("No answer text found. Run again with --raw and check the response.")

    print("ANSWER:\n" + "-" * 60)
    print(answer)
    print("-" * 60)

    if "google.com/maps" in answer:
        print("CHECK: map links present -> the agent used its tools (real data).")
    else:
        print("CHECK: no map links in the answer -> the model may have answered from memory. Tell me.")


if __name__ == "__main__":
    main()