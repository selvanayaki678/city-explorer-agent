"""
Diagnose Bedrock access from the same Python environment the agent uses.

Save at the project root (next to the agentcore\\ and app\\ folders), then run:
    .\\app\\Myagent\\.venv\\Scripts\\python.exe test_bedrock.py

It prints: which AWS account/role you are using, which Claude inference
profiles Bedrock lists for your region, and whether each candidate model
actually answers a one-line prompt.
"""

import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

REGION = os.getenv("AWS_REGION", "us-west-2")
print(f"Region: {REGION}")

# 1. Who am I? (fails here = credentials or network problem, not Bedrock)
try:
    ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    print(f"Account: {ident['Account']}\nCaller : {ident['Arn']}\n")
except (BotoCoreError, ClientError) as e:
    raise SystemExit(f"STS failed (credentials or network problem): {e}")

candidates = [
    os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
    "amazon.nova-lite-v1:0",
]

# 2. Control plane (host: bedrock.<region>.amazonaws.com) - list real Claude profile IDs
try:
    bedrock = boto3.client("bedrock", region_name=REGION)
    profiles = bedrock.list_inference_profiles(typeEquals="SYSTEM_DEFINED", maxResults=100)
    claude = [
        p["inferenceProfileId"]
        for p in profiles["inferenceProfileSummaries"]
        if "claude" in p["inferenceProfileId"]
    ]
    print("Claude inference profiles visible in this region:")
    for pid in claude:
        print(f"  {pid}")
    candidates += claude[:3]
except (BotoCoreError, ClientError) as e:
    print(f"Could not list inference profiles (control-plane call): {e}")

# 3. Runtime (host: bedrock-runtime.<region>.amazonaws.com) - the call that is failing
runtime = boto3.client("bedrock-runtime", region_name=REGION)
print("\nTesting models:")
for model_id in dict.fromkeys(candidates):  # de-duplicate, keep order
    try:
        resp = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Say hi"}]}],
            inferenceConfig={"maxTokens": 20},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        print(f"  OK    {model_id}: {text}")
    except (BotoCoreError, ClientError) as e:
        print(f"  FAIL  {model_id}: {e}")