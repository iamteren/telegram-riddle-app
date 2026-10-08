import json
import boto3
from botocore.exceptions import ClientError

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS"
}

def run_coding_riddle_agent(user_input, chat_history=None):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "amazon.nova-pro-v1:0"

    system_prompt = [{"text": (
        "You are an interactive coding riddle master agent. Present clever programming "
        "puzzles, data structure riddles, and algorithmic brainteasers at the requested "
        "difficulty level. Format problems clearly with TITLE: tags, TAGS: tags, and PROBLEM: "
        "description with examples, constraints, and expected output. "
        "Evaluate solutions thoroughly — check correctness, time complexity, space complexity, "
        "and edge cases. Give hints when asked without revealing the full solution."
    )}]

    messages = chat_history if chat_history else []
    messages.append({"role": "user", "content": [{"text": user_input}]})

    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompt,
            inferenceConfig={"maxTokens": 2000, "temperature": 0.7}
        )
        reply = response["output"]["message"]["content"][0]["text"]
        messages.append({"role": "assistant", "content": [{"text": reply}]})
        return reply, messages
    except ClientError as e:
        return f"Error: {e}", messages

def lambda_handler(event, context):
    # Handle preflight OPTIONS request
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    try:
        body = json.loads(event.get("body", "{}"))
        user_input = body.get("input", "Give me a medium difficulty coding problem.")
        chat_history = body.get("chat_history", None)

        reply, updated_history = run_coding_riddle_agent(user_input, chat_history)

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "reply": reply,
                "chat_history": updated_history
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)})
        }
