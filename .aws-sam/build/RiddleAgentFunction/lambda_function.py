import json
import boto3
from botocore.exceptions import ClientError

def run_coding_riddle_agent(user_input, chat_history=None):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "amazon.nova-pro-v1:0"

    system_prompt = [{"text": (
        "You are an interactive coding riddle master agent. Your job is to present "
        "clever programming puzzles, data structure riddles, or algorithmic brainteasers "
        "to the user. Evaluate their answers, give hints if they struggle, and keep track "
        "of their score or progress in a friendly, conversational tone."
    )}]

    messages = chat_history if chat_history else []
    messages.append({"role": "user", "content": [{"text": user_input}]})

    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompt,
            inferenceConfig={"maxTokens": 1000, "temperature": 0.7}
        )
        reply = response["output"]["message"]["content"][0]["text"]
        messages.append({"role": "assistant", "content": [{"text": reply}]})
        return reply, messages
    except ClientError as e:
        return f"Error invoking Amazon Nova: {e}", messages

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))

        # Telegram webhook path
        if "message" in body:
            user_text = body["message"].get("text", "Give me a riddle.")
            reply, _ = run_coding_riddle_agent(user_text)
            return {
                "statusCode": 200,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"reply": reply})
            }

        # Frontend / Mini App path
        user_input = body.get("input", "Give me a medium difficulty data structure riddle.")
        reply, _ = run_coding_riddle_agent(user_input)
        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"reply": reply})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
