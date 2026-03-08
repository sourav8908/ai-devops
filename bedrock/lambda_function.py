# Lambda Function — DevOps AI Copilot
# Phase 2 Week 3
# Sourav Mohanty

import json
import boto3
from pinecone import Pinecone
import os
# ─── Configuration ───────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "dibs-copilot")
REGION = "eu-west-2"

# ─── Initialize Clients ───────────────────────
# These run ONCE when Lambda starts
# Then reused for all requests (faster!)
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ─── Get Embedding ────────────────────────────
def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

# ─── Search Pinecone ──────────────────────────
def search_knowledge_base(question):
    question_embedding = get_embedding(question)
    results = index.query(
        vector=question_embedding,
        top_k=5,
        include_metadata=True
    )
    chunks = [
        match["metadata"]["text"]
        for match in results["matches"]
    ]
    return chunks

# ─── Ask Claude ───────────────────────────────
def ask_claude(question, chunks):
    context = "\n\n---\n\n".join(chunks)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{
            "role": "user",
            "content": f"""You are DevOps AI
Copilot by Sourav Mohanty on AWS.
Answer ONLY from context. Plain text only.

CONTEXT: {context}

QUESTION: {question}

ANSWER:"""
        }]
    })

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=body
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# ─── Lambda Handler ───────────────────────────
def lambda_handler(event, context):

    # Handle both direct and API Gateway calls
    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    else:
        body = event

    # Get question
    question = body.get("question", "")

    # Validate input
    if not question:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": "No question provided"
            })
        }

    try:
        # Search Pinecone
        chunks = search_knowledge_base(question)

        if not chunks:
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "answer": "I could not find this in the documentation."
                })
            }

        # Ask Claude
        answer = ask_claude(question, chunks)

        # Return answer
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "answer": answer,
                "question": question
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": str(e)
            })
        }