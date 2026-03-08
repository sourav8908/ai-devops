# Bedrock Test — Simple
# Make sure AWS Bedrock is working

import boto3
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

# ─── Test 1: Claude via Bedrock ──────────────
def test_claude():
    print("Testing Claude on Bedrock...")
    
    llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="ap-south-1"
)
    
    response = llm.invoke(
        "Say hello and tell me you are "
        "running on AWS Bedrock in one sentence."
    )
    
    print(f"Claude says: {response.content}")
    print("Claude on Bedrock working ✅\n")

# ─── Test 2: Embeddings via Bedrock ──────────
def test_embeddings():
    print("Testing Embeddings on Bedrock...")
    
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="ap-south-1"
    )
    
    # Convert one sentence to vector
    vector = embeddings.embed_query(
        "Hello from AWS Bedrock!"
    )
    
    print(f"Vector length: {len(vector)} dimensions")
    print(f"First 5 numbers: {vector[:5]}")
    print("Embeddings on Bedrock working ✅\n")

# ─── Main ─────────────────────────────────────
def main():
    print("=" * 45)
    print("  AWS Bedrock Connection Test")
    print("=" * 45)
    
    try:
        test_claude()
    except Exception as e:
        print(f"Claude test failed: {str(e)}\n")
    
    try:
        test_embeddings()
    except Exception as e:
        print(f"Embeddings test failed: {str(e)}\n")
    
    print("=" * 45)
    print("  Test Complete!")
    print("=" * 45)

if __name__ == "__main__":
    main()