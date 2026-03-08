# Pinecone RAG — DevOps Copilot
# Phase 2 — Cloud Vector Database
# Sourav Mohanty

import os
import json
import boto3
import PyPDF2
from pinecone import Pinecone

# ─── Configuration ───────────────────────────
PINECONE_API_KEY = "pcsk_4iRZ1u_4yqpg8jKtHPreXz8bVfxrWtwXyJiYoroTi3Hci53c1xCQ8dTF8KftNFBW2hEb4g"
INDEX_NAME = "dibs-copilot"
DOCS_FOLDER = "documents"
REGION = "ap-south-1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ─── AWS Bedrock Client ───────────────────────
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION
)

# ─── Pinecone Client ──────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ─── Get Embedding from Bedrock ───────────────
def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

# ─── Ask Claude via Bedrock ───────────────────
def ask_claude(question, context):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{
            "role": "user",
            "content": f"""You are DevOps AI 
Copilot by Sourav Mohanty.
Answer ONLY from context below.
If not found say: "I could not find this."
Plain text only. No markdown.

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

# ─── Read PDF ─────────────────────────────────
def read_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {i}]\n{page_text}"
    return text

# ─── Split Into Chunks ────────────────────────
def split_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ─── Index Documents to Pinecone ─────────────
def build_knowledge_base():
    print("\nBuilding knowledge base in Pinecone...")

    # Read all PDFs
    all_chunks = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(
                DOCS_FOLDER, filename
            )
            print(f"Reading: {filename}")
            text = read_pdf(filepath)
            chunks = split_chunks(text)
            all_chunks.extend(chunks)
            print(f"Got {len(chunks)} chunks ✅")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Creating embeddings via Bedrock...")
    print("Storing in Pinecone cloud...")

    # Process in batches
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]

        # Create embeddings
        vectors = []
        for j, chunk in enumerate(batch):
            embedding = get_embedding(chunk)
            vectors.append({
                "id": f"chunk_{i + j}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "chunk_id": i + j
                }
            })

        # Store in Pinecone
        index.upsert(vectors=vectors)
        print(f"Stored {i + len(batch)}/{len(all_chunks)}")

    print("\nKnowledge base ready in Pinecone! ✅")
    print("Data is now on the cloud! ☁️")

# ─── Search Pinecone ──────────────────────────
def search_knowledge_base(question):
    # Convert question to embedding
    question_embedding = get_embedding(question)

    # Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=5,
        include_metadata=True
    )

    # Extract text chunks
    chunks = [
        match["metadata"]["text"]
        for match in results["matches"]
    ]

    return chunks

# ─── Main ─────────────────────────────────────
def main():
    os.makedirs(DOCS_FOLDER, exist_ok=True)

    # Check PDFs exist
    pdfs = [f for f in os.listdir(DOCS_FOLDER)
            if f.endswith(".pdf")]
    if not pdfs:
        print(f"Add PDFs to {DOCS_FOLDER}/ folder!")
        return

    print("\n" + "=" * 50)
    print("  DevOps Copilot — Pinecone + Bedrock")
    print("  Cloud Vector DB + Claude AI")
    print("  Built by Sourav Mohanty")
    print("=" * 50)

    # Check if Pinecone has data
    stats = index.describe_index_stats()
    total_vectors = stats["total_vector_count"]

    if total_vectors == 0:
        print("No data in Pinecone — building now...")
        build_knowledge_base()
    else:
        print(f"Found {total_vectors} vectors in Pinecone ✅")
        print("Skipping rebuild — using existing data!")

    print("\nCommands: quit, rebuild, help")
    print("=" * 50)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "quit":
            print("Goodbye! 🚀")
            break

        if question.lower() == "rebuild":
            print("Rebuilding knowledge base...")
            index.delete(delete_all=True)
            build_knowledge_base()
            continue

        if question.lower() == "help":
            print("\nSample questions:")
            print("• What AWS services does DIBS use?")
            print("• How do I onboard a new tenant?")
            print("• What happens when login fails?")
            print("• What is the monthly cost?")
            continue

        if not question:
            continue

        print("\nSearching Pinecone cloud...")

        try:
            # Search cloud vector DB
            chunks = search_knowledge_base(question)

            if not chunks:
                print("No relevant docs found.")
                continue

            # Build context
            context = "\n\n---\n\n".join(chunks)

            # Ask Claude
            answer = ask_claude(question, context)

            print(f"\nCopilot: {answer}")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()