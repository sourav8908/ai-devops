# bedrock_rag_fixed.py
# No langchain.chains dependency!

import os
import json
import shutil
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Configuration ───────────────────────────
DOCS_FOLDER = "documents"
DB_FOLDER = "bedrock_knowledge"
REGION = "ap-south-1"

# ─── AWS Clients ─────────────────────────────
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION
)

EMBEDDINGS = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=REGION
)

# ─── Ask Claude Directly ─────────────────────
def ask_claude(question, context):
    prompt = f"""You are a DevOps AI Copilot 
built by Sourav Mohanty on AWS Bedrock.

Answer ONLY from context below.
If not found say: "I could not find this."
Plain text only. No markdown.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=body
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# ─── Load Documents ───────────────────────────
def load_documents():
    print("Loading documents...")
    all_pages = []

    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(
                DOCS_FOLDER, filename
            )
            pages = PyPDFLoader(filepath).load()
            all_pages.extend(pages)
            print(f"Loaded: {filename} ✅")

    return all_pages

# ─── Build Knowledge Base ─────────────────────
def build_knowledge_base():
    print("\nBuilding knowledge base...")
    pages = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks ✅")

    print("Creating embeddings via Bedrock...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        persist_directory=DB_FOLDER
    )
    print("Knowledge base ready ✅")
    return vectorstore

# ─── Load Knowledge Base ──────────────────────
def load_knowledge_base():
    print("Loading existing knowledge base...")
    return Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=EMBEDDINGS
    )

# ─── Search + Answer ──────────────────────────
def search_and_answer(vectorstore, question):
    # Search ChromaDB
    docs = vectorstore.similarity_search(
        question, k=5
    )

    # Build context from results
    context = "\n\n---\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Get page sources
    pages = list(set([
        doc.metadata.get("page", "?")
        for doc in docs
    ]))

    # Ask Claude via Bedrock
    answer = ask_claude(question, context)

    return answer, pages

# ─── Main ─────────────────────────────────────
def main():
    os.makedirs(DOCS_FOLDER, exist_ok=True)

    pdfs = [f for f in os.listdir(DOCS_FOLDER)
            if f.endswith(".pdf")]
    if not pdfs:
        print(f"Add PDFs to {DOCS_FOLDER}/ folder!")
        return

    if os.path.exists(DB_FOLDER):
        vectorstore = load_knowledge_base()
    else:
        vectorstore = build_knowledge_base()

    print("\n" + "=" * 50)
    print("  DevOps Copilot — AWS Bedrock")
    print("  Built by Sourav Mohanty")
    print("  No LangChain chains needed!")
    print("=" * 50)
    print("  Commands: quit, rebuild, help")
    print("=" * 50)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "quit":
            print("Goodbye! 🚀")
            break

        if question.lower() == "rebuild":
            if os.path.exists(DB_FOLDER):
                shutil.rmtree(DB_FOLDER)
            vectorstore = build_knowledge_base()
            print("Rebuilt! ✅")
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

        print("\nSearching via AWS Bedrock...")

        try:
            answer, pages = search_and_answer(
                vectorstore, question
            )
            print(f"\nCopilot: {answer}")
            print(f"Sources: Pages {pages}")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()