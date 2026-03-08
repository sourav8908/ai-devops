# Bedrock RAG — DevOps Copilot on AWS
# Phase 2 — Week 1
# Sourav Mohanty

import os
import shutil

# Document loaders and vectorstores
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# # LangChain experimental chain for RAG
# from langchain_experimental.chains import RetrievalQA

# # Text splitting
# from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Prompts
from langchain.prompts import PromptTemplate

# AWS Bedrock
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

# ─── Configuration ───────────────────────────
DOCS_FOLDER = "documents"
DB_FOLDER = "bedrock_knowledge"
REGION = "ap-south-1"

# ─── AWS Bedrock Models ───────────────────────
EMBEDDINGS = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=REGION
)

LLM = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name=REGION,
    model_kwargs={"max_tokens": 500}
)

# ─── RAG Prompt ──────────────────────────────
PROMPT_TEMPLATE = """You are an expert DevOps
AI Copilot assistant built by Sourav Mohanty
running on AWS Bedrock.

Answer ONLY from the context provided below.
If answer not found say:
"I could not find this in the documentation."

Plain text only. No markdown. No stars.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

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
    print("\nBuilding knowledge base on AWS...")

    pages = load_documents()

    # Split chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks ✅")

    # Store with Bedrock embeddings
    print("Creating embeddings via AWS Bedrock...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        persist_directory=DB_FOLDER
    )
    print("Knowledge base ready on AWS ✅")
    return vectorstore

# ─── Load Knowledge Base ──────────────────────
def load_knowledge_base():
    print("Loading existing knowledge base...")
    return Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=EMBEDDINGS
    )

# ─── Build Chain ─────────────────────────────
def build_chain(vectorstore):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# ─── Main ─────────────────────────────────────
def main():
    # Setup
    os.makedirs(DOCS_FOLDER, exist_ok=True)

    # Check PDFs exist
    pdfs = [f for f in os.listdir(DOCS_FOLDER)
            if f.endswith(".pdf")]
    if not pdfs:
        print(f"Add PDFs to {DOCS_FOLDER}/ folder!")
        return

    # Build or load knowledge base
    if os.path.exists(DB_FOLDER):
        vectorstore = load_knowledge_base()
    else:
        vectorstore = build_knowledge_base()

    # Build chain
    chain = build_chain(vectorstore)

    # Welcome
    print("\n" + "=" * 50)
    print("  DevOps AI Copilot — Powered by AWS Bedrock")
    print("  Claude AI + Titan Embeddings")
    print("  Built by Sourav Mohanty")
    print("=" * 50)
    print("  Commands: 'quit' 'rebuild' 'help'")
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
            chain = build_chain(vectorstore)
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
            result = chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            pages = list(set([
                doc.metadata.get("page", "?")
                for doc in sources
            ]))

            print(f"\nCopilot: {answer}")
            print(f"Sources: Pages {pages}")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()