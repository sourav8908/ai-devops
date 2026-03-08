# Project 5 — DevOps AI Copilot
# Production-grade RAG Assistant
# Sourav Mohanty — AI Journey

import os
import shutil
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─── Configuration ───────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DB_FOLDER = "copilot_knowledge"
DOCS_FOLDER = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5

# ─── Embeddings Model (free, local) ──────────
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ─── Claude LLM ──────────────────────────────
LLM = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=API_KEY,
    max_tokens=500
)

# ─── RAG Prompt ──────────────────────────────
PROMPT_TEMPLATE = """You are an expert DevOps 
AI Copilot assistant built by Sourav Mohanty.

You help engineers by answering questions from 
their technical documentation and runbooks.

STRICT RULES:
1. Answer ONLY from the context provided
2. If answer not in context — say exactly:
   "I could not find this in the documentation.
    Please check manually or contact your team."
3. Always mention WHICH section you found 
   the answer in
4. Keep answers clear and practical
5. Plain text only — no markdown

CONTEXT FROM DOCUMENTATION:
{context}

ENGINEER'S QUESTION: {question}

YOUR ANSWER:"""

# ─── Load Documents ───────────────────────────
def load_documents(docs_folder):
    print(f"\nLoading documents from: {docs_folder}")
    all_pages = []
    loaded_files = []

    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_folder, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            all_pages.extend(pages)
            loaded_files.append(filename)
            print(f"Loaded: {filename} ({len(pages)} pages)")

    print(f"\nTotal: {len(loaded_files)} files, "
          f"{len(all_pages)} pages loaded ✅")
    return all_pages, loaded_files

# ─── Build Knowledge Base ─────────────────────
def build_knowledge_base():
    print("Building knowledge base...")

    # Load all PDFs
    pages, files = load_documents(DOCS_FOLDER)

    if not pages:
        print("ERROR: No PDF files found!")
        print(f"Add PDFs to: {DOCS_FOLDER}/")
        return None

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks ✅")

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        persist_directory=DB_FOLDER
    )
    print(f"Knowledge base built ✅")
    return vectorstore

# ─── Load Existing Knowledge Base ────────────
def load_knowledge_base():
    print("Loading existing knowledge base...")
    vectorstore = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=EMBEDDINGS
    )
    print("Knowledge base loaded ✅")
    return vectorstore

# ─── Build RAG Chain ──────────────────────────
def build_chain(vectorstore):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    chain = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# ─── Ask Question ─────────────────────────────
def ask_question(chain, question):
    result = chain.invoke({"query": question})
    answer = result["result"]

    # Show source documents used
    sources = result["source_documents"]
    source_pages = []
    for doc in sources:
        page = doc.metadata.get("page", "unknown")
        if page not in source_pages:
            source_pages.append(page)

    return answer, source_pages

# ─── Display Welcome ──────────────────────────
def show_welcome(ready):
    print("\n" + "=" * 55)
    print("   🤖 DevOps AI Copilot")
    print("   Built by Sourav Mohanty")
    print("   Powered by Claude AI + LangChain")
    print("=" * 55)

    if ready:
        print("   Status: READY ✅")
    else:
        print("   Status: No documents loaded ❌")

    print("\n   Commands:")
    print("   'quit'    — exit")
    print("   'rebuild' — reindex all documents")
    print("   'help'    — show sample questions")
    print("=" * 55)

# ─── Show Help ────────────────────────────────
def show_help():
    print("\n Sample questions you can ask:")
    print(" • What AWS services does this system use?")
    print(" • How do I onboard a new tenant?")
    print(" • What happens when a user cannot log in?")
    print(" • What are the security layers?")
    print(" • What is the monthly AWS cost estimate?")
    print(" • What happens if file upload fails?")
    print(" • What are the pending features?")

# ─── Main ─────────────────────────────────────
def main():

    # Create documents folder if not exists
    os.makedirs(DOCS_FOLDER, exist_ok=True)

    # Check if PDF exists in documents folder
    pdf_files = [f for f in os.listdir(DOCS_FOLDER)
                 if f.endswith(".pdf")]

    if not pdf_files:
        print(f"\nNo PDFs found in '{DOCS_FOLDER}' folder!")
        print(f"Please add your PDF files to: {DOCS_FOLDER}/")
        print("Then run the program again.")
        return

    # Build or load knowledge base
    if os.path.exists(DB_FOLDER):
        vectorstore = load_knowledge_base()
    else:
        vectorstore = build_knowledge_base()

    if not vectorstore:
        return

    # Build RAG chain
    chain = build_chain(vectorstore)

    # Show welcome screen
    show_welcome(ready=True)
    show_help()

    # Conversation history for context
    conversation_count = 0

    # Main chat loop
    while True:
        question = input("\nYou: ").strip()

        # Commands
        if question.lower() == "quit":
            print(f"\nGoodbye! You asked {conversation_count} questions.")
            print("Great work Sourav! 🚀")
            break

        if question.lower() == "rebuild":
            if os.path.exists(DB_FOLDER):
                shutil.rmtree(DB_FOLDER)
            vectorstore = build_knowledge_base()
            chain = build_chain(vectorstore)
            print("Knowledge base rebuilt! ✅")
            continue

        if question.lower() == "help":
            show_help()
            continue

        if not question:
            continue

        # Ask question
        print("\nSearching documentation...")

        try:
            answer, source_pages = ask_question(
                chain, question
            )
            conversation_count += 1

            print(f"\nCopilot: {answer}")
            print(f"\nSources: Pages {source_pages}")
            print(f"Question #{conversation_count}")

        except Exception as e:
            if "overloaded" in str(e).lower():
                print("API busy. Please try again.")
            else:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()