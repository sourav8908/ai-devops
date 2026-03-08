# Project 4 — RAG with LangChain
# Same as Project 3 but cleaner!
# Sourav's AI Journey - Phase 1

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import time


# ─── Settings ────────────────────────────────
PDF_FILE = "DIBS-Vault-Phase4-Documentation.pdf"
DB_FOLDER = "dibs_langchain_db"
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# ─── Step 1: Load PDF ────────────────────────
def load_pdf(filepath):
    print(f"Loading PDF: {filepath}")
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages ✅")
    return pages

# ─── Step 2: Split Into Chunks ───────────────
def split_documents(pages):
    print("Splitting into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks ✅")
    return chunks

# ─── Step 3: Create Vector Store ─────────────
def create_vectorstore(chunks):
    print("Creating embeddings and vector store...")
    
    # Free local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )
    
    print(f"Vector store created ✅")
    return vectorstore

# ─── Step 4: Load Existing Vector Store ──────
def load_vectorstore():
    print("Loading existing knowledge base...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )
    
    print("Knowledge base loaded ✅")
    return vectorstore

# ─── Step 5: Build RAG Chain ─────────────────
def build_rag_chain(vectorstore):
    print("Building RAG chain...")
    
    # Claude LLM
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=API_KEY,
        max_tokens=400
    )
    
    # Retriever — searches vectorstore
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    
    # Custom prompt template
    prompt_template = """You are an expert 
assistant for the DIBS Vault platform 
built by Sourav at Unloq Solutions.

Answer the question using ONLY the context 
provided below. If the answer is not in the 
context, say "I could not find this in the 
documentation."

Always respond in plain text only.
No markdown, no stars, no special characters.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Build complete RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("RAG chain ready ✅")
    return chain

# ─── Main ────────────────────────────────────
def main():
    print("=" * 50)
    print("  DIBS Vault AI Assistant — LangChain")
    print("=" * 50)
    
    # Build or load knowledge base
    if os.path.exists(DB_FOLDER):
        print("Found existing knowledge base!")
        vectorstore = load_vectorstore()
    else:
        print("Building new knowledge base...")
        pages = load_pdf(PDF_FILE)
        chunks = split_documents(pages)
        vectorstore = create_vectorstore(chunks)
    
    # Build RAG chain
    chain = build_rag_chain(vectorstore)
    
    print("\n" + "=" * 50)
    print("  Ready! Ask anything about DIBS Vault")
    print("  Type 'quit' to exit")
    print("  Type 'rebuild' to reindex document")
    print("=" * 50)
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == "quit":
            print("Goodbye!")
            break
        
        if question.lower() == "rebuild":
            import shutil
            shutil.rmtree(DB_FOLDER)
            pages = load_pdf(PDF_FILE)
            chunks = split_documents(pages)
            vectorstore = create_vectorstore(chunks)
            chain = build_rag_chain(vectorstore)
            print("Knowledge base rebuilt! ✅")
            continue
            
        if question.strip() == "":
            continue
        
        print("\nSearching documentation...")

        # Retry logic for Claude API
        for attempt in range(5):
            try:
                result = chain.invoke({"query": question})
                answer = result["result"]
                break
            except Exception:
                print("Claude API busy, retrying...")
                time.sleep(3)
        else:
            answer = "Claude API is overloaded right now. Try again later."

        print(f"\nAI: {answer}")


if __name__ == "__main__":
    main()






# LangChain RAG — Core Only (10 lines!)

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_anthropic import ChatAnthropic
# from langchain.chains import RetrievalQA

# pages = PyPDFLoader("DIBS-Vault-Phase4-Documentation.pdf").load()
# chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(pages)
# vectorstore = Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
# chain = RetrievalQA.from_chain_type(ChatAnthropic(model="claude-haiku-4-5-20251001", anthropic_api_key="your-key"), retriever=vectorstore.as_retriever())
# print(chain.invoke({"query": "What AWS services does DIBS Vault use?"})["result"])