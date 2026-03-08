# Project 3 - File 1: Index Your Documents
# Run this ONCE to build your knowledge base

import chromadb
from sentence_transformers import SentenceTransformer
# import pypdf2
import PyPDF2
import os

# ─── Settings ───────────────────────────────
PDF_FILE = "DIBS-Vault-Phase4-Documentation.pdf"
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks
DB_FOLDER = "dibs_knowledge"
COLLECTION_NAME = "dibs_vault_docs"

# ─── Step 1: Read PDF ────────────────────────
def read_pdf(filepath):
    print(f"Reading PDF: {filepath}")
    text = ""
    
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n"
                text += page_text
        
        print(f"Read {total_pages} pages successfully ✅")
    
    return text

# ─── Step 2: Split Into Chunks ───────────────
def split_into_chunks(text, chunk_size, overlap):
    print(f"Splitting text into chunks...")
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move forward with overlap
        start = end - overlap
    
    print(f"Created {len(chunks)} chunks ✅")
    return chunks

# ─── Step 3: Store in ChromaDB ───────────────
def store_in_chromadb(chunks):
    print(f"Loading embedding model...")
    
    # Load free local embedding model
    # This runs on YOUR laptop - no API needed!
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Creating embeddings and storing in ChromaDB...")
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=DB_FOLDER)
    
    # Delete existing collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared old collection")
    except:
        pass
    
    # Create fresh collection
    collection = client.create_collection(COLLECTION_NAME)
    
    # Process chunks in batches
    batch_size = 50
    total_stored = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Create embeddings for batch
        embeddings = model.encode(batch).tolist()
        
        # Create IDs for batch
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        
        # Store in ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=batch,
            ids=ids
        )
        
        total_stored += len(batch)
        print(f"Stored {total_stored}/{len(chunks)} chunks...")
    
    print(f"\nAll chunks stored in ChromaDB ✅")
    return collection

# ─── Main ────────────────────────────────────
def main():
    print("=" * 50)
    print("  DIBS Vault — Building Knowledge Base")
    print("=" * 50)
    
    # Check PDF exists
    if not os.path.exists(PDF_FILE):
        print(f"ERROR: {PDF_FILE} not found!")
        print(f"Make sure PDF is in same folder as this script")
        return
    
    # Step 1: Read PDF
    text = read_pdf(PDF_FILE)
    
    # Step 2: Split into chunks
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Step 3: Store in ChromaDB
    store_in_chromadb(chunks)
    
    print("\n" + "=" * 50)
    print("  Knowledge Base Ready!")
    print(f"  Stored in folder: {DB_FOLDER}")
    print("  Now run: python rag_chatbot.py")
    print("=" * 50)

if __name__ == "__main__":
    main()