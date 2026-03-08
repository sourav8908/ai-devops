# Project 3 - File 2: RAG Chatbot
# Answers questions from YOUR documentation

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
import time


# ─── Settings ────────────────────────────────
DB_FOLDER = "dibs_knowledge"
COLLECTION_NAME = "dibs_vault_docs"
TOP_K = 5 # number of relevant chunks to retrieve

# ─── Load Models ─────────────────────────────
print("Loading AI models...")
claude_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
db_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = db_client.get_collection(COLLECTION_NAME)
print("Models loaded ✅")

# ─── Step 1: Search ChromaDB ─────────────────
def search_knowledge_base(question):
    # Convert question to embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()
    
    # Search for similar chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K
    )
    
    # Return the relevant text chunks
    relevant_chunks = results['documents'][0]
    return relevant_chunks

# ─── Step 2: Ask Claude With Context ─────────
def ask_claude(question, relevant_chunks):
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(relevant_chunks)
    
    # Build prompt with context
    prompt = f"""You are an expert assistant for 
the DIBS Vault platform built by Sourav at 
Unloq Solutions.

Answer the question using ONLY the context 
provided below. If the answer is not in the 
context, say "I could not find this in the 
documentation."

CONTEXT FROM DOCUMENTATION:
{context}

QUESTION: {question}

ANSWER:"""
    
    # Retry up to 3 times if overloaded
    for attempt in range(3):
        try:
            response = claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
            
        except Exception as e:
            if "overloaded" in str(e).lower():
                print(f"API busy, retrying in 10 seconds... (attempt {attempt + 1}/3)")
                time.sleep(10)
            else:
                return f"Error: {str(e)}"
    
    return "API is too busy right now. Please try again in a minute."

# ─── Main Chat Loop ───────────────────────────
def main():
    print("\n" + "=" * 50)
    print("  DIBS Vault AI Assistant")
    print("  Ask anything about your system!")
    print("  Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == "quit":
            print("Goodbye!")
            break
            
        if question.strip() == "":
            continue
        
        print("\nSearching documentation...")
        
        # Step 1: Find relevant chunks
        relevant_chunks = search_knowledge_base(question)
        
        print(f"Found {len(relevant_chunks)} relevant sections")
        print("Asking Claude...\n")
        
        # Step 2: Get answer from Claude
        answer = ask_claude(question, relevant_chunks)
        
        print(f"AI: {answer}")

if __name__ == "__main__":
    main()