# Project 2: AI Chatbot With Persistent Memory
# Sourav's AI Journey - Phase 1

import anthropic
import json
import os

# Connect to Claude
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Memory file path
MEMORY_FILE = "chat_memory.json"

def load_memory():
    """Load conversation history from file"""
    # Check if memory file exists
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    # If no file exists yet - start fresh
    return []

def save_memory(conversation_history):
    """Save conversation history to file"""
    with open(MEMORY_FILE, "w") as f:
        json.dump(conversation_history, f, indent=2)

def chat(user_input, conversation_history):
    """Send message to Claude and get response"""
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Send to Claude
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="""You are a helpful DevOps assistant 
        for a Cloud engineer. 
        You have memory of all past conversations.
        When relevant, refer to things the user 
        told you before.
        Always respond in plain text only.
        Never use markdown or special characters.""",
        messages=conversation_history
    )
    
    # Extract response text
    ai_message = response.content[0].text
    
    # Add AI response to history
    conversation_history.append({
        "role": "assistant",
        "content": ai_message
    })
    
    return ai_message, conversation_history

def main():
    # Load existing memory
    conversation_history = load_memory()
    
    print("=" * 40)
    print("   Sourav's AI Chatbot - With Memory")
    print("=" * 40)
    
    # Tell user if memory exists
    if len(conversation_history) > 0:
        print(f"Memory loaded! I remember our")
        print(f"{len(conversation_history)} past messages.")
    else:
        print("Fresh start - no memory yet!")
    
    print("Type 'quit' to exit")
    print("Type 'forget' to clear all memory")
    print("=" * 40)
    
    while True:
        user_input = input("\nYou: ")
        
        # Exit
        if user_input.lower() == "quit":
            save_memory(conversation_history)
            print("Memory saved! See you next time!")
            break
        
        # Clear memory command
        if user_input.lower() == "forget":
            conversation_history = []
            save_memory(conversation_history)
            print("Memory cleared! Fresh start.")
            continue
        
        # Skip empty input
        if user_input.strip() == "":
            continue
        
        # Get AI response
        ai_message, conversation_history = chat(
            user_input, 
            conversation_history
        )
        
        # Save after every message
        save_memory(conversation_history)
        
        print(f"\nAI: {ai_message}")

# Run the program
if __name__ == "__main__":
    main()