# Project 1: My First AI Chatbot
# Sourav's AI Journey - Phase 1

import anthropic
import os

# Connect to Claude using your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Store conversation history
conversation_history = []

print("=" * 40)
print("   Sourav's First AI Chatbot")
print("   Type 'quit' to exit")
print("=" * 40)

# Main chat loop
while True:

    # Get input from user
    user_input = input("\nYou: ")

    # Exit if user types quit
    if user_input.lower() == "quit":
        print("Goodbye! Great first project!")
        break

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # Send to Claude and get response
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""You are a helpful DevOps 
assistant for a Cloud engineer.
Always respond in plain text only.
Never use markdown, stars, headers, 
or bullet points with special characters.
Use simple numbered lists instead.""",
        messages=conversation_history
    )

    # Extract text from response
    ai_message = response.content[0].text

    # Add AI response to history
    conversation_history.append({
        "role": "assistant",
        "content": ai_message
    })

    # Print response
    print(f"\nAI: {ai_message}")