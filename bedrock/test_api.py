# test_api.py — Final Clean Version
import requests
import json

URL = "https://2cqc53tued.execute-api.eu-west-2.amazonaws.com/ask"

questions = [
    "What happens when login fails?",
    "What AWS services does DIBS use?",
    "What is the monthly cost?",
    "What are the 4 security layers?",
    "How do I onboard a new tenant?"
]

print("=" * 55)
print("  DIBS DevOps AI Copilot — LIVE API TEST")
print("  Built by Sourav Mohanty")
print("=" * 55)

for question in questions:
    print(f"\nQuestion: {question}")
    print("-" * 40)
    
    response = requests.post(
        URL,
        json={"question": question}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
    else:
        print(f"Error: {response.text}")
    
    print()