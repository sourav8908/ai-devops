# 🤖 AI DevOps Journey
### From Zero to Cloud AI Engineer
**Built by Sourav Mohanty — M.Tech, IIT Patna**

---

## 🚀 What I Built

Starting from complete zero AI knowledge,
I built a production-grade DevOps AI Copilot
deployed on AWS cloud in under 2 weeks.

**Live :**
http://dibs-copilot-ui.s3-website.eu-west-2.amazonaws.com

---

## 📁 Projects

### Project 1 — Basic Chatbot
Simple Claude AI chatbot using Anthropic API.
`chatbot.py`

### Project 2 — Memory Chatbot  
Persistent memory chatbot saving conversations
to JSON across sessions.
`chatbot_memory.py`

### Project 3 — RAG Pure Python
RAG system built from scratch using ChromaDB
and sentence transformers.
`rag_chatbot.py`

### Project 4 — RAG LangChain
Same RAG system rebuilt using LangChain
framework — 10 lines vs 100 lines.
`rag_langchain.py`

### Project 5 — DevOps AI Copilot (Local)
Production-grade AI Copilot answering
questions from engineering documentation
with source citations.
`devops_copilot.py`

### Project 6 — Cloud AI Copilot 🏆
Full production deployment on AWS:
- AWS Bedrock (Claude + Titan)
- Pinecone Vector Database
- AWS Lambda + API Gateway
- Live REST API endpoint
`lambda_function.py`

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| AI Models | Claude 3 Haiku, Amazon Titan |
| Cloud | AWS Bedrock, Lambda, API Gateway |
| Vector DB | Pinecone, ChromaDB |
| Framework | LangChain, boto3 |
| Language | Python |
| Background | AWS, Docker, Kubernetes |

---

## 🏗️ Architecture
```
User Question
     ↓
API Gateway (AWS eu-west-2)
     ↓
Lambda Function
     ↓
Bedrock Titan → 1024d vector
     ↓
Pinecone → relevant chunks
     ↓
Bedrock Claude → answer
     ↓
Response to user ✅
```

---

## 📚 What I Learned

- LLMs, RAG, Vector Databases
- AWS Bedrock integration
- Serverless AI deployment
- Production API development
- Prompt Engineering

---

## 👨‍💻 About Me

- M.Tech Cloud Computing — IIT Patna
- Background: AWS, Docker, Kubernetes
- ISRO Internship experience
- Currently building: MLOps skills

📧 sourav_24a07res194@iitp.ac.in
🔗 github.com/sourav8908
