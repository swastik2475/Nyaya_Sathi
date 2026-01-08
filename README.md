NyayaSathi – AI Legal Query Assistant
NyayaSathi is an AI-powered Retrieval-Augmented Generation (RAG) system designed to make legal information easier to access and understand. Instead of manually searching large legal documents, NyayaSathi retrieves the most relevant sections from law PDFs and generates clear, helpful responses based on user queries.
This project aims to assist students, citizens, and professionals by providing instant access to legal references, simplified explanations, and context-aware answers.

Project Overview
NyayaSathi uses a combination of text chunking, embeddings, and a FAISS vector store to search through legal documents. When a user asks a question, the system retrieves the most relevant context and passes it to an LLM, which forms a natural-language response.
The system is designed to run locally, making it private and accessible even on lower-end hardware.
Key Features
1. Smart Legal Search
Extracts text from large PDF law documents
Splits text into manageable chunks
Creates embeddings through the all-MiniLM-L6-v2 model
Stores and retrieves chunks using FAISS
2. AI-Generated Answers
Combines retrieved context with a lightweight LLM
Generates user-friendly legal explanations
Reduces the need for manual navigation through complex documents
3. Client-Server Architecture
Backend built with FastAPI
Streamlit frontend for an easy-to-use interface
Both modules communicate through API endpoints
4. Voice Support
Optional speech-to-text input
Optional text-to-speech output
Helpful for users who prefer listening or cannot type
5. Privacy and Local Execution
Runs entirely on the user’s local machine
No external API calls
Suitable for scenarios requiring confidentiality
Tech Stack
Python
FastAPI
Streamlit
FAISS
Sentence Transformers (all-MiniLM-L6-v2)
Local LLM models such as Gemma, Llama, or MiniLM
SpeechRecognition and pyttsx3 (for audio features)

Folder Structure
nyayasathi/
│── data/                     # Raw law PDFs
│── processed/                # Cleaned and chunked text
│── faiss_index/              # index.faiss and index.pkl
│── app/
│   ├── rag.py                # Core RAG pipeline
│   ├── nyaya.py                # FastAPI backend
│   ├── app.py             # Streamlit frontend
│── README.md


How It Works
Load PDF documents
Extract and clean the text
Break text into chunks
Generate embeddings for each chunk
Store embeddings in a FAISS index
User enters a query
System retrieves the most relevant chunks
LLM generates the final answer using the retrieved context

Use Cases
Law students preparing for exams or research
Citizens looking for general legal information
NGOs guiding users on legal matters
Professionals validating specific sections
Anyone needing quick references from legal documents

Disclaimer
NyayaSathi provides general legal information.
It is not intended to replace professional legal advice from a certified lawyer.
