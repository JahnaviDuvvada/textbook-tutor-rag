# 📚 Personalized Textbook Tutor (RAG)

## 🚀 Overview
This project is an AI-powered Textbook Tutor that allows users to:
- Ask questions from a PDF
- Get context-based answers
- Generate quizzes
- Generate summaries

## 🧠 How it works
- Extracts text from PDF
- Splits into chunks
- Uses TF-IDF to retrieve relevant content
- Sends context to Groq LLM for answering

## 🛠 Tech Stack
- Python (Flask)
- Groq API (LLM)
- TF-IDF (RAG)
- HTML, CSS

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python app.py
