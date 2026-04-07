# app.py
# Flask + Groq API + RAG (Improved Production Version)

import os
import uuid
from groq import Groq
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from utils import extract_text_from_pdf, chunk_text, retrieve_relevant_chunks

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "textbook_tutor_secret_key_2024"

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory store (session-based)
store = {}
MAX_SESSIONS = 50  # prevent memory overflow


# ---------------- SESSION ---------------- #
def get_session_data():
    # 🧠 Memory cleanup
    if len(store) > MAX_SESSIONS:
        print("Clearing old sessions...")
        store.clear()

    sid = session.get("sid")
    if not sid or sid not in store:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        store[sid] = {"chunks": [], "history": []}
    return store[sid]


# ---------------- GROQ API ---------------- #
def ask_groq(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a textbook tutor.\n"
                        "- Answer ONLY from the provided context.\n"
                        "- Be clear and concise.\n"
                        "- If answer not found, say: "
                        "'I couldn't find this in the textbook.'"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Groq Error:", e)
        return "⚠️ Error: Unable to get response from AI."


# ---------------- ROUTES ---------------- #

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]

    if file.filename == "" or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    os.makedirs("uploads", exist_ok=True)

    # ✅ Unique filename (fix overwrite issue)
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    upload_path = os.path.join("uploads", unique_name)
    file.save(upload_path)

    raw_text = extract_text_from_pdf(upload_path)

    if not raw_text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    chunks = chunk_text(raw_text)

    data = get_session_data()
    data["chunks"] = chunks
    data["history"] = []

    print(f"PDF uploaded. Total chunks: {len(chunks)}")

    return jsonify({
        "message": "PDF uploaded successfully",
        "chunks": len(chunks)
    })


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    data = get_session_data()

    if not data["chunks"]:
        return jsonify({"error": "Please upload a PDF first"}), 400

    print(f"User Question: {user_message}")

    # ✅ Better chunk control
    relevant = retrieve_relevant_chunks(user_message, data["chunks"])
    context = "\n\n".join(chunk[:1000] for chunk in relevant[:3])

    print(f"Chunks Retrieved: {len(relevant)}")

    # Include recent chat history
    history_text = "\n".join(
        [f"{h['role']}: {h['text']}" for h in data["history"][-4:]]
    )

    prompt = f"""
Previous Conversation:
{history_text}

Context from Textbook:
{context}

Question:
{user_message}

Answer clearly based ONLY on the context:
"""

    answer = ask_groq(prompt)

    data["history"].append({"role": "user", "text": user_message})
    data["history"].append({"role": "bot", "text": answer})

    return jsonify({"answer": answer})


@app.route("/quiz", methods=["POST"])
def quiz():
    data = get_session_data()

    if not data["chunks"]:
        return jsonify({"error": "Please upload a PDF first"}), 400

    context = "\n\n".join(chunk[:1000] for chunk in data["chunks"][:5])

    prompt = f"""
Generate 5 multiple-choice questions from the following textbook content.

Format:
Q: ...
A) ...
B) ...
C) ...
D) ...
Answer: ...

Content:
{context}
"""

    return jsonify({"quiz": ask_groq(prompt)})


@app.route("/summary", methods=["POST"])
def summary():
    data = get_session_data()

    if not data["chunks"]:
        return jsonify({"error": "Please upload a PDF first"}), 400

    context = "\n\n".join(chunk[:1000] for chunk in data["chunks"][:5])

    prompt = f"""
Summarize the following content:

- Use bullet points
- Keep it simple
- Be concise

Content:
{context}

Summary:
"""

    return jsonify({"summary": ask_groq(prompt)})


@app.route("/history", methods=["GET"])
def history():
    data = get_session_data()
    return jsonify({"history": data["history"]})


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)