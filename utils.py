# utils.py
# Contains: PDF text extraction + text chunking + TF-IDF embeddings for RAG
# Replaces: pdf_parser.py + chunk/retrieval logic + embeddings
# Uses scikit-learn TF-IDF instead of heavy sentence-transformers

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ── Extract text from PDF page by page ────────────────────────────
# Memory-efficient: reads one page at a time (important for 5.7GB RAM)
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            page_text = page.get_text()
            if not page_text.strip():
                print(f"⚠️ Empty text on page {i}")
            else:
                text += page_text
    print(f"✅ Total extracted characters: {len(text)}")
    return text


# ── Split text into chunks for RAG ────────────────────────────────
# Each chunk = ~500 words with 50-word overlap for context continuity
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


# ── Retrieve relevant chunks using TF-IDF embeddings ──────────────
# Much better than keyword matching - understands semantic similarity
# Uses cosine similarity between query vector and chunk vectors
def retrieve_relevant_chunks(query, chunks, top_k=3):
    if not chunks:
        return []

    try:
        # Combine query + all chunks for vectorization
        all_texts = [query] + chunks

        # Build TF-IDF matrix
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Query vector is the first row
        query_vector = tfidf_matrix[0]

        # Chunk vectors are the rest
        chunk_vectors = tfidf_matrix[1:]

        # Calculate cosine similarity between query and each chunk
        similarities = cosine_similarity(query_vector, chunk_vectors).flatten()

        # Get top_k most similar chunk indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant = [chunks[i] for i in top_indices]
        print(f"✅ Retrieved {len(relevant)} relevant chunks using TF-IDF")
        return relevant

    except Exception as e:
        print(f"⚠️ TF-IDF error: {e}, falling back to keyword matching")
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]