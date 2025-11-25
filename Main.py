import os
import re
import json
import numpy as np
import pickle
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


CACHE_DIR = "cache"
PDF_DIR = "pdfs"


# -----------------------------
# CLEAN PDF TEXT (remove noise)
# -----------------------------
def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", "", text)
    return text.strip()


# -----------------------------
# EXTRACT PDF TEXT
# -----------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += clean_text(content) + " "
    return text


# -----------------------------
# LOAD ALL PDFs
# -----------------------------
def load_pdfs():
    filenames = []
    texts = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            print("Reading:", filename)
            full_path = os.path.join(PDF_DIR, filename)
            text = extract_text_from_pdf(full_path)
            if text.strip():
                filenames.append(filename)
                texts.append(text)
    return filenames, texts


# -----------------------------
# SAVE CACHE
# -----------------------------
def save_cache(filenames, texts, vectorizer, tfidf_matrix, embeddings):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    with open(os.path.join(CACHE_DIR, "filenames.json"), "w") as f:
        json.dump(filenames, f)

    np.save(os.path.join(CACHE_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(CACHE_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(CACHE_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)


# -----------------------------
# LOAD CACHE
# -----------------------------
def load_cache():
    try:
        with open(os.path.join(CACHE_DIR, "filenames.json"), "r") as f:
            filenames = json.load(f)

        embeddings = np.load(os.path.join(CACHE_DIR, "embeddings.npy"))

        with open(os.path.join(CACHE_DIR, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        with open(os.path.join(CACHE_DIR, "tfidf.pkl"), "rb") as f:
            tfidf_matrix = pickle.load(f)

        return filenames, None, vectorizer, tfidf_matrix, embeddings

    except:
        return None, None, None, None, None


# -----------------------------
# CREATE NEW INDEX (when needed)
# -----------------------------
def build_indexes(texts):
    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    print("Building semantic AI index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    return model, vectorizer, tfidf_matrix, embeddings


# -----------------------------
# HYBRID SEARCH ENGINE
# -----------------------------
def hybrid_search(query, vectorizer, tfidf_matrix, model, embeddings, filenames, texts):
    tfidf_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]

    semantic_vec = model.encode([query], convert_to_numpy=True)
    semantic_scores = cosine_similarity(semantic_vec, embeddings)[0]

    tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)

    hybrid = 0.4 * tfidf_norm + 0.6 * semantic_norm

    ranked = hybrid.argsort()[::-1]

    return ranked, hybrid


# -----------------------------
# CREATE SMART SNIPPET
# -----------------------------
def make_snippet(full_text, query):
    text = full_text
    query_lower = query.lower()

    idx = text.lower().find(query_lower.split()[0])
    if idx == -1:
        return text[:300]

    start = max(0, idx - 150)
    end = min(len(text), idx + 150)

    return text[start:end]


# -----------------------------
# HIGHLIGHT KEYWORDS
# -----------------------------
def highlight(text, query):
    tokens = query.split()
    for t in tokens:
        text = re.sub(rf"({t})", r"[\1]", text, flags=re.IGNORECASE)
    return text


# -----------------------------
# MAIN
# -----------------------------
def main():

    cached = load_cache()

    if cached[0] is not None:
        print("Loaded indexes from cache!")
        filenames, _, vectorizer, tfidf_matrix, embeddings = cached
        model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        print("Cache not found — indexing PDFs...")
        filenames, texts = load_pdfs()
        model, vectorizer, tfidf_matrix, embeddings = build_indexes(texts)
        save_cache(filenames, texts, vectorizer, tfidf_matrix, embeddings)

    print("\nReady! Hybrid search engine loaded.")

    while True:
        query = input("\nSearch (or type 'reload' to update PDFs): ").strip()

        if query == "":
            break

        if query.lower() == "reload":
            print("Reloading PDFs & rebuilding index...")
            filenames, texts = load_pdfs()
            model, vectorizer, tfidf_matrix, embeddings = build_indexes(texts)
            save_cache(filenames, texts, vectorizer, tfidf_matrix, embeddings)
            print("Updated!")
            continue

        filenames2 = filenames
        texts2 = texts if "texts" in locals() else None

        ranked, scores = hybrid_search(query, vectorizer, tfidf_matrix, model, embeddings, filenames2, texts2)

        print("\nResults:")
        for idx in ranked[:3]:
            snippet = make_snippet(texts2[idx], query)
            snippet = highlight(snippet, query)
            print(f"\n📄 {filenames2[idx]}")
            print("Score:", round(float(scores[idx]), 3))
            print(snippet, "...\n")


if __name__ == "__main__":
    main()
