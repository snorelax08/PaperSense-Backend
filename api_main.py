import os
import re
import numpy as np
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PDF_DIR = "pdfs"

filenames: List[str] = []
texts: List[str] = []
vectorizer = None
tfidf_matrix = None
embeddings = None
model = None

app = FastAPI(title="PaperSense API")


# --- CORS so React can call this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Serve PDFs as static files ---
if os.path.exists(PDF_DIR):
    app.mount("/pdfs", StaticFiles(directory=PDF_DIR), name="pdfs")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return clean_text(text)


def load_pdfs():
    global filenames, texts

    if not os.path.exists(PDF_DIR):
        raise RuntimeError(f"PDF folder '{PDF_DIR}' not found")

    filenames = []
    texts = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(PDF_DIR, filename)
            print("Reading:", filename)
            text = extract_text_from_pdf(full_path)
            if text.strip():
                filenames.append(filename)
                texts.append(text)

    if not filenames:
        raise RuntimeError("No readable PDFs found in 'pdfs' folder")


def build_indexes():
    global vectorizer, tfidf_matrix, embeddings, model, texts

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Creating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def hybrid_search(query: str, top_k: int = 5):
    if vectorizer is None or tfidf_matrix is None or embeddings is None or model is None:
        raise RuntimeError("Search engine not initialized")

    tfidf_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]

    semantic_vec = model.encode([query], convert_to_numpy=True)
    semantic_scores = cosine_similarity(semantic_vec, embeddings)[0]

    tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)

    hybrid = 0.4 * tfidf_norm + 0.6 * semantic_norm

    ranked = hybrid.argsort()[::-1]

    return ranked[:top_k], hybrid


def make_snippet(full_text: str, query: str, window: int = 250) -> str:
    text = full_text
    text_lower = text.lower()
    tokens = [t for t in query.lower().split() if len(t) > 2]

    best_pos = None
    for tok in tokens:
        idx = text_lower.find(tok)
        if idx != -1:
            best_pos = idx
            break

    if best_pos is None:
        start = 0
    else:
        start = max(0, best_pos - window // 2)

    end = min(len(text), start + window)
    return text[start:end]


def highlight(snippet: str, query: str) -> str:
    highlighted = snippet
    for tok in query.split():
        if len(tok) < 3:
            continue
        pattern = re.compile(re.escape(tok), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: "[" + m.group(0) + "]", highlighted)
    return highlighted


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    filename: str
    score: float
    snippet: str
    url: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.on_event("startup")
def startup_event():
    print("Initializing PaperSense engine...")
    load_pdfs()
    build_indexes()
    print("PaperSense API ready.")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    ranked_indices, scores = hybrid_search(req.query)

    results: List[SearchResult] = []
    for idx in ranked_indices:
        snippet_raw = make_snippet(texts[idx], req.query)
        snippet_hl = highlight(snippet_raw, req.query)
        results.append(
            SearchResult(
                filename=filenames[idx],
                score=float(round(float(scores[idx]), 4)),
                snippet=snippet_hl,
                url=f"/pdfs/{filenames[idx]}",
            )
        )

    return SearchResponse(results=results)


@app.post("/reload")
def reload_pdfs():
    load_pdfs()
    build_indexes()
    return {"status": "reloaded", "documents": len(filenames)}
