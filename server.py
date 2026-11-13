import os
import re
import pickle
import time
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Initialize NLTK Resources
# ---------------------------------------------------------------------------
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> list[str]:
    """
    5-Step NLP Preprocessing Pipeline:
    1. Lowercasing
    2. Punctuation & Noise Removal
    3. Tokenization
    4. Stop-words Removal
    5. Lemmatization (Root Extraction)
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # note: any thing that is not a lowercase letter, number, or whitespace (such as punctuation marks like ., ,, !, ?, -, and special symbols like @, #, $, %). /////////// '': The replacement string. ///////////// text: The input string being processed. /////////////// re is a library in python for regular expression
    tokens = text.split() # split the string to tokens (words). example: The Sun is shining => 1: sun , 2: shining 
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ] 
    return cleaned_tokens # examples: dogs => dog , churches => church

# ---------------------------------------------------------------------------
# 2. FastAPI Application Setup & CORS Configuration
# ---------------------------------------------------------------------------
app = FastAPI(title="Cached Local BM25 Textbook Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for dataset paragraphs, tokenized corpus, and BM25 index
raw_passages = []
tokenized_corpus = []
bm25_index = None

CACHE_FILE = "bm25_cache.pkl"
DATASET_FILE = "medical_dataset.txt"

# ---------------------------------------------------------------------------
# 3. Load Dataset & Use Cached BM25 Index (Persistent Storage)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_dataset_and_cache():
    global raw_passages, tokenized_corpus, bm25_index

    if not os.path.exists(DATASET_FILE):
        print(f"❌ Error: Could not find '{DATASET_FILE}' in local directory!")
        return

    dataset_mtime = os.path.getmtime(DATASET_FILE)
    cache_valid = False

    if os.path.exists(CACHE_FILE):
        cache_mtime = os.path.getmtime(CACHE_FILE)
        if cache_mtime >= dataset_mtime:
            cache_valid = True

    if cache_valid:
        print(f"Loading preprocessed BM25 index from cache ('{CACHE_FILE}')...")
        with open(CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
            raw_passages = cached_data["raw_passages"]
            tokenized_corpus = cached_data["tokenized_corpus"]
            bm25_index = cached_data["bm25_index"]
        print(f"Successfully loaded {len(raw_passages)} records from cache instantly!")
    else:
        print(f"Dataset updated or cache missing. Processing '{DATASET_FILE}'...")
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        raw_passages = [p.strip() for p in content.split("\n") if p.strip()]

        print(f"Running 5-step NLTK pipeline on {len(raw_passages)} passages...")
        tokenized_corpus = [preprocess_text(p) for p in raw_passages]

        print("Building and caching BM25 Index...")
        bm25_index = BM25Okapi(tokenized_corpus)

        # Save everything to cache file
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "raw_passages": raw_passages,
                "tokenized_corpus": tokenized_corpus,
                "bm25_index": bm25_index
            }, f)
        print(f"✅ Cache saved successfully to '{CACHE_FILE}'!")

# ---------------------------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None

    def get_text(self) -> str:
        return (self.message or self.query or "").strip()

@app.get("/")
def home():
    return {"status": "online", "indexed_passages": len(raw_passages), "cached": True}

@app.post("/search")
@app.post("/chat")
def search_textbook(req: QueryRequest):
    if not bm25_index:
        raise HTTPException(status_code=500, detail="BM25 Index is not initialized.")
    
    start_time = time.time()
    query_text = req.get_text()
    
    if not query_text:
        return {
            "query_tokens": [],
            "total_matches": 0,
            "latency_ms": 0,
            "results": []
        }

    # Preprocess incoming query using the exact same 5-step pipeline
    query_tokens = preprocess_text(query_text)
    
    if not query_tokens:
        return {
            "query_tokens": [],
            "total_matches": 0,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "results": []
        }

    # Score passages with BM25
    scores = bm25_index.get_scores(query_tokens)
    
    # Get top 3 matching results
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    
    results = []
    total_matches = 0
    
    for rank, idx in enumerate(ranked_indices, start=1):
        score = float(scores[idx])
        if score > 0.0:
            total_matches += 1
            passage_tokens = tokenized_corpus[idx]
            matched_tokens = list(set([t for t in query_tokens if t in passage_tokens]))
            
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "content": raw_passages[idx],
                "matched_tokens": matched_tokens
            })

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query_tokens": query_tokens,
        "total_matches": total_matches,
        "latency_ms": latency_ms,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)