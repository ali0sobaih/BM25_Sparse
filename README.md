Here i created some simple projects with basic search engines, RAG, datasets normalization and some other lovely topics!

#  Local BM25 Lexical Search Engine

A lightweight, high-performance local Search and Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **Rank-BM25**, and a custom **5-step NLTK NLP pipeline**, paired with a modern, responsive **Tailwind CSS** frontend interface.

---

##  Tech Stack & Architecture

* **Backend Framework:** FastAPI (Python)
* **Search Algorithm:** BM25Okapi (`rank-bm25`)
* **NLP Preprocessing:** NLTK (Stop-words removal, WordNet Lemmatization, Tokenization, Regex Noise Stripping)
* **Caching Layer:** Python `pickle` with automated timestamp validation (`os.path.getmtime`) to skip redundant tokenization on startup.
* **Frontend:** HTML5, Tailwind CSS, JavaScript (`fetch` API) with Dark/Light mode support.

---

##  5-Step NLP Preprocessing Pipeline

To ensure clean keyword matching, incoming queries and dataset passages undergo a rigorous 5-step normalization pipeline:
1. **Lowercasing:** Standardizes all alphabetical characters.
2. **Noise & Punctuation Stripping:** Removes all non-alphanumeric symbols via regex (`[^a-z0-9\s]`).
3. **Tokenization:** Splits raw strings into individual word lists (`text.split()`).
4. **Stop-word Filtering:** Removes common English noise words using NLTK's corpus.
5. **Lemmatization:** Extracts root word forms using WordNet (e.g., *"arteries"* $\rightarrow$ *"artery"*).

---

##  Getting Started Locally

### Prerequisites
* Python 3.10+
* Git


### 1. Clone the Repository
```bash
git clone [https://github.com/ali0sobaih/BM25_Sparse.git](https://github.com/ali0sobaih/BM25_Sparse.git)
cd BM25_Sparse
```

### 2.FOR INSTALLING THE REQUIREMENTS
* pip install -r requirements.txt 