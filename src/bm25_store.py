import os
import re
import pickle
import threading
from rank_bm25 import BM25Plus
from src import config

class BM25Store:
    """Lexical keyword retrieval engine using BM25Plus.
    Maintains synchronized inverted token indices with document chunks and source filtering.
    BM25Plus prevents negative IDF on small document collections or common query tokens.
    """

    def __init__(self, index_path: str = config.BM25_INDEX_PATH):
        self.index_path = index_path
        self.corpus: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Plus | None = None
        self._lock = threading.Lock()
        self.load()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric and underscore terms, preserving technical identifiers."""
        if not text:
            return []
        # Extracts alphanumeric terms including underscores and hyphens (e.g. ERR_404_NOT_FOUND)
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        return [t.strip("-_") for t in tokens if t.strip("-_")]

    def add_documents(self, documents: list[dict]):
        """Add document chunks to BM25 index and update model.
        Each document dict should contain: {'text': str, 'source': str, 'page': int, optional 'chunk_id': str}
        """
        if not documents:
            return

        with self._lock:
            for idx, doc in enumerate(documents):
                doc_copy = dict(doc)
                if "chunk_id" not in doc_copy:
                    source = doc_copy.get("source", "unknown")
                    page = doc_copy.get("page", 1)
                    doc_copy["chunk_id"] = f"{source}_p{page}_c{len(self.corpus) + idx}"
                tokens = self._tokenize(doc_copy.get("text", ""))
                self.corpus.append(doc_copy)
                self.tokenized_corpus.append(tokens)

            if self.tokenized_corpus:
                self.bm25 = BM25Plus(self.tokenized_corpus)
            self.persist()

    def search(
        self,
        query: str,
        n_results: int = config.BM25_TOP_K,
        source_filter: list[str] = None
    ) -> list[dict]:
        """Retrieve top matching documents for query via BM25 scores with optional source filtering."""
        with self._lock:
            if not self.bm25 or not self.corpus:
                return []

            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            scores = self.bm25.get_scores(query_tokens)
            
            scored_docs = []
            for i, score in enumerate(scores):
                if score <= 0.0:
                    continue
                doc = self.corpus[i]
                if source_filter is not None and doc.get("source") not in source_filter:
                    continue
                res = dict(doc)
                res["score"] = float(score)
                scored_docs.append(res)

            # Sort descending by BM25 score
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:n_results]

    def delete_source(self, source: str):
        """Remove all chunks associated with a source document and rebuild index."""
        with self._lock:
            new_corpus = []
            new_tokenized = []
            for doc, tokens in zip(self.corpus, self.tokenized_corpus):
                if doc.get("source") != source:
                    new_corpus.append(doc)
                    new_tokenized.append(tokens)

            self.corpus = new_corpus
            self.tokenized_corpus = new_tokenized

            if self.tokenized_corpus:
                self.bm25 = BM25Plus(self.tokenized_corpus)
            else:
                self.bm25 = None

            self.persist()

    def persist(self):
        """Atomically persist corpus and tokenized corpus to disk."""
        if not self.index_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        temp_path = f"{self.index_path}.tmp"
        payload = {
            "corpus": self.corpus,
            "tokenized_corpus": self.tokenized_corpus
        }
        with open(temp_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(temp_path, self.index_path)

    def load(self):
        """Load index from disk if present."""
        if not self.index_path or not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "rb") as f:
                payload = pickle.load(f)
            self.corpus = payload.get("corpus", [])
            self.tokenized_corpus = payload.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Plus(self.tokenized_corpus)
        except Exception:
            self.corpus = []
            self.tokenized_corpus = []
            self.bm25 = None

    def clear(self):
        """Reset index completely."""
        with self._lock:
            self.corpus = []
            self.tokenized_corpus = []
            self.bm25 = None
            if self.index_path and os.path.exists(self.index_path):
                try:
                    os.remove(self.index_path)
                except OSError:
                    pass
