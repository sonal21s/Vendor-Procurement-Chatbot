import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

STOPWORDS = {
    "how", "many", "what", "which", "who", "are", "is", "the", "in", "of",
    "for", "and", "or", "with", "have", "has", "do", "does", "tell", "me",
    "give", "list", "show", "all", "any", "their", "its", "from", "that",
    "this", "these", "those", "a", "an", "to", "by", "be", "been",
}


class VectorStore:
    def __init__(self):
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.embeddings: np.ndarray | None = None
        self.documents: list[str] = []

    def index(self, chunks: list[str]) -> None:
        if not chunks:
            return
        self.documents = chunks
        emb = self.encoder.encode(chunks, show_progress_bar=False)
        self.embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def query(self, question: str, n_results: int = 10) -> list[str]:
        if self.embeddings is None or not self.documents:
            return []

        # Keyword search — finds ALL docs containing meaningful terms from the query
        keywords = [
            w.lower().strip("?.,!") for w in question.split()
            if len(w) > 2 and w.lower().strip("?.,!") not in STOPWORDS
        ]
        keyword_indices: set[int] = set()
        if keywords:
            for i, doc in enumerate(self.documents):
                doc_lower = doc.lower()
                if any(kw in doc_lower for kw in keywords):
                    keyword_indices.add(i)

        # Vector search — semantic similarity top-N
        q = self.encoder.encode([question], show_progress_bar=False)
        q = q / np.linalg.norm(q)
        scores = (self.embeddings @ q.T).flatten()
        top_k = min(n_results, len(self.documents))
        vector_indices = list(np.argsort(scores)[::-1][:top_k])

        # If keyword search found matches, return only those (precise, no noise).
        # Fall back to vector search only when keywords find nothing.
        if keyword_indices:
            return [self.documents[i] for i in keyword_indices]
        return [self.documents[i] for i in vector_indices]
