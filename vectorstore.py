import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"


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
        # Normalize once so cosine similarity = dot product at query time
        self.embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def query(self, question: str, n_results: int = 5) -> list[str]:
        if self.embeddings is None or not self.documents:
            return []
        q = self.encoder.encode([question], show_progress_bar=False)
        q = q / np.linalg.norm(q)
        scores = (self.embeddings @ q.T).flatten()
        top_k = min(n_results, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]
