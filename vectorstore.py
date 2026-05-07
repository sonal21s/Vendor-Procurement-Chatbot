import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "vendors"
EMBED_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.collection = self.client.create_collection(COLLECTION_NAME)

    def index(self, chunks: list[str]) -> None:
        if not chunks:
            return
        embeddings = self.encoder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"row_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    def query(self, question: str, n_results: int = 5) -> list[str]:
        total = self.collection.count()
        if total == 0:
            return []
        embedding = self.encoder.encode([question], show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(n_results, total),
        )
        return results["documents"][0]
