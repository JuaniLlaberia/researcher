from typing import Any, Dict, List

from sqlalchemy import select

from src.core.config import settings
from src.core.embedder import Embedder, EmbedderConfig
from src.db.session import SessionLocal
from src.db.models.paper import Paper, PaperChunk
from src.preprocessing.chunker import Chunk

class VectorStore:
    """
    Persistence + similarity search for papers and their chunk embeddings.

    Attributes:
        embedder (Embedder): Embedder model instance.
    """
    def __init__(self):
        """
        Initializes the VectorStore class.
        """
        if not settings.embedder_provider or not settings.embedder_model:
            raise ValueError("Missing `EMBEDDER_PROVIDER` or `EMBEDDER_MODEL` env values.")

        self.embedder = Embedder(config=EmbedderConfig(
            provider=settings.embedder_provider,
            model=settings.embedder_model,
        ))

    def ingest_documents(self, paper_meta: Dict[str, Any], chunks: List[Chunk]) -> str:
        """
        Creates the paper and its chunk rows (with embeddings) in the database.

        Args:
            paper_meta (Dict[str, Any]): Paper metadata (title, source, url, etc).
            chunks (List[Chunk]): Section-attributed chunks from the chunker.
        Returns:
            str: Generated paper ID.
        """
        embeddings = self.embedder.embed_documents(texts=[c.text for c in chunks])

        for emb in embeddings:
            if len(emb) != settings.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {len(emb)} does not match configured "
                    f"EMBEDDING_DIM={settings.embedding_dim}. Check EMBEDDER_MODEL "
                    f"and the Vector(N) column on PaperChunk."
                )

        with SessionLocal.begin() as session:
            paper = Paper(**paper_meta)
            session.add(paper)
            session.flush()  # Generates the ID

            paper_chunks = [
                PaperChunk(
                    paper_id=paper.id,
                    chunk_index=i,
                    content=chunk.text,
                    section=chunk.section,
                    embedding=emb,
                )
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]
            session.add_all(paper_chunks)

            return str(paper.id)

    def retrieve_documents(self, query: str, k: int = 5):
        """
        Retrieves the K most similar chunks for the provided query.

        Args:
            query (str): Search query.
            k (int): Number of results to retrieve.
        Returns:
            list[tuple[PaperChunk, float]]: (chunk, similarity) rows, most similar first.
        """
        query_embedding = self.embedder.embed_query(text=query)

        with SessionLocal() as session:
            similarity_score = 1 - PaperChunk.embedding.cosine_distance(query_embedding)
            results = (
                select(PaperChunk, similarity_score.label("similarity"))
                .order_by(similarity_score.desc())  # most similar first
                .limit(k)
            )

            return session.execute(results).all()
