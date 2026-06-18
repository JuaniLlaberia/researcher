from typing import Any, Dict, List, Tuple

import torch
from sqlalchemy import func, select
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import settings
from src.core.embedder import Embedder, EmbedderConfig
from src.db.session import SessionLocal
from src.db.models.paper import Paper, PaperChunk
from src.preprocessing.chunker import Chunk

def select_device() -> str:
    """
    Pick the torch device.

    Returns:
        str: Name of device to use.
    """
    if settings.reranker_device:
        return settings.reranker_device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class VectorStore:
    """
    Persistence + hybrid retrieval for papers and their chunk embeddings pipeline.
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

        self._tokenizer = None
        self._reranker = None
        self._device = None

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

    def paper_exists(self, source: str, external_id: str) -> bool:
        """
        True if a paper is already stored. Used by the searcher to skip re-downloading papers already ingested.

        Args:
            source (str): Paper source.
            external_id (str): External identifier (example, form arxiv).
        Returns:
            bool: Whether the paper exists or not.
        """
        with SessionLocal() as session:
            stmt = (
                select(Paper.id)
                .where(Paper.source == source, Paper.external_id == external_id)
                .limit(1)
            )
            return session.execute(stmt).first() is not None

    @staticmethod
    def _rows_to_maps(rows) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]:
        """
        (chunk, title, url, score) rows -> (ids, id->text, id->metadata).
        """
        ids: List[str] = []
        id_to_text: Dict[str, str] = {}
        id_to_meta: Dict[str, Dict[str, Any]] = {}
        for chunk, title, url, _score in rows:
            cid = str(chunk.id)
            ids.append(cid)
            id_to_text[cid] = chunk.content
            id_to_meta[cid] = {
                "paper_id": str(chunk.paper_id),
                "title": title,
                "url": url,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
            }
        return ids, id_to_text, id_to_meta

    def _semantic_query(self, query: str, k: int = 25):
        """
        Searches by cosine similarity.
        
        Args:
            query (str): Search query.
            k (int): Amount of results to retrieve. Default = 25.
        """
        query_embedding = self.embedder.embed_query(text=query)

        with SessionLocal() as session:
            similarity = (1 - PaperChunk.embedding.cosine_distance(query_embedding)).label("score")
            stmt = (
                select(PaperChunk, Paper.title, Paper.url, similarity)
                .join(Paper, PaperChunk.paper_id == Paper.id)
                .order_by(similarity.desc())  # most similar first
                .limit(k)
            )
            rows = session.execute(stmt).all()

        return self._rows_to_maps(rows)

    def _keyword_query(self, query: str, k: int = 25):
        """
        Searches by keyword by Postgres full-text.
        
        Args:
            query (str): Search query.
            k (int): Amount of results to retrieve. Default = 25.
        """
        with SessionLocal() as session:
            ts_query = func.plainto_tsquery("english", query)
            rank = func.ts_rank(PaperChunk.content_tsv, ts_query).label("score")
            stmt = (
                select(PaperChunk, Paper.title, Paper.url, rank)
                .join(Paper, PaperChunk.paper_id == Paper.id)
                .where(PaperChunk.content_tsv.op("@@")(ts_query))
                .order_by(rank.desc())
                .limit(k)
            )
            rows = session.execute(stmt).all()

        return self._rows_to_maps(rows)

    def _rrf(self, rankings: List[List[str]], k: int = 10) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked id lists with Reciprocal Rank Fusion.

        Args:
            rankings (List[List[str]]): One ranked list of doc ids per arm.
            k (int): RRF dampening constant.
        Returns:
            List[Tuple[str, float]]: (doc_id, fused_score) sorted best first.
        """
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _load_reranker(self):
        """
        Lazily load the cross-encoder onto the chosen device.
        """
        if self._reranker is None:
            self._device = select_device()
            self._tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model)
            self._reranker = (
                AutoModelForSequenceClassification
                .from_pretrained(settings.reranker_model)
                .to(self._device)
            )
            self._reranker.eval()
        return self._reranker

    def _rerank(self, pairs: List[List[str]]) -> List[float]:
        """
        Cross-encoder relevance scores for [query, passage] pairs.
        """
        if not pairs:
            return []

        model = self._load_reranker()
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self._device)
            scores = model(**inputs, return_dict=True).logits.view(-1).float()

        return scores.cpu().tolist()

    def retrieve_documents(
        self,
        main_query: str,
        queries: List[str],
        candidates: int = 20,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieve: multi-query semantic + keyword, fused with RRF and re-ranked against the main query.

        Args:
            main_query (str): The user's actual question (used for keyword + rerank).
            queries (List[str]): Semantic query variants (e.g. LLM-expanded).
            candidates (int): How many fused candidates to send to the reranker.
            top_k (int): How many final results to return.
        Returns:
            List[Dict[str, Any]]: {id, text, metadata, score} sorted best first.
        """
        rankings: List[List[str]] = []
        doc_text: Dict[str, str] = {}
        doc_meta: Dict[str, Dict[str, Any]] = {}

        # Semantic
        for q in queries:
            ids, id_to_text, id_to_meta = self._semantic_query(query=q)
            rankings.append(ids)
            doc_text.update(id_to_text)
            doc_meta.update(id_to_meta)

        # Keyword
        kids, ktext, kmeta = self._keyword_query(query=main_query)
        rankings.append(kids)
        doc_text.update(ktext)
        doc_meta.update(kmeta)

        fused = self._rrf(rankings)
        candidate_ids = [doc_id for doc_id, _ in fused[:candidates]]

        # Rerank the fused candidates against the main query.
        pairs = [[main_query, doc_text[doc_id]] for doc_id in candidate_ids]
        scores = self._rerank(pairs)
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)

        return [
            {
                "id": doc_id,
                "text": doc_text[doc_id],
                "metadata": doc_meta[doc_id],
                "score": float(score),
            }
            for doc_id, score in reranked[:top_k]
            if score > 0
        ]
