from typing import Any, Dict

from src.researcher.tools.download_paper import download_paper
from src.preprocessing.docling import DocumentProcessor
from src.preprocessing.chunker import Chunker
from src.db.vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

class IngestionPipeline:
    """
    End-to-end paper ingestion: download -> Docling parse -> chunk -> embed + store.
    """
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.processor = DocumentProcessor()
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.store = VectorStore()

    def run(self, url: str, paper_meta: Dict[str, Any]) -> str:
        """
        Ingest a single paper from a PDF URL.

        Args:
            url (str): Direct URL to the paper PDF.
            paper_meta (Dict[str, Any]): Paper metadata (title, source, url, etc).
        Returns:
            str: The stored paper's ID.
        """
        pdf_bytes = download_paper(url)
        markdown = self.processor.process_document(pdf_bytes)
        chunks = self.chunker.run(markdown)

        return self.store.ingest_documents(paper_meta=paper_meta, chunks=chunks)
