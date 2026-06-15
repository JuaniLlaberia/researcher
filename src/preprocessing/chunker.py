import logging
from typing import List

from pydantic import BaseModel
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunk(BaseModel):
    """A single chunk handed from preprocessing to the vector store."""
    text: str
    section: str | None = None


# Markdown headers Docling emits, mapped to metadata keys (deepest wins for section).
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


class Chunker:
    """
    Section-aware, two-stage splitter:

    1. `MarkdownHeaderTextSplitter` segments Docling's Markdown by heading,
       attaching the header path as metadata (so chunks respect section bounds).
    2. `RecursiveCharacterTextSplitter` sub-splits each section to enforce
       `chunk_size`, carrying the section metadata down to every sub-chunk.

    Degrades gracefully: a PDF with no headings yields one section that stage 2
    still splits by size.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def _section_of(metadata: dict) -> str | None:
        """Deepest available header becomes the chunk's section label."""
        for key in ("h3", "h2", "h1"):
            if key in metadata:
                return metadata[key]
        return None

    def run(self, doc_content: str) -> List[Chunk]:
        """
        Split document Markdown into section-attributed chunks.

        Args:
            doc_content (str): Document content as Markdown (from Docling).
        Returns:
            List[Chunk]: Size-bounded chunks, each tagged with its section.
        """
        logging.info(f"Starting document splitter process ({len(doc_content)} characters)...")

        sections = self.header_splitter.split_text(doc_content)
        documents = self.text_splitter.split_documents(sections)

        chunks = [
            Chunk(text=doc.page_content, section=self._section_of(doc.metadata))
            for doc in documents
        ]
        logging.info(f"Document was split successfully into {len(chunks)} chunks")

        return chunks
