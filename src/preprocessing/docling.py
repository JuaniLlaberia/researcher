import io

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream

class DocumentProcessor:
    """
    Wraps Docling to turn an in-memory PDF (raw bytes) into structured Markdown,
    preserving sections, tables and figures. No file on disk is required.
    """
    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def process_document(self, pdf_bytes: bytes, name: str = "paper.pdf") -> str:
        """
        Parse a PDF given as raw bytes and export it to Markdown.

        Args:
            pdf_bytes (bytes): The raw PDF content (e.g. from `download_paper`).
            name (str): A label used only for format detection; must end in `.pdf`.
                        It is not a filesystem path.
        Returns:
            str: The document rendered as Markdown.
        """
        source = DocumentStream(name=name, stream=io.BytesIO(pdf_bytes))
        result = self.converter.convert(source)
        return result.document.export_to_markdown()
