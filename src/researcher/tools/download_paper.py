import ssl
import urllib.request

import certifi

def download_paper(url: str) -> bytes:
    """
    Downloads a research paper PDF from a given URL into memory.

    Args:
        url: The direct HTTP/HTTPS URL to the PDF file (e.g., an arXiv PDF link).
    Returns:
        bytes: The raw binary data of the downloaded PDF file.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url=url, headers=headers)

    ctx = ssl.create_default_context(cafile=certifi.where())

    with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
        pdf_bytes = response.read()

    return pdf_bytes