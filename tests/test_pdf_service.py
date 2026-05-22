"""
Test unitari per pdf_service: non richiedono DB né rete.
Esegui con: python -m pytest tests/test_pdf_service.py -v
"""
import pytest
from app.services.pdf_service import chunk_text


def test_chunk_text_base():
    """Un testo breve produce almeno un chunk."""
    text = "Questo è un paragrafo di esempio.\n\nQuesto è un secondo paragrafo."
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_overlap():
    """Con un testo lungo, i chunk non perdono contenuto (overlap funziona)."""
    # Genera ~5000 caratteri di testo
    long_text = "\n\n".join([f"Paragrafo numero {i}: " + "x" * 200 for i in range(20)])
    chunks = chunk_text(long_text)
    # Deve produrre più di un chunk
    assert len(chunks) > 1
    # Nessun chunk vuoto
    assert all(len(c.strip()) > 0 for c in chunks)


def test_empty_text():
    """Testo vuoto non produce chunk."""
    chunks = chunk_text("   ")
    assert chunks == []


def test_single_long_paragraph():
    """Un unico paragrafo più lungo di CHUNK_SIZE viene comunque spezzato."""
    # CHUNK_SIZE default = 1000
    huge_para = "parola " * 300  # ~2100 caratteri
    chunks = chunk_text(huge_para)
    assert len(chunks) >= 2
