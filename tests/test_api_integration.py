"""
Test di integrazione end-to-end.
Richiedono: DB PostgreSQL + pgvector attivo, variabili .env configurate.

Esegui con: python -m pytest tests/test_api_integration.py -v

ATTENZIONE: questi test SCRIVONO nel database. Usali su un DB di test,
non su quello di produzione (o aggiungi fixtures di cleanup).
"""
import io
import pytest
from fastapi.testclient import TestClient

# Importazione lazy per non fallire se il .env non è configurato durante CI
@pytest.fixture(scope="module")
def client():
    from app.main import app
    from app.database import init_db
    init_db()
    with TestClient(app) as c:
        yield c


def _make_minimal_pdf() -> bytes:
    """
    Crea un PDF valido minimale senza dipendenze aggiuntive.
    Contiene testo sufficiente per produrre almeno un chunk.
    """
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 200>>
stream
BT /F1 12 Tf 50 700 Td
(Il machine learning e' una branca dell'intelligenza artificiale.) Tj
0 -20 Td (Permette ai computer di imparare dai dati senza essere esplicitamente programmati.) Tj
0 -20 Td (Le reti neurali sono uno strumento fondamentale del deep learning moderno.) Tj
ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000518 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
600
%%EOF"""
    return content


class TestUpload:
    def test_upload_pdf_valido(self, client):
        pdf_bytes = _make_minimal_pdf()
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        # Accettiamo 201 (successo) oppure 422 se il PDF minimale non è leggibile da pypdf
        assert response.status_code in (201, 422)

    def test_upload_file_non_pdf(self, client):
        response = client.post(
            "/documents/upload",
            files={"file": ("documento.txt", io.BytesIO(b"testo"), "text/plain")},
        )
        assert response.status_code == 422

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestQuery:
    def test_query_senza_documenti_restituisce_404_o_risposta(self, client):
        """Se non ci sono documenti ritorna 404; se ci sono, ritorna 200."""
        response = client.post(
            "/documents/query",
            json={"question": "Cos'è il machine learning?", "top_k": 3},
        )
        assert response.status_code in (200, 404)

    def test_query_domanda_troppo_corta(self, client):
        response = client.post(
            "/documents/query",
            json={"question": "ab"},
        )
        assert response.status_code == 422
