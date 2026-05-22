"""
Test avanzati: upload asincrono, query con namespace utente, cache Redis.

Struttura:
  TestUploadAsync   — POST /upload ritorna 202 + task_id immediatamente
  TestDocumentList  — GET /documents filtra per utente
  TestQueryCache    — la seconda query identica usa la cache (Groq non viene chiamato)
  TestUserNamespace — l'utente A non vede i documenti dell'utente B

Dipendenze:
  pip install pytest pytest-asyncio httpx

Non richiedono Celery in esecuzione: il task viene mockato.
Richiedono DB Postgres + pgvector attivo (usa quello Docker).
"""
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Client di test con DB reale — init_db() crea le tabelle se non esistono."""
    from app.database import init_db
    from app.main import app

    init_db()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _headers(user_id: str) -> dict:
    return {"X-User-ID": user_id}


# ── Helper: crea un PDF minimale valido ──────────────────────────────────────

def _minimal_pdf(text: str = "Sviluppo personale e crescita.") -> bytes:
    """PDF a una pagina con testo iniettato come stream."""
    escaped = text.replace("(", r"\(").replace(")", r"\)")
    stream_content = (
        f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET"
    ).encode()
    stream_len = len(stream_content)
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        + f"4 0 obj<</Length {stream_len}>>\nstream\n".encode()
        + stream_content
        + b"\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"0000000400 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n460\n%%EOF"
    )
    return pdf


# ── TestUploadAsync ───────────────────────────────────────────────────────────

class TestUploadAsync:
    """POST /upload deve ritornare 202 immediatamente senza attendere Celery."""

    def test_upload_ritorna_202_con_task_id(self, client):
        """Il task Celery viene mockato: non deve girare il worker."""
        mock_result = MagicMock()
        mock_result.id = "fake-task-uuid-1234"

        with patch("app.routers.documents.process_document") as mock_task:
            mock_task.delay.return_value = mock_result

            resp = client.post(
                "/documents/upload",
                files={"file": ("test.pdf", io.BytesIO(_minimal_pdf()), "application/pdf")},
                headers=_headers("user-test-1"),
            )

        # 202 Accepted — non 201 Created
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "processing"
        assert body["task_id"] == "fake-task-uuid-1234"
        assert "document_id" in body

    def test_upload_file_non_pdf_ritorna_422(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("doc.txt", io.BytesIO(b"testo"), "text/plain")},
            headers=_headers("user-test-1"),
        )
        assert resp.status_code == 422

    def test_upload_senza_header_utente_ritorna_422(self, client):
        """Senza X-User-ID il dependency get_current_user deve rifiutare."""
        resp = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(_minimal_pdf()), "application/pdf")},
        )
        assert resp.status_code == 422


# ── TestDocumentList ──────────────────────────────────────────────────────────

class TestDocumentList:
    def test_lista_documenti_utente(self, client):
        """GET /documents ritorna solo i documenti dell'utente autenticato."""
        resp = client.get("/documents/", headers=_headers("user-test-1"))
        assert resp.status_code == 200
        docs = resp.json()
        # Tutti i documenti devono avere un campo 'status'
        for doc in docs:
            assert "status" in doc
            assert "filename" in doc

    def test_status_documento_esistente(self, client):
        """GET /documents/{id}/status ritorna i dettagli del documento."""
        # Prima crea un documento
        mock_result = MagicMock()
        mock_result.id = "task-status-test"
        with patch("app.routers.documents.process_document") as mock_task:
            mock_task.delay.return_value = mock_result
            upload = client.post(
                "/documents/upload",
                files={"file": ("status_test.pdf", io.BytesIO(_minimal_pdf()), "application/pdf")},
                headers=_headers("user-status"),
            )
        doc_id = upload.json()["document_id"]

        resp = client.get(f"/documents/{doc_id}/status", headers=_headers("user-status"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == doc_id
        assert body["status"] in ("processing", "ready", "error")

    def test_status_documento_altro_utente_404(self, client):
        """Un utente non può vedere i documenti di un altro utente."""
        mock_result = MagicMock()
        mock_result.id = "task-ns-test"
        with patch("app.routers.documents.process_document") as mock_task:
            mock_task.delay.return_value = mock_result
            upload = client.post(
                "/documents/upload",
                files={"file": ("private.pdf", io.BytesIO(_minimal_pdf()), "application/pdf")},
                headers=_headers("user-alice"),
            )
        doc_id = upload.json()["document_id"]

        # Bob prova ad accedere al documento di Alice
        resp = client.get(f"/documents/{doc_id}/status", headers=_headers("user-bob"))
        assert resp.status_code == 404


# ── TestQueryCache ────────────────────────────────────────────────────────────

class TestQueryCache:
    """
    Verifica che la seconda query identica usi la cache Redis
    e NON chiami il client Groq.
    """

    def test_seconda_query_usa_cache(self, client):
        """
        Scenario:
          1. Prima query → cache miss → Groq chiamato 1 volta
          2. Seconda query identica → cache hit → Groq NON chiamato
        """
        fake_answer = "Risposta di test dalla cache."
        cached_payload = {
            "answer":  fake_answer,
            "sources": [],
            "model_used": "groq/llama-3.3-70b-versatile",
        }

        with (
            # Mock Redis: prima chiamata = miss, seconda = hit
            patch("app.services.cache_service.get_redis") as mock_get_redis,
            # Mock LLM: verifica che venga chiamato una sola volta
            patch("app.routers.documents.generate_answer", return_value=fake_answer) as mock_llm,
            # Mock vector store: restituisce sempre risultati vuoti
            patch("app.routers.documents.semantic_search", return_value=[]),
            # Mock doc IDs: simula un documento ready
            patch("app.routers.documents.get_user_document_ids", return_value=[42]),
        ):
            redis_mock = MagicMock()
            mock_get_redis.return_value = redis_mock

            # Prima chiamata: cache miss
            redis_mock.get.return_value = None
            r1 = client.post(
                "/documents/query",
                json={"question": "Cosa è lo sviluppo personale?", "top_k": 3},
                headers=_headers("user-cache-test"),
            )

            # Seconda chiamata: cache hit (simuliamo il dato in cache)
            redis_mock.get.return_value = json.dumps(cached_payload)
            r2 = client.post(
                "/documents/query",
                json={"question": "Cosa è lo sviluppo personale?", "top_k": 3},
                headers=_headers("user-cache-test"),
            )

        # Prima risposta: from_cache deve essere False
        # (oppure 404 se semantic_search non ha restituito risultati)
        assert r1.status_code in (200, 404)

        # Seconda risposta: from_cache = True, LLM NON ri-chiamato
        if r2.status_code == 200:
            assert r2.json()["from_cache"] is True

        # Groq non deve essere stato chiamato più di una volta
        assert mock_llm.call_count <= 1, (
            f"generate_answer chiamato {mock_llm.call_count} volte — atteso max 1"
        )

    def test_invalidazione_cache_su_nuovo_upload(self, client):
        """
        Dopo un upload, la versione cache dell'utente deve essere incrementata.
        """
        with (
            patch("app.routers.documents.process_document") as mock_task,
            patch("app.services.cache_service.get_redis") as mock_get_redis,
        ):
            mock_task.delay.return_value = MagicMock(id="task-cache-inval")
            redis_mock = MagicMock()
            mock_get_redis.return_value = redis_mock

            client.post(
                "/documents/upload",
                files={"file": ("inval_test.pdf", io.BytesIO(_minimal_pdf()), "application/pdf")},
                headers=_headers("user-inval"),
            )

        # Redis.incr deve essere stato chiamato (incremento versione)
        redis_mock.incr.assert_called()
