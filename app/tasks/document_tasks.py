"""
Task Celery per l'elaborazione asincrona dei documenti.

Flusso:
  1. Il router salva il PDF su disco e crea un Document con status='processing'
  2. Dispatch: process_document.delay(document_id, file_path, user_id)
  3. Il worker esegue:
       a. Legge il PDF dal disco
       b. RecursiveCharacterTextSplitter con metadata pagina
       c. Embedding batch (sentence-transformers locale)
       d. Salvataggio chunk su pgvector
       e. Aggiorna Document.status = 'ready'
       f. Invalida la cache Redis dell'utente
       g. Rimuove il file temporaneo
  4. In caso di errore: Document.status = 'error', error_message = str(e)
"""
import logging
import os

from app.database import SessionLocal
from app.models import Document
from app.services.cache_service import invalidate_user_cache
from app.services.embedding_service import embed_texts
from app.services.pdf_service import chunk_text_with_metadata
from app.services.vector_store import save_chunks_for_document
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_document(self, document_id: int, file_path: str, user_id: str) -> dict:
    """
    Task principale di ingestione.

    bind=True → `self` permette retry e aggiornamento dello stato.
    max_retries=3 → riprova in caso di errori transitori (es. DB down).
    """
    db = SessionLocal()
    try:
        # ── 1. Marca come in lavorazione ────────────────────────────────────
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document {document_id} non trovato nel DB")
        doc.status = "processing"
        db.commit()
        logger.info(f"[task] Avvio elaborazione document_id={document_id} user='{user_id}'")

        # ── 2. Leggi PDF ─────────────────────────────────────────────────────
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # ── 3. Chunking con metadata ─────────────────────────────────────────
        chunks = chunk_text_with_metadata(
            pdf_bytes=file_bytes,
            filename=doc.filename,
        )
        if not chunks:
            raise ValueError("Nessun chunk estratto dal documento.")
        logger.info(f"[task] {len(chunks)} chunk estratti per document_id={document_id}")

        # ── 4. Embedding batch ───────────────────────────────────────────────
        texts = [c["content"] for c in chunks]
        embeddings = embed_texts(texts)

        # ── 5. Salvataggio su pgvector ───────────────────────────────────────
        save_chunks_for_document(
            db=db,
            doc_id=document_id,
            user_id=user_id,
            chunks_with_meta=chunks,
            embeddings=embeddings,
        )

        # ── 6. Aggiorna documento come pronto ────────────────────────────────
        doc.status = "ready"
        doc.total_chunks = len(chunks)
        db.commit()
        logger.info(f"[task] document_id={document_id} → ready ({len(chunks)} chunk)")

        # ── 7. Invalida cache utente ─────────────────────────────────────────
        invalidate_user_cache(user_id)

        return {"document_id": document_id, "total_chunks": len(chunks), "status": "ready"}

    except Exception as exc:
        logger.exception(f"[task] Errore elaborazione document_id={document_id}: {exc}")
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = "error"
                doc.error_message = str(exc)
                db.commit()
        except Exception:
            pass
        # Retry automatico per errori transitori
        raise self.retry(exc=exc)

    finally:
        db.close()
        # ── 8. Rimuovi file temporaneo ───────────────────────────────────────
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
