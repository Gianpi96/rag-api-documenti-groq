import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models import Document
from app.schemas import (
    ChunkOut,
    DocumentOut,
    DocumentStatusOut,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.cache_service import (
    get_cached_response,
    invalidate_user_cache,
    set_cached_response,
)
from app.services.embedding_service import embed_query
from app.services.llm_service import generate_answer
from app.services.vector_store import get_user_document_ids, semantic_search
from app.tasks.document_tasks import process_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_MB = 50
UPLOAD_DIR = Path(settings.upload_dir)


# ── POST /documents/upload ────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Carica un PDF — elaborazione asincrona via Celery",
)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Ritorna immediatamente con {task_id, status: 'processing'}.
    L'embedding e il salvataggio su pgvector avvengono in background.
    Usa GET /documents/{id}/status per monitorare il progresso.
    """
    # ── Validazione ───────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(422, "Solo file PDF sono accettati.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File troppo grande ({size_mb:.1f} MB). Limite: {MAX_UPLOAD_MB} MB.")

    # ── Salva file su disco per il worker Celery ──────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in file.filename)
    file_path = str(UPLOAD_DIR / f"u{user_id}_{safe_name}")
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # ── Crea il record Document con status='processing' ───────────────────────
    doc = Document(
        user_id=user_id,
        filename=file.filename,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # ── Dispatch task Celery ──────────────────────────────────────────────────
    task = process_document.delay(doc.id, file_path, user_id)

    # Salviamo il task_id per poterlo consultare via /status
    doc.task_id = task.id
    db.commit()

    # Invalida cache: un nuovo documento cambia i risultati potenziali
    invalidate_user_cache(user_id)

    logger.info(f"[upload] user='{user_id}' file='{file.filename}' doc_id={doc.id} task={task.id}")

    return UploadResponse(
        document_id=doc.id,
        task_id=task.id,
        status="processing",
        filename=doc.filename,
        message="Documento in elaborazione. Usa GET /documents/{id}/status per monitorare.",
    )


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=list[DocumentOut],
    summary="Lista tutti i documenti dell'utente autenticato",
)
def list_documents(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ritorna i documenti dell'utente con il loro status corrente."""
    docs = (
        db.query(Document)
        .filter(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return docs


# ── GET /documents/{id}/status ────────────────────────────────────────────────

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusOut,
    summary="Stato di elaborazione di un documento",
)
def get_document_status(
    document_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Possibili valori di status:
      - processing: il worker Celery sta elaborando
      - ready:      indicizzato e pronto per le query
      - error:      fallito (error_message contiene il dettaglio)
    """
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == user_id)
        .first()
    )
    if not doc:
        raise HTTPException(404, f"Documento {document_id} non trovato.")
    return doc


# ── POST /documents/query ─────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ricerca semantica + risposta LLM con cache Redis",
)
def query_documents(
    body: QueryRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Pipeline:
      1. Recupera gli ID dei documenti 'ready' dell'utente
      2. Controlla la cache Redis (chiave = hash(user+query+doc_ids))
      3. Se cache miss: embedding query → ANN pgvector → Groq LLM
      4. Salva in cache con TTL 1 ora
      5. Ritorna risposta + sources + from_cache flag
    """
    # ── Documenti disponibili dell'utente ─────────────────────────────────────
    doc_ids = get_user_document_ids(db, user_id)
    if not doc_ids:
        raise HTTPException(
            404,
            "Nessun documento 'ready' trovato. Carica almeno un PDF e aspetta l'elaborazione.",
        )

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cached = get_cached_response(user_id, body.question, doc_ids)
    if cached:
        return QueryResponse(**cached, from_cache=True)

    # ── Embedding della query ─────────────────────────────────────────────────
    try:
        q_embedding = embed_query(body.question)
    except Exception as e:
        raise HTTPException(500, f"Errore embedding query: {e}")

    # ── Ricerca semantica filtrata per user_id ────────────────────────────────
    results = semantic_search(
        db=db,
        query_embedding=q_embedding,
        top_k=body.top_k,
        user_id=user_id,
    )
    if not results:
        raise HTTPException(404, "Nessun chunk trovato per questa query.")

    # ── Generazione risposta LLM ──────────────────────────────────────────────
    context_texts = [r["content"] for r in results]
    try:
        answer = generate_answer(question=body.question, context_chunks=context_texts)
    except Exception as e:
        raise HTTPException(500, f"Errore LLM: {e}")

    sources = [
        ChunkOut(
            id=r["id"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            page_number=r.get("page_number"),
            similarity=r["similarity"],
        )
        for r in results
    ]

    model_used = f"groq/{settings.groq_model}"
    response_data = {
        "answer":     answer,
        "sources":    [s.model_dump() for s in sources],
        "model_used": model_used,
    }

    # ── Salva in cache ────────────────────────────────────────────────────────
    set_cached_response(user_id, body.question, doc_ids, response_data)

    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=model_used,
        from_cache=False,
    )
