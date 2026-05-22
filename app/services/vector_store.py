"""
Operazioni CRUD sul vector store pgvector — v2.

Novità rispetto a v1:
  - save_chunks_for_document(): accetta chunks con metadata (page_number, char_offset)
    e user_id per il namespace utente.
  - semantic_search(): filtra per user_id (WHERE user_id = :user_id).
    Imposta hnsw.ef_search=100 per compensare la riduzione di accuracy
    causata dal pre-filtering sull'indice HNSW.
"""
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models import Document, DocumentChunk


def save_chunks_for_document(
    db: Session,
    doc_id: int,
    user_id: str,
    chunks_with_meta: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Salva i chunk di un documento già esistente nel DB.
    Chiamata dal Celery task dopo che il Document è stato creato.

    chunks_with_meta: lista di dict con chiavi content, page_number, char_offset, filename
    """
    chunk_objects = [
        DocumentChunk(
            document_id=doc_id,
            user_id=user_id,
            chunk_index=i,
            content=meta["content"],
            page_number=meta.get("page_number"),
            char_offset=meta.get("char_offset", 0),
            embedding=embedding,
        )
        for i, (meta, embedding) in enumerate(zip(chunks_with_meta, embeddings))
    ]
    db.bulk_save_objects(chunk_objects)
    db.commit()


def semantic_search(
    db: Session,
    query_embedding: list[float],
    top_k: int,
    user_id: str,
) -> list[dict]:
    """
    ANN search filtrata per user_id (namespace utente).

    Nota su hnsw.ef_search:
      L'HNSW index esegue una ricerca approssimata; con un WHERE filter
      molti candidati vengono scartati. Alzare ef_search a 100 (default: 40)
      fa esplorare più candidati e riduce i falsi negativi.
      Il costo è ~2× più lento, accettabile per query interattive.
    """
    # SET LOCAL deve essere una chiamata separata (psycopg3 non ammette
    # più comandi in un prepared statement)
    db.execute(text("SET LOCAL hnsw.ef_search = 100"))

    sql = text("""
        SELECT
            dc.id,
            dc.document_id,
            dc.chunk_index,
            dc.content,
            dc.page_number,
            1 - (dc.embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM document_chunks dc
        WHERE dc.user_id = :user_id
        ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
    """)

    result = db.execute(
        sql,
        {
            "query_vec": str(query_embedding),
            "user_id": user_id,
            "top_k": top_k,
        },
    )

    rows = result.fetchall()
    return [
        {
            "id":          row.id,
            "document_id": row.document_id,
            "chunk_index": row.chunk_index,
            "content":     row.content,
            "page_number": row.page_number,
            "similarity":  float(row.similarity),
        }
        for row in rows
    ]


def get_user_document_ids(db: Session, user_id: str) -> list[int]:
    """Ritorna tutti i document_id 'ready' dell'utente — usati come chiave cache."""
    rows = (
        db.query(Document.id)
        .filter(Document.user_id == user_id, Document.status == "ready")
        .all()
    )
    return [r.id for r in rows]
