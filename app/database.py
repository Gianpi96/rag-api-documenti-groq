from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency FastAPI: apre una sessione per request e la chiude al termine."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Inizializza il database in modo idempotente:
      1. Abilita l'estensione pgvector
      2. Crea le tabelle se non esistono (checkfirst=True)
      3. Crea gli indici con IF NOT EXISTS — sicuro ad ogni riavvio
    """
    with engine.connect() as conn:
        # ── Estensione pgvector ───────────────────────────────────────────────
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # ── Tabelle (checkfirst evita errori se già esistono) ─────────────────────
    Base.metadata.create_all(bind=engine, checkfirst=True)

    # ── Indici — sempre con IF NOT EXISTS ─────────────────────────────────────
    with engine.connect() as conn:
        # B-tree su user_id per il WHERE filter nelle query RAG
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_document_chunks_user_id
            ON document_chunks (user_id)
        """))

        # HNSW per la ricerca vettoriale con cosine distance
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))

        conn.commit()
