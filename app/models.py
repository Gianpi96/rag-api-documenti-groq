"""
Schema DB pgvector — v2
========================

Tabella documents
-----------------
  id            SERIAL PRIMARY KEY
  user_id       TEXT NOT NULL                  -- namespace utente
  filename      TEXT NOT NULL
  status        TEXT DEFAULT 'processing'      -- processing | ready | error
  task_id       TEXT                           -- Celery task ID
  total_chunks  INTEGER DEFAULT 0
  error_message TEXT                           -- popolato se status='error'
  created_at    TIMESTAMPTZ DEFAULT now()

Tabella document_chunks
-----------------------
  id            SERIAL PRIMARY KEY
  document_id   INTEGER REFERENCES documents(id) ON DELETE CASCADE
  user_id       TEXT NOT NULL                  -- denormalizzato per WHERE veloce
  chunk_index   INTEGER NOT NULL               -- posizione del chunk nel doc
  content       TEXT NOT NULL                  -- testo grezzo del chunk
  page_number   INTEGER                        -- pagina PDF di origine
  char_offset   INTEGER DEFAULT 0             -- posizione nel testo completo
  embedding     VECTOR(384)                    -- all-MiniLM-L6-v2
  created_at    TIMESTAMPTZ DEFAULT now()

Indice HNSW su embedding + indice su user_id per filtering efficiente.
"""
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base

EMBEDDING_DIM = 384


class Document(Base):
    __tablename__ = "documents"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(String, nullable=False, index=True)
    filename      = Column(Text, nullable=False)
    status        = Column(String, default="processing", nullable=False)
    task_id       = Column(String, nullable=True)
    total_chunks  = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id     = Column(String, nullable=False, index=True)   # denormalizzato
    chunk_index = Column(Integer, nullable=False)
    content     = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    char_offset = Column(Integer, default=0)
    embedding   = Column(Vector(EMBEDDING_DIM), nullable=False)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")


# Gli indici vengono creati in database.py con IF NOT EXISTS
# per essere idempotenti ad ogni riavvio dell'applicazione.
