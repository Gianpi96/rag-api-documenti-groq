from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


# ── Chunk ────────────────────────────────────────────────────────────────────

class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    chunk_index: int
    content: str
    page_number: int | None = None
    similarity: float = Field(description="Cosine similarity [0-1]")


# ── Upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Risposta immediata: il documento è in coda, non ancora elaborato."""
    document_id: int
    task_id: str
    status: str = "processing"
    filename: str
    message: str


# ── Document list ─────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: str
    total_chunks: int
    created_at: datetime


class DocumentStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: str            # processing | ready | error
    total_chunks: int
    error_message: str | None = None
    task_id: str | None = None


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkOut]
    model_used: str
    from_cache: bool = Field(default=False, description="True se la risposta viene dalla cache Redis")
