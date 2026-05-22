import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inizializza il DB (estensione pgvector + tabelle) all'avvio."""
    logger.info("Inizializzazione database pgvector...")
    init_db()
    logger.info("Database pronto.")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="RAG API — Documenti con Groq",
    description="Upload PDF → chunking → embedding locale → pgvector → risposta con llama3-70b",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)


@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok"}
