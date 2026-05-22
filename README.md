# RAG API — Documenti con Groq

Sistema **Retrieval-Augmented Generation** costruito con FastAPI, pgvector e Groq.  
Carica PDF, li indicizza automaticamente in background e risponde a domande semantiche sul loro contenuto.

---

## Stack

| Componente | Tecnologia |
|---|---|
| API | FastAPI + Uvicorn |
| Embedding | sentence-transformers `all-MiniLM-L6-v2` (locale, gratuito) |
| Vector store | PostgreSQL + pgvector (HNSW index) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Task queue | Celery + Redis |
| Cache query | Redis (TTL 1h, invalidazione O(1)) |

---

## Architettura

```
POST /documents/upload
  └─► salva PDF su disco
  └─► crea Document(status=processing)
  └─► Celery task → chunking + embedding + pgvector
  └─► 202 { task_id, document_id } ← risposta immediata

GET /documents/{id}/status
  └─► processing | ready | error

POST /documents/query
  └─► Redis cache lookup
      ├─► HIT  → risposta in <10ms (from_cache: true)
      └─► MISS → pgvector ANN → Groq LLM → salva cache
```

### Schema DB pgvector

```
documents
  id, user_id, filename, status, task_id, total_chunks, error_message, created_at

document_chunks
  id, document_id, user_id, chunk_index, content
  page_number, char_offset, embedding VECTOR(384), created_at

Indici:
  HNSW su embedding (vector_cosine_ops, m=16, ef_construction=64)
  B-tree su user_id
```

---

## Prerequisiti

- Python 3.12+
- Docker Desktop
- Chiave API Groq (gratuita su [console.groq.com](https://console.groq.com))

---

## Setup

### 1. Clona e configura l'ambiente

```bash
git clone <repo-url>
cd rag-api-documenti-groq
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Variabili ambiente

```bash
cp .env.example .env
```

Compila `.env`:

```ini
GROQ_API_KEY=gsk_...
DATABASE_URL=postgresql+psycopg://postgres:1234@localhost:5433/rag_db
REDIS_URL=redis://localhost:6379/0
```

### 3. Avvia i servizi Docker

```bash
# Postgres + pgvector
docker run -d --name rag-postgres \
  -e POSTGRES_PASSWORD=1234 \
  -e POSTGRES_DB=rag_db \
  -p 5433:5432 \
  -v rag-postgres-data:/var/lib/postgresql \
  pgvector/pgvector:pg18

# Redis
docker run -d --name rag-redis \
  -p 6379:6379 \
  -v rag-redis-data:/data \
  redis:7-alpine
```

Oppure con Docker Compose (solo infrastruttura):

```bash
docker compose up -d postgres redis
```

### 4. Avvia l'applicazione (3 terminali)

```bash
# Terminale 1 — API
python -m uvicorn app.main:app --reload --port 8000

# Terminale 2 — Worker Celery (su Windows usare -P solo)
celery -A app.tasks.celery_app worker --loglevel=info -P solo

# Terminale 3 — test / comandi
```

---

## Utilizzo

### Upload PDF

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-User-ID: mario" \
  -F "file=@documento.pdf"
```

```json
{
  "document_id": 1,
  "task_id": "abc-123",
  "status": "processing",
  "filename": "documento.pdf"
}
```

### Monitora elaborazione

```bash
curl http://localhost:8000/documents/1/status \
  -H "X-User-ID: mario"
```

```json
{ "status": "ready", "total_chunks": 127 }
```

### Lista documenti

```bash
curl http://localhost:8000/documents/ \
  -H "X-User-ID: mario"
```

### Query semantica

```bash
curl -X POST http://localhost:8000/documents/query \
  -H "X-User-ID: mario" \
  -H "Content-Type: application/json" \
  -d '{"question": "Di cosa parla il documento?", "top_k": 3}'
```

```json
{
  "answer": "Il documento parla di...",
  "sources": [
    { "id": 7, "chunk_index": 6, "page_number": 2, "similarity": 0.87, "content": "..." }
  ],
  "model_used": "groq/llama-3.3-70b-versatile",
  "from_cache": false
}
```

La seconda query identica risponde con `"from_cache": true` senza chiamare Groq.

---

## Namespace utenti

Ogni richiesta richiede l'header `X-User-ID`. I documenti e le query sono isolati per utente: l'utente `mario` non vede i documenti di `luigi` e viceversa.

> In produzione sostituire con autenticazione JWT.

---

## Test

```bash
# Unit test chunking (no DB, no rete)
python -m pytest tests/test_pdf_service.py -v

# Unit test embedding (scarica modello al primo run ~90 MB)
python -m pytest tests/test_embedding_service.py -v -s

# Test integrazione: upload async, namespace, cache Redis
python -m pytest tests/test_advanced.py -v -s

# Tutti i test
python -m pytest -v
```

---

## Riavvio rapido

```bash
docker start rag-postgres rag-redis
celery -A app.tasks.celery_app worker --loglevel=info -P solo
python -m uvicorn app.main:app --reload --port 8000
```

---

## Prossimi step

- [ ] Autenticazione JWT (sostituire header X-User-ID)
- [ ] OCR per PDF scansionati (pytesseract + pdf2image)
- [ ] Filtro per documento singolo nella query
- [ ] Alembic migrations per evoluzioni dello schema
- [ ] Reranking con cross-encoder
