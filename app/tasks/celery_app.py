"""
Configurazione Celery.

Broker e backend: Redis (stessa istanza usata per la cache query).
Il worker viene avviato separatamente:
    celery -A app.tasks.celery_app worker --loglevel=info -P solo

Nota: su Windows usare -P solo (non fork) perché multiprocessing
non funziona correttamente con i child process di Celery su Windows.
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.document_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,  # silenzia il deprecation warning
)
