"""
Cache Redis per le risposte RAG.

Strategia di invalidazione basata su "versione utente":
  - Ogni utente ha un contatore in Redis: rag:ver:{user_id}
  - La chiave della cache include questa versione
  - Quando l'utente carica un nuovo documento, incrementiamo il contatore
  - Tutte le vecchie chiavi diventano automaticamente irraggiungibili
    senza dover fare SCAN (operazione O(N))

Struttura chiave:
  rag:{user_id}:{sha256(version + question + sorted_doc_ids)}

TTL: 1 ora (3600 secondi) — configurabile.
"""
import hashlib
import json
import logging
from functools import lru_cache

import redis as redis_lib

from app.config import settings

logger = logging.getLogger(__name__)

CACHE_TTL = 3600  # secondi


@lru_cache(maxsize=1)
def get_redis() -> redis_lib.Redis:
    """Singleton Redis client — caricato una volta sola."""
    return redis_lib.from_url(settings.redis_url, decode_responses=True)


# ── Versione utente (per invalidazione O(1)) ─────────────────────────────────

def _get_user_version(user_id: str) -> str:
    try:
        version = get_redis().get(f"rag:ver:{user_id}")
        return version or "0"
    except Exception:
        return "0"


def invalidate_user_cache(user_id: str) -> None:
    """
    Invalida TUTTA la cache dell'utente in O(1).
    Incrementa il contatore di versione: tutte le chiavi con la versione
    precedente non verranno più trovate (il TTL le pulirà da solo).
    """
    try:
        get_redis().incr(f"rag:ver:{user_id}")
        logger.info(f"[cache] Versione cache invalidata per user='{user_id}'")
    except Exception as e:
        logger.warning(f"[cache] Impossibile invalidare cache: {e}")


# ── Chiave cache ──────────────────────────────────────────────────────────────

def _make_key(user_id: str, question: str, doc_ids: list[int]) -> str:
    version = _get_user_version(user_id)
    payload = f"{version}:{question}:{sorted(doc_ids)}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"rag:{user_id}:{digest}"


# ── Get / Set ─────────────────────────────────────────────────────────────────

def get_cached_response(
    user_id: str,
    question: str,
    doc_ids: list[int],
) -> dict | None:
    """
    Cerca in cache. Ritorna il dict della risposta serializzato,
    oppure None se non presente o Redis non disponibile.
    """
    try:
        key = _make_key(user_id, question, doc_ids)
        raw = get_redis().get(key)
        if raw:
            logger.info(f"[cache] HIT per user='{user_id}' key={key[:20]}…")
            return json.loads(raw)
        logger.debug(f"[cache] MISS per user='{user_id}'")
        return None
    except Exception as e:
        logger.warning(f"[cache] Errore lettura cache: {e}")
        return None


def set_cached_response(
    user_id: str,
    question: str,
    doc_ids: list[int],
    response: dict,
    ttl: int = CACHE_TTL,
) -> None:
    """
    Salva la risposta in cache con TTL.
    Se Redis non è disponibile, logga e continua senza errori.
    """
    try:
        key = _make_key(user_id, question, doc_ids)
        get_redis().setex(key, ttl, json.dumps(response))
        logger.info(f"[cache] SET per user='{user_id}' ttl={ttl}s")
    except Exception as e:
        logger.warning(f"[cache] Errore scrittura cache: {e}")
