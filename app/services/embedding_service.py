"""
Wrapper attorno a sentence-transformers.

Il modello viene caricato UNA SOLA VOLTA all'avvio dell'applicazione
(singleton) perché il caricamento costa ~1-2 secondi e ~90 MB di RAM.
Non thread-unsafe: SentenceTransformer è già thread-safe per l'inferenza.
"""
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Carica il modello una volta sola (lazy, al primo utilizzo)."""
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Riceve una lista di stringhe, restituisce una lista di vettori float.
    encode() normalizza automaticamente se normalize_embeddings=True:
    i vettori risultanti hanno norma 1, quindi cosine distance == dot product.
    """
    model = get_embedding_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Convenienza per embeddare una singola query."""
    return embed_texts([text])[0]
