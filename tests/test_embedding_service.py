"""
Test unitari per embedding_service.
Richiedono solo sentence-transformers installato, nessuna rete dopo il primo download.
Esegui con: python -m pytest tests/test_embedding_service.py -v
"""
import pytest
from app.services.embedding_service import embed_query, embed_texts


def test_embed_query_dimensione():
    """all-MiniLM-L6-v2 produce vettori di 384 dimensioni."""
    vec = embed_query("Che cos'è il machine learning?")
    assert len(vec) == 384


def test_embed_texts_batch():
    """embed_texts restituisce tanti vettori quanti i testi in input."""
    texts = ["Primo testo.", "Secondo testo.", "Terzo testo."]
    vecs = embed_texts(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_normalizzazione():
    """Con normalize_embeddings=True la norma di ogni vettore deve essere ~1."""
    import math
    vec = embed_query("Test di normalizzazione")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Norma attesa 1.0, ottenuta {norm}"


def test_similarita_semantica():
    """Due frasi semanticamente simili devono avere cosine similarity > 0.7."""
    import numpy as np
    v1 = embed_query("Il gatto è sul tavolo")
    v2 = embed_query("Il felino si trova sul ripiano")
    v3 = embed_query("La borsa di Milano ha chiuso in rialzo")

    sim_simile = float(np.dot(v1, v2))
    sim_diverso = float(np.dot(v1, v3))

    assert sim_simile > sim_diverso, (
        f"Similarità semantica attesa > diverso: {sim_simile:.3f} vs {sim_diverso:.3f}"
    )
