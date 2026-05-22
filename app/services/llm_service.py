"""
Interazione con Groq (llama3-70b-8192).

Il prompt segue il pattern RAG classico:
  - SYSTEM: definisce il ruolo e la regola "rispondi solo con il contesto"
  - USER: contesto recuperato + domanda dell'utente

Tenere SYSTEM e USER separati migliora la cache dei token su Groq
e rende più semplice il fine-tuning futuro del prompt.
"""
from groq import Groq
from app.config import settings

_client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """Sei un assistente esperto nell'analisi di documenti.
Rispondi SOLO usando le informazioni presenti nel contesto fornito.
Se la risposta non è nel contesto, dì esplicitamente "Non ho informazioni sufficienti nel documento per rispondere."
Sii preciso, cita i passaggi rilevanti quando utile, e rispondi in italiano."""


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Genera una risposta usando i chunk come contesto.
    context_chunks: lista di testi recuperati da pgvector, ordinati per rilevanza.
    """
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""CONTESTO DAL DOCUMENTO:
{context_block}

DOMANDA:
{question}

RISPOSTA:"""

    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,   # bassa temperatura = risposte più deterministiche/fattuali
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
