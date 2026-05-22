"""
RecursiveCharacterTextSplitter con metadata di pagina.

Strategia:
  1. Estrai il testo PDF pagina per pagina, tracciando il char_offset
     di inizio di ogni pagina nel testo concatenato.
  2. Applica RecursiveCharacterTextSplitter sul testo completo:
     prova a spezzare su ["\n\n", "\n", ". ", " ", ""] in quest'ordine,
     usando il primo separatore che mantiene i pezzi entro chunk_size.
  3. Per ogni chunk, calcola page_number cercando in quale intervallo
     di pagine cade il char_offset iniziale del chunk.
  4. Aggiunge overlap: i primi `chunk_overlap` caratteri del chunk
     precedente vengono preposti al chunk corrente per preservare contesto.

Vantaggi rispetto al semplice split su paragrafi:
  - Rispetta la gerarchia semantica (paragrafi > frasi > parole > caratteri)
  - Chunk di dimensione prevedibile (max chunk_size + overlap)
  - Metadata pagina permettono citazioni precise nelle risposte RAG
"""
import io
from pypdf import PdfReader
from app.config import settings


# ── Separatori in ordine di preferenza ───────────────────────────────────────
_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]


# ── Core splitter ─────────────────────────────────────────────────────────────

def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Divide `text` in pezzi di al massimo `chunk_size` caratteri.
    Prova ogni separatore in ordine; se nessuno funziona usa il taglio duro.
    """
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    sep, *rest = separators

    if sep == "":
        # Fallback: taglio fisso
        return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = [p for p in text.split(sep) if p]
    if len(parts) <= 1:
        # Questo separatore non ha prodotto split → prova il successivo
        return _split_recursive(text, rest or [""], chunk_size)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part)

        # Il singolo pezzo è già troppo grande → ricorri
        if part_len > chunk_size:
            if current_parts:
                chunks.append(sep.join(current_parts).strip())
                current_parts, current_len = [], 0
            sub = _split_recursive(part, rest or [""], chunk_size)
            chunks.extend(sub)
            continue

        # Aggiungere `part` supererebbe chunk_size → flush e ricomincia
        added = current_len + len(sep) + part_len if current_parts else part_len
        if added > chunk_size and current_parts:
            chunks.append(sep.join(current_parts).strip())
            current_parts, current_len = [part], part_len
        else:
            current_parts.append(part)
            current_len = added

    if current_parts:
        chunks.append(sep.join(current_parts).strip())

    return [c for c in chunks if c.strip()]


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pages(pdf_bytes: bytes) -> tuple[str, list[int]]:
    """
    Ritorna:
      full_text  — intero testo del PDF concatenato
      page_starts — lista di char_offset in cui inizia ogni pagina (0-indexed)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    page_starts: list[int] = []

    for page in reader.pages:
        page_starts.append(len(full_text))
        text = page.extract_text() or ""
        full_text += text.strip() + "\n\n"

    return full_text, page_starts


def _page_of(char_offset: int, page_starts: list[int]) -> int:
    """Ritorna il numero di pagina (1-indexed) per un dato char_offset."""
    page = 1
    for i, start in enumerate(page_starts):
        if char_offset >= start:
            page = i + 1
        else:
            break
    return page


# ── API pubblica ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Estrae il testo grezzo (per retrocompatibilità con i test esistenti)."""
    full_text, _ = _extract_pages(pdf_bytes)
    return full_text


def chunk_text_with_metadata(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Principale funzione di chunking.

    Ritorna una lista di dict:
    {
        "content":     str,   # testo del chunk (con overlap dal precedente)
        "page_number": int,   # pagina PDF di origine (1-indexed)
        "char_offset": int,   # posizione nel testo completo
        "filename":    str,   # nome file originale
    }
    """
    csize   = chunk_size   or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    full_text, page_starts = _extract_pages(pdf_bytes)

    raw_chunks = _split_recursive(full_text, _SEPARATORS, csize)
    raw_chunks = [c for c in raw_chunks if len(c) > 10]

    result: list[dict] = []
    search_start = 0

    for i, chunk in enumerate(raw_chunks):
        # Trova la posizione del chunk nel testo originale
        char_offset = full_text.find(chunk[:50], search_start)
        if char_offset == -1:
            char_offset = search_start

        # Preponi overlap dal chunk precedente
        if i > 0 and overlap > 0:
            prev_tail = raw_chunks[i - 1][-overlap:]
            content = (prev_tail + " " + chunk).strip()
        else:
            content = chunk

        result.append({
            "content":     content,
            "page_number": _page_of(char_offset, page_starts),
            "char_offset": char_offset,
            "filename":    filename,
        })

        search_start = char_offset + len(chunk) - overlap

    return result


def chunk_text(text: str) -> list[str]:
    """
    Retrocompatibilità: usata dai test unitari esistenti.
    Fa chunking senza metadata.
    """
    raw = _split_recursive(text, _SEPARATORS, settings.chunk_size)
    chunks: list[str] = []
    for i, chunk in enumerate(raw):
        if i > 0 and settings.chunk_overlap > 0:
            prev_tail = raw[i - 1][-settings.chunk_overlap:]
            chunks.append((prev_tail + " " + chunk).strip())
        else:
            chunks.append(chunk)
    return [c for c in chunks if len(c) > 10]
