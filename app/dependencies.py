"""
Dependency FastAPI per l'autenticazione utente.

Per ora usa un semplice header X-User-ID.
In produzione sostituire con JWT (es. fastapi-users o python-jose).

Utilizzo nel router:
    from app.dependencies import get_current_user
    user_id: str = Depends(get_current_user)
"""
from fastapi import Header, HTTPException, status


async def get_current_user(x_user_id: str = Header(..., alias="X-User-ID")) -> str:
    """
    Legge l'header X-User-ID e lo usa come identificatore utente.
    Tutti i documenti e le query vengono isolati per questo valore.
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header X-User-ID mancante o vuoto.",
        )
    return x_user_id.strip()
