# Immagine per deployment containerizzato (API + worker)
FROM python:3.12-slim

WORKDIR /app

# Dipendenze OS minime per psycopg (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crea directory per upload temporanei
RUN mkdir -p uploads

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
