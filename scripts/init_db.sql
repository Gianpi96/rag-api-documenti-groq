-- Esegui questo script UNA SOLA VOLTA come superuser postgres
-- per preparare il database prima di avviare l'applicazione.

-- 1. Crea il database (se non esiste)
-- Nota: questo comando va eseguito FUORI da una transazione,
--       quindi usalo dalla shell psql o da pgAdmin.
-- CREATE DATABASE rag_db;

-- 2. Connettiti a rag_db e abilita pgvector
-- \c rag_db
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. Verifica che l'estensione sia attiva
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
