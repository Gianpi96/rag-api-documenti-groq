from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: str
    database_url: str
    embedding_model: str = "all-MiniLM-L6-v2"
    groq_model: str = "llama-3.3-70b-versatile"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    # Redis — broker Celery e cache query
    redis_url: str = "redis://localhost:6379/0"

    # Directory dove salvare i PDF in attesa di elaborazione Celery
    upload_dir: str = "uploads"


settings = Settings()
