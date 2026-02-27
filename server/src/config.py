from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVER_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SQLite
    sqlite_url: str = f"sqlite:///{SERVER_DIR}/heaven.db"

    # ChromaDB
    chroma_persist_dir: str = str(SERVER_DIR / "chroma_data")
    embedding_model: str = "all-MiniLM-L6-v2"  # swap when model layer is decided

    # LLM provider selection — controls what registry.primary / registry.cheap resolve to.
    # Supported values: "claude" | "openai_compatible"
    # "openai_compatible" covers DeepSeek, OpenRouter, Gemini (via OpenAI endpoint), etc.
    primary_provider: str = "claude"
    primary_model: str = "claude-sonnet-4-6"
    cheap_provider: str = "claude"
    cheap_model: str = "claude-haiku-4-5-20251001"

    # Anthropic (used when provider = "claude")
    anthropic_api_key: str = ""

    # OpenAI-compatible (used when provider = "openai_compatible")
    # Override openai_base_url for DeepSeek, OpenRouter, Gemini, etc.:
    #   DeepSeek:   https://api.deepseek.com/v1
    #   OpenRouter: https://openrouter.ai/api/v1
    #   Gemini:     https://generativelanguage.googleapis.com/v1beta/openai
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # External APIs
    wolfram_app_id: str = ""
    semantic_scholar_api_key: str = ""  # optional — raises rate limit without key

    # Lean 4
    lean_executable: str = "lean"       # must be on PATH
    lean_project_dir: str = str(SERVER_DIR / "lean_project")
    lean_timeout_seconds: int = 60


settings = Settings()
