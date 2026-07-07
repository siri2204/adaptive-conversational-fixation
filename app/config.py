"""
Central configuration for the fixation-detection backend.

All settings are overridable via environment variables (or a .env file).
Defaults are chosen so the app runs fully offline out of the box (mock LLM +
mock embeddings), so you can develop/test the logic without any API keys or
network access, then flip two env vars to go "live" with Gemini +
sentence-transformers.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- LLM backend ---
    llm_backend: str = Field(default="mock", description="'gemini' or 'mock'")
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")

    # --- Embedding backend ---
    embedding_backend: str = Field(default="mock", description="'sentence-transformers' or 'mock'")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384, description="Used by the mock backend to size vectors")

    # --- Database ---
    database_url: str = Field(default="sqlite:///./fixation.db")

    # --- Fixation detection ---
    fixation_window: int = Field(default=6, description="Number of most-recent turns considered")
    fixation_similarity_threshold: float = Field(
        default=0.82, description="Avg pairwise cosine similarity above this = converged"
    )
    fixation_dispersion_threshold: float = Field(
        default=0.15, description="Embedding dispersion below this = converged"
    )
    fixation_score_threshold: float = Field(
        default=0.65, description="Combined fixation_score above this triggers adaptive intervention"
    )

    # --- Intervention strategies ---
    fixed_interval_turns: int = Field(default=5, description="Turns between interventions for fixed-interval strategy")
    adaptive_cooldown_turns: int = Field(default=4, description="Min turns between adaptive interventions")

    class Config:
        env_file = ".env"
        env_prefix = "FIXATION_"


settings = Settings()
