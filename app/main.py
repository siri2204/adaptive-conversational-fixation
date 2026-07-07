from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import sessions
from app.config import settings

app = FastAPI(
    title="Conversational Fixation Detection & Mitigation API",
    description=(
        "Stateful backend for detecting semantic fixation in long-form human-AI "
        "co-creation and delivering structured exploration-tree interventions."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend": settings.llm_backend,
        "embedding_backend": settings.embedding_backend,
        "gemini_model": settings.gemini_model,
        "gemini_api_key_set": bool(settings.gemini_api_key),
        "fixation_score_threshold": settings.fixation_score_threshold,
    }
