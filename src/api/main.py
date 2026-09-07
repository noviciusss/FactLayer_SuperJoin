"""Main FastAPI application entrypoint."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.routes import router
from src.config import settings
from src.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables
    init_db()
    yield


app = FastAPI(
    title="Fact Knowledge Layer API",
    description="Domain-agnostic fact extraction, evidence grounding, and cross-document reconciliation engine.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount previews directory for visual evidence inspection
if settings.PREVIEW_DIR.exists():
    app.mount("/previews", StaticFiles(directory=str(settings.PREVIEW_DIR)), name="previews")

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
