"""FastAPI application entry point.

Run with:  uvicorn msbgen.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from msbgen import __version__
from msbgen.api.routes import router as api_router
from msbgen.config import STATIC_DIR


def create_app() -> FastAPI:
    app = FastAPI(
        title="MSBGen",
        description="Generative Image Reconstruction from Most-Significant Bitplanes",
        version=__version__,
    )
    app.include_router(api_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
