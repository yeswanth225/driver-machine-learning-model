"""
Main FastAPI Application Entrypoint
====================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from backend.app.api.endpoints import router as api_router
from backend.app.api.sensor_endpoints import sensor_router

app = FastAPI(
    title="Intelligent Driver Activity Monitoring & Safety Agent",
    description="Multimodal Driver Telemetry, Driving Style & Maneuver Analysis, and Deterministic Safety Risk Assessment API.",
    version="1.0.0",
    redirect_slashes=False,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes under /api
app.include_router(api_router, prefix="/api")
app.include_router(sensor_router, prefix="/api")


def _find_static_dir() -> Optional[Path]:
    """Dynamically locate the static bundle directory across local and serverless environments."""
    candidates = [
        Path(__file__).resolve().parent / "static",
        Path.cwd() / "public",
        Path.cwd() / "backend" / "app" / "static",
        Path(__file__).resolve().parent.parent.parent / "public",
        Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
    ]
    for c in candidates:
        if c.exists() and (c / "index.html").exists():
            return c
    return Path(__file__).resolve().parent / "static"


STATIC_DIR = _find_static_dir()

if STATIC_DIR and STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    ASSETS_DIR = STATIC_DIR / "assets"
    if ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount Mobile Web Sensor Logger for local Wi-Fi acquisition if available
MOBILE_LOGGER_DIR = Path(__file__).resolve().parent.parent.parent / "mobile" / "web-logger"
if MOBILE_LOGGER_DIR.exists():
    try:
        app.mount("/sensor-logger", StaticFiles(directory=str(MOBILE_LOGGER_DIR), html=True), name="sensor-logger")
    except Exception:
        pass


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the root monitoring dashboard."""
    if STATIC_DIR and STATIC_DIR.exists():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            try:
                content = index_file.read_text(encoding="utf-8")
                return HTMLResponse(content=content)
            except Exception:
                return FileResponse(str(index_file))
    return {
        "service": "Intelligent Driver Activity Monitoring & Safety Agent",
        "status": "healthy",
        "docs_url": "/docs",
        "api_health": "/api/health",
    }


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str):
    """Fallback router for client-side React SPA navigation."""
    if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    if STATIC_DIR and STATIC_DIR.exists():
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))

        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            try:
                content = index_file.read_text(encoding="utf-8")
                return HTMLResponse(content=content)
            except Exception:
                return FileResponse(str(index_file))

    return {
        "service": "Intelligent Driver Activity Monitoring & Safety Agent",
        "status": "healthy",
        "docs_url": "/docs",
        "api_health": "/api/health",
    }
