"""
Main FastAPI Application Entrypoint
====================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.api.endpoints import router as api_router
from backend.app.api.sensor_endpoints import sensor_router

app = FastAPI(
    title="Intelligent Driver Activity Monitoring & Safety Agent",
    description="Multimodal Driver Telemetry, Driving Style & Maneuver Analysis, and Deterministic Safety Risk Assessment API.",
    version="1.0.0",
)

# Configure CORS for local development and demonstration
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

STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ASSETS_DIR = STATIC_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount Mobile Web Sensor Logger for local Wi-Fi acquisition
MOBILE_LOGGER_DIR = Path(__file__).resolve().parent.parent.parent / "mobile" / "web-logger"
if MOBILE_LOGGER_DIR.exists():
    app.mount("/sensor-logger", StaticFiles(directory=str(MOBILE_LOGGER_DIR), html=True), name="sensor-logger")




@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the root monitoring dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "Intelligent Driver Activity Monitoring & Safety Agent",
        "docs_url": "/docs",
        "api_health": "/api/health",
    }
