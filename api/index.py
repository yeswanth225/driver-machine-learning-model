import sys
import os
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add repository root and backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT), str(REPO_ROOT / "backend"), os.getcwd(), os.path.join(os.getcwd(), "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

app = FastAPI(
    title="Intelligent Driver Activity Monitoring & Safety Agent",
    description="Multimodal Driver Telemetry, Driving Style & Maneuver Analysis, and Deterministic Safety Risk Assessment API.",
    version="1.0.0",
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

startup_error = None

@app.get("/api/test")
def test_diag():
    return {
        "status": "ok",
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "files_in_cwd": os.listdir("."),
        "sys_path": sys.path[:5],
        "has_startup_error": startup_error is not None,
        "startup_error": startup_error,
    }

try:
    from backend.app.api.endpoints import router as api_router
    from backend.app.api.sensor_endpoints import sensor_router
    from backend.app.main import serve_dashboard, serve_spa

    app.include_router(api_router, prefix="/api")
    app.include_router(sensor_router, prefix="/api")
    app.include_router(api_router)
    app.include_router(sensor_router)

    app.add_api_route("/", serve_dashboard, methods=["GET"], include_in_schema=False)
    app.add_api_route("/{full_path:path}", serve_spa, methods=["GET"], include_in_schema=False)
except Exception:
    startup_error = traceback.format_exc()
    print("STARTUP IMPORT ERROR IN API/INDEX.PY:\n", startup_error, flush=True)

    @app.get("/api/health")
    def fallback_health():
        return {
            "status": "degraded",
            "error": "Startup error during module loading",
            "details": startup_error,
        }

handler = app
