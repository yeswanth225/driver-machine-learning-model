import sys
import os
import traceback
from pathlib import Path

# Add repository root and backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

backend_dir = REPO_ROOT / "backend"
if backend_dir.exists() and str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from backend.app.main import app
except Exception as e:
    err_trace = traceback.format_exc()
    print("FATAL STARTUP ERROR IN API/INDEX.PY:\n", err_trace)
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="IDAMS Diagnostics")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
    async def catch_all_error(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": "FastAPI Serverless Startup Failure",
                "exception": str(e),
                "traceback": err_trace,
                "cwd": os.getcwd(),
                "sys_path": sys.path[:5],
            },
        )
