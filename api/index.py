"""
Vercel Serverless Function Entrypoint
====================================
Exposes the single unified FastAPI application instance `app`.
Gracefully captures and returns any cold-start / import-time errors.
"""

import sys
import os
import traceback
from pathlib import Path

# Add repository root and backend directory to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "backend"),
    os.getcwd(),
    os.path.join(os.getcwd(), "backend"),
]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.app.main import app
except Exception:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _err = traceback.format_exc()
    print("FATAL IMPORT ERROR IN BACKEND.APP.MAIN:\n", _err, flush=True)

    app = FastAPI(title="IDAMS Diagnostics")

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def catch_startup_error(full_path: str):
        return JSONResponse(
            status_code=500,
            content={
                "status": "FATAL_STARTUP_ERROR",
                "message": "FastAPI failed to import backend.app.main at startup.",
                "traceback": _err,
                "cwd": os.getcwd(),
                "sys_path": sys.path[:10],
                "files_in_cwd": os.listdir(".") if os.path.exists(".") else [],
            },
        )
