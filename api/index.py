"""
Vercel Serverless Function Entrypoint
====================================
Exposes the single unified FastAPI application instance `app`.
"""

import sys
import os
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

from backend.app.main import app

# NOTE: Do NOT assign `handler = app`!
# Vercel's Python runtime requires `app` to detect and invoke FastAPI natively via ASGI.
