import sys
import os
from pathlib import Path

# Add repository root and backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT), str(REPO_ROOT / "backend"), os.getcwd(), os.path.join(os.getcwd(), "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.main import app

# Expose both app and handler for universal Vercel serverless compatibility
handler = app
