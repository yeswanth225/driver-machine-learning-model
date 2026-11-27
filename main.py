"""
Local Development & Root Entrypoint
===================================
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "backend"),
    os.getcwd(),
    os.path.join(os.getcwd(), "backend"),
]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
