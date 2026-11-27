from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import traceback
from pathlib import Path

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        diag = {}
        diag["python_version"] = sys.version
        diag["cwd"] = os.getcwd()
        diag["sys_path"] = sys.path

        try:
            diag["cwd_files"] = os.listdir(".")
        except Exception as e:
            diag["cwd_files_error"] = str(e)

        try:
            diag["parent_files"] = os.listdir("..")
        except Exception as e:
            diag["parent_files_error"] = str(e)

        # Test imports
        imports_to_test = [
            "fastapi",
            "pydantic",
            "aiofiles",
            "uvicorn",
            "numpy",
            "pandas",
            "sklearn",
            "joblib",
        ]
        import_results = {}
        for mod in imports_to_test:
            try:
                __import__(mod)
                import_results[mod] = "OK"
            except Exception:
                import_results[mod] = traceback.format_exc()
        diag["third_party_imports"] = import_results

        # Add sys.path resolution for repo root and backend
        repo_root = Path(__file__).resolve().parent.parent
        for p in [str(repo_root), str(repo_root / "backend"), os.getcwd(), os.path.join(os.getcwd(), "backend")]:
            if p and p not in sys.path:
                sys.path.insert(0, p)

        # Test backend.app.main import
        try:
            from backend.app.main import app
            diag["backend_app_main"] = "OK"
            diag["routes_count"] = len(app.routes)
        except Exception:
            diag["backend_app_main"] = traceback.format_exc()

        # Test frozen models existence
        models_dir = repo_root / "models" / "experiments"
        diag["models_dir_exists"] = models_dir.exists()
        if models_dir.exists():
            diag["models_files"] = [str(p.relative_to(repo_root)) for p in models_dir.rglob("*") if p.is_file()]

        # Return JSON
        body = json.dumps(diag, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
