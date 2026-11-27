import sys
import os
import traceback
from pathlib import Path

# Add repository root and backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent
for p in [str(REPO_ROOT), str(REPO_ROOT / "backend"), os.getcwd(), os.path.join(os.getcwd(), "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

real_app = None
import_error = None

try:
    from backend.app.main import app as real_app
except Exception:
    import_error = traceback.format_exc()
    print("STARTUP IMPORT ERROR IN MAIN.PY:\n", import_error, flush=True)


async def app(scope, receive, send):
    if scope.get("type") == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    if import_error or real_app is None:
        err_msg = f"STARTUP IMPORT ERROR:\n{import_error}"
        body = err_msg.encode("utf-8")
        headers = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 500, "headers": headers})
        await send({"type": "http.response.body", "body": body})
        return

    try:
        await real_app(scope, receive, send)
    except Exception:
        exc_text = f"RUNTIME INVOCATION ERROR:\n{traceback.format_exc()}"
        print(exc_text, flush=True)
        body = exc_text.encode("utf-8")
        headers = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 500, "headers": headers})
        await send({"type": "http.response.body", "body": body})


handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
