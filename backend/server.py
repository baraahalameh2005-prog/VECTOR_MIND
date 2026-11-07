"""
server.py — tiny web server for the chat UI (Python standard library only).

    GET  /          -> frontend/index.html
    POST /api/ask   -> body {"question": "...", "history": [{"question","answer"}, ...]}
                       returns rag_pipeline.ask(question, history) as JSON:
                       {"status": "answered"|"refused", "answer": "...", "sources": [{"source","page"}]}

Run from the project root:   backend\\.venv\\Scripts\\python backend\\server.py
Then open http://localhost:8000
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import PROJECT_ROOT
from generator import warm_up
from rag_pipeline import ask, build_index

HOST, PORT = "localhost", 8000
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"
_ask_lock = threading.Lock()   # one question at a time: the local model handles one request well


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/ask":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(body.get("question", "")).strip()
            history = body.get("history") or []
            if not isinstance(history, list):
                history = []
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": "invalid JSON"})
            return
        if not question:
            self._json(400, {"error": "question is empty"})
            return
        try:
            with _ask_lock:
                result = ask(question, history)
        except RuntimeError as error:          # e.g. Ollama not running
            self._json(503, {"error": str(error)})
            return
        self._json(200, result)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[server] {fmt % args}\n")


if __name__ == "__main__":
    print("Building the index...")
    build_index()
    print("Warming up the model...")
    try:
        warm_up()
    except Exception as error:          # server still starts; first question will just be slower
        print(f"[WARNING] model warm-up failed: {error}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Ready: http://{HOST}:{PORT}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
