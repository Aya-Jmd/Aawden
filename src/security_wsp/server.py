"""Loopback-only evidence viewer. No uploaded code, shell execution, or remote inference."""
from __future__ import annotations
import io
import json
import mimetypes
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .artifacts import read_json, verify_run


def serve(run: Path, port=8765):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose a port between 1024 and 65535")
    root = Path(__file__).resolve().parents[2]
    protocol = read_json(root / "configs/protocol.json")
    if not (run / "dashboard.json").exists():
        from .demo import run_demo
        run_demo(run, protocol)
    verify_run(run)
    state = {"run": run.resolve(), "job": {"status": "idle"}, "lock": threading.Lock()}
    web = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and str(args[0]).startswith("POST"):
                super().log_message(format, *args)

        def reply(self, data, status=200, content_type="application/json", filename=None):
            payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/api/overview":
                    self.reply(read_json(state["run"] / "dashboard.json"))
                elif route == "/api/aya-result":
                    path = root / "docs/AYA_VERIFICATION.json"
                    if not path.is_file():
                        return self.reply({"available": False, "message": "No saved Aya audit is included"}, 404)
                    self.reply(read_json(path))
                elif route == "/api/manifest":
                    self.reply(read_json(state["run"] / "manifest.json"))
                elif route == "/api/job":
                    self.reply(state["job"])
                elif route == "/api/report":
                    self.reply((state["run"] / "report.md").read_bytes(), content_type="text/markdown; charset=utf-8", filename="aawden-report.md")
                elif route == "/api/export":
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        for name in ("manifest.json", "summary.json", "results.json", "dataset.json", "oracles.json", "trials.json",
                                     "original.json", "edited.json", "frozen.json", "reload-verification.json", "report.md", "auditkit.json"):
                            path = state["run"] / name
                            if path.is_file():
                                archive.write(path, name)
                    self.reply(buffer.getvalue(), content_type="application/zip", filename="aawden-evidence.zip")
                elif route in ("/", "/index.html", "/app.js", "/style.css"):
                    path = web / ("index.html" if route == "/" else route[1:])
                    content_type = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8"}[path.suffix[1:]]
                    self.reply(path.read_bytes(), content_type=content_type)
                else:
                    self.reply({"error": "Not found"}, 404)
            except (OSError, ValueError, KeyError) as exc:
                self.reply({"error": type(exc).__name__, "message": "The evidence artifact could not be read. Check the server log."}, 500)

        def do_POST(self):
            route = urlsplit(self.path).path
            origin = self.headers.get("Origin", "")
            allowed = (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
            if self.headers.get("X-Aawden-Local") != "1" or (origin and origin not in allowed):
                return self.reply({"error": "Local action header and matching origin required"}, 403)
            if route == "/api/verify":
                try:
                    return self.reply(verify_run(state["run"]))
                except ValueError as exc:
                    return self.reply({"error": str(exc)}, 409)
            if route != "/api/demo":
                return self.reply({"error": "Not found"}, 404)
            with state["lock"]:
                if state["job"]["status"] == "running":
                    return self.reply({"error": "A pipeline run is already active"}, 409)
                state["job"] = {"status": "running", "started_at": time.time()}
            def work():
                try:
                    from .demo import run_demo
                    new_run = root / "runs" / ("dashboard-" + str(time.time_ns()))
                    run_demo(new_run, protocol)
                    with state["lock"]:
                        state["run"] = new_run
                        state["job"] = {"status": "completed", "message": "Demo completed. This is constructed toy evidence."}
                except Exception as exc:
                    with state["lock"]:
                        state["job"] = {"status": "failed", "message": str(exc)}
            threading.Thread(target=work, daemon=True).start()
            self.reply({"status": "running"}, 202)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Aawden evidence console: http://127.0.0.1:{port}\nEvidence: constructed toy demo. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()

