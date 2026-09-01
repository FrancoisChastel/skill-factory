"""A tiny stdlib HTTP server backing the dashboard.

Endpoints:
    GET  /                       -> dashboard HTML
    GET  /api/runs               -> list of run summaries under the runs root
    GET  /api/run?dir=<path>     -> full result.json payload for one run
    POST /api/optimize {config}  -> run an optimization from a config path, save it

Bound to localhost by default. The ``dir`` parameter is constrained to the runs
root to prevent path traversal.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from skill_factory.persistence import list_runs, load_run

_STATIC = Path(__file__).parent / "static"


def serve(runs_dir: str = "runs", host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the dashboard server (blocking)."""
    runs_root = Path(runs_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    handler = _make_handler(runs_root)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"Skill Factory dashboard → {url}  (runs: {runs_root})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nStopping…")
    finally:
        server.server_close()


def _make_handler(runs_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - silence default logging
            return

        def do_GET(self):  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_file(_STATIC / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/runs":
                self._send_json({"runs": list_runs(runs_root)})
            elif parsed.path == "/api/run":
                self._handle_run_detail(parse_qs(parsed.query))
            else:
                self._send_error(404, "not found")

        def do_POST(self):  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            if parsed.path == "/api/optimize":
                self._handle_optimize()
            else:
                self._send_error(404, "not found")

        # --- handlers --------------------------------------------------
        def _handle_run_detail(self, query: dict) -> None:
            raw = (query.get("dir") or [""])[0]
            target = (runs_root / raw).resolve() if raw else runs_root
            if not _within(runs_root, target):
                self._send_error(400, "dir outside runs root")
                return
            try:
                self._send_json(load_run(target))
            except FileNotFoundError:
                self._send_error(404, "run not found")

        def _handle_optimize(self) -> None:
            try:
                body = self._read_json()
                config_path = body.get("config")
                if not config_path:
                    self._send_error(400, "missing 'config'")
                    return
                # Imported lazily: launching a run needs the full builder stack.
                from skill_factory.builder import run_optimization
                from skill_factory.config import load_config
                from skill_factory.persistence import save_run

                config = load_config(config_path)
                result = run_optimization(config)
                out_dir = body.get("out") or config.output.get("dir", "runs/latest")
                artifacts = save_run(result, out_dir)
                self._send_json(
                    {
                        "ok": True,
                        "dir": str(Path(artifacts.run_dir).resolve()),
                        "baseline_score": result.baseline_score,
                        "best_score": result.best_score,
                        "improvement": result.improvement,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report any failure to the client
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        # --- io helpers ------------------------------------------------
        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self._send_error(404, "missing asset")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

    return Handler


def _within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False
