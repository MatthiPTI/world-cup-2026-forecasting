"""Mini-serveur local pour le site de comparaison des pronos.

Usage:
    uv run --env-file .env mpp-site            # http://127.0.0.1:8765
    uv run --env-file .env mpp-site --port 9000

Sert site/index.html et expose deux endpoints :
    GET  /api/data  → {predictions, points, saisie}
    POST /api/save  → écrit data/mes_pronos.json (pronos perso, résultats réels,
                      score moyen des collègues)

Aucune dépendance externe (stdlib uniquement), bind 127.0.0.1 seulement.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SITE_DIR = PROJECT_ROOT / "site"
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"
POINTS_FILE = DATA_DIR / "mpp_points.json"
SAISIE_FILE = DATA_DIR / "mes_pronos.json"

DEFAULT_SAISIE: dict = {"mes_scores": {}, "resultats": {}, "score_moyen_collegues": None}


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path in ("/", "/index.html"):
            html = (SITE_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/data":
            self._send_json({
                "predictions": _load_json(PREDICTIONS_FILE, []),
                "points": _load_json(POINTS_FILE, {}),
                "saisie": _load_json(SAISIE_FILE, DEFAULT_SAISIE),
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/save":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            saisie = json.loads(self.rfile.read(length).decode("utf-8"))
            assert isinstance(saisie, dict)
        except (json.JSONDecodeError, AssertionError, UnicodeDecodeError):
            self._send_json({"error": "invalid JSON"}, 400)
            return
        DATA_DIR.mkdir(exist_ok=True)
        with open(SAISIE_FILE, "w", encoding="utf-8") as f:
            json.dump(saisie, f, indent=2, ensure_ascii=False)
        self._send_json({"ok": True})

    def log_message(self, fmt: str, *args) -> None:
        pass  # silence les logs de requêtes


def main() -> None:
    parser = argparse.ArgumentParser(description="Site local de comparaison des pronos")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Ne pas ouvrir le navigateur automatiquement")
    args = parser.parse_args()

    if not PREDICTIONS_FILE.exists():
        print("⚠ data/predictions.json absent — lance d'abord : "
              "uv run --env-file .env mpp-predict --model bayes")

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Site des pronos : {url}  (Ctrl+C pour arrêter)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")


if __name__ == "__main__":
    main()
