"""Manual override: a phone-friendly page that forces any reveal, bypassing recognition.

    GET  /                      override.html
    GET  /api/people            [{id, name, house}], houses
    GET  /api/status            show state, hat health, recognition state
    POST /api/reveal            {"id": 3} or {"house": "slytherin"} (guest) or {"house": "random"}
    POST /api/stop
    POST /api/recognition       {"enabled": true|false}

Standalone (override page + hat, no display):
    python mirror/server.py
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from common import Config, People
from show import ShowController

log = logging.getLogger("sortinghat.server")
HTML = Path(__file__).with_name("override.html")


class OverrideServer:
    def __init__(self, cfg: Config, people: People, show: ShowController,
                 recognition: Any | None = None, hat_online: Callable[[], bool | None] = lambda: None):
        self.cfg = cfg
        self.people = people
        self.show = show
        self.recognition = recognition  # anything with .enabled (bool) and .last (dict | None)
        self.hat_online = hat_online
        self.httpd: ThreadingHTTPServer | None = None

    def status(self) -> dict[str, Any]:
        st = self.show.status
        rec = self.recognition
        return {
            "state": st.state,
            "person": st.person,
            "house": st.house,
            "source": st.source,
            "audio": st.audio,
            "state_age_s": round(time.time() - st.since, 1),
            "hat_online": self.hat_online(),
            "recognition_available": rec is not None and getattr(rec, "available", True),
            "recognition_enabled": bool(rec and rec.enabled),
            "last_match": rec.last if rec else None,
        }

    def handle(self, method: str, path: str, query: dict[str, list[str]], body: dict[str, Any]
               ) -> tuple[int, Any]:
        if path == "/api/people":
            return 200, {
                "people": [{"id": p.id, "name": p.name, "house": p.house} for p in self.people.all()],
                "houses": {k: {"label": h.label, "primary": h.primary, "secondary": h.secondary}
                           for k, h in self.cfg.houses.items()},
            }
        if path == "/api/status":
            return 200, self.status()
        if method != "POST":
            return 405, {"error": "use POST"}
        if path == "/api/reveal":
            if "id" in body:
                person = self.people.by_id(int(body["id"]))
                if person is None:
                    return 404, {"error": f"no person with id {body['id']}"}
            elif "house" in body:
                house = body["house"]
                if house == "random":
                    house = random.choice(list(self.cfg.houses))
                if house not in self.cfg.houses:
                    return 404, {"error": f"unknown house {house}"}
                person = self.people.guest(house)
            else:
                return 400, {"error": "need id or house"}
            ok, reason = self.show.trigger(person, "override", preempt=True)
            return (200 if ok else 409), {"ok": ok, "reason": reason, "name": person.name, "house": person.house}
        if path == "/api/stop":
            self.show.stop()
            return 200, {"ok": True}
        if path == "/api/recognition":
            if self.recognition is None:
                return 409, {"error": "recognition not running"}
            self.recognition.enabled = bool(body.get("enabled"))
            log.warning("recognition %s from override page", "ENABLED" if self.recognition.enabled else "DISABLED")
            return 200, {"ok": True, "enabled": self.recognition.enabled}
        return 404, {"error": "not found"}

    def serve_forever(self) -> None:
        server = self
        token = self.cfg.server.token

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("%s " + fmt, self.address_string(), *args)

            def _send(self, code: int, payload: Any, ctype: str = "application/json") -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _dispatch(self, method: str) -> None:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                try:
                    if url.path in ("/", "/index.html", "/override.html"):
                        self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
                        return
                    if token and query.get("token", [""])[0] != token and self.headers.get("X-Token") != token:
                        self._send(HTTPStatus.FORBIDDEN, {"error": "bad token"})
                        return
                    body: dict[str, Any] = {}
                    length = int(self.headers.get("Content-Length") or 0)
                    if length:
                        body = json.loads(self.rfile.read(length) or b"{}")
                    code, payload = server.handle(method, url.path, query, body)
                    self._send(code, payload)
                except Exception as e:  # noqa: BLE001 - the page must keep working
                    log.exception("override request failed")
                    self._send(500, {"error": str(e)})

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch("POST")

        self.httpd = ThreadingHTTPServer((self.cfg.server.host, self.cfg.server.port), Handler)
        self.httpd.daemon_threads = True
        log.info("override page on http://%s:%d/", self.cfg.server.host, self.cfg.server.port)
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        if self.httpd:
            self.httpd.shutdown()


def main() -> None:
    import argparse

    from common import load_config, setup_logging
    from hat import HatClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, name="server")
    people = People(cfg.resolve(cfg.paths.people), cfg.houses)
    hat = HatClient(cfg)
    show = ShowController(cfg, hat, lambda state, p: log.info("display -> %s %s", state, p.name if p else ""))
    stop = threading.Event()
    threading.Thread(target=hat.monitor, args=(stop,), daemon=True).start()
    OverrideServer(cfg, people, show, hat_online=lambda: hat.online).serve_forever()


if __name__ == "__main__":
    main()
