"""The party app: mirror display + recognition + override page + hat health checks, in one process.

    python mirror/main.py                    # party mode: fullscreen, errors logged and survived
    python mirror/main.py --dev --windowed   # development: any thread crash is fatal and loud
    python mirror/main.py --no-recognition   # the "dumb" show: phone override + keyboard only

Keyboard on the mirror (if one is plugged in): same keys as display.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from common import People, Person, Supervisor, load_config, setup_logging
from display import MirrorDisplay, key_handler
from hat import HatClient
from server import OverrideServer
from show import ShowController

log = logging.getLogger("sortinghat")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--dev", action="store_true", help="fail loud: a crash in any thread stops the app")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--no-recognition", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, verbose=args.verbose)
    sup = Supervisor(dev=args.dev)
    people = People(cfg.resolve(cfg.paths.people), cfg.houses)

    display = MirrorDisplay(cfg, windowed=args.windowed)
    hat = HatClient(cfg)
    show = ShowController(cfg, hat, display.post)

    recognizer = None
    if not args.no_recognition:
        from recognize import Recognizer

        def on_match(person: Person, info: dict[str, Any]) -> bool:
            ok, _ = show.trigger(person, "recognition")
            return ok

        recognizer = Recognizer(cfg, people, on_match, is_busy=lambda: show.busy)
        sup.spawn("recognize", lambda: recognizer.run(sup.stopping), restart_delay_s=5.0)

    server = OverrideServer(cfg, people, show, recognizer, hat_online=lambda: hat.online)
    sup.spawn("server", server.serve_forever)
    sup.spawn("hat-ping", lambda: hat.monitor(sup.stopping))

    def status() -> str:
        rec = "off" if recognizer is None else f"{recognizer.fps:.1f}fps" if recognizer.available else "starting"
        hat_s = "on" if hat.online else "off" if hat.online is False else "?"
        return f"hat={hat_s} rec={rec}"

    display.status_line = status
    on_key = key_handler(people, cfg, lambda p: show.trigger(p, "keyboard", preempt=True), show.stop)
    try:
        display.run(on_key=on_key, should_stop=lambda: sup.fatal is not None, dev=args.dev)
    finally:
        sup.stopping.set()  # worker threads are daemons; they die with the process
        hat.local.stop()
    if sup.fatal is not None:
        log.error("exiting because of: %r", sup.fatal)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
