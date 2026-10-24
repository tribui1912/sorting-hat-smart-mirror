"""The sorting sequence: idle -> thinking (hat mutters) -> announcement -> reveal -> idle.

One sequence runs at a time. Recognition triggers are refused while busy; manual
override triggers preempt whatever is running (the operator is always right).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from common import Config, Person
from hat import HatClient

log = logging.getLogger("sortinghat.show")

StateListener = Callable[[str, Person | None], None]  # ("idle"|"thinking"|"reveal", person)


@dataclass
class ShowStatus:
    state: str = "idle"
    person: str | None = None
    house: str | None = None
    source: str | None = None
    audio: str | None = None  # where the announcement played: hat / local / none
    since: float = 0.0


class ShowController:
    def __init__(self, cfg: Config, hat: HatClient, listener: StateListener):
        self.cfg = cfg.show
        self.hat = hat
        self.listener = listener
        self.status = ShowStatus(since=time.time())
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        w = self._worker
        return w is not None and w.is_alive()

    def trigger(self, person: Person, source: str, preempt: bool = False) -> tuple[bool, str]:
        """Start a sorting. Returns (accepted, reason)."""
        with self._lock:
            if self.busy:
                if not preempt:
                    return False, "busy"
                self._abort_locked()
            self._abort.clear()
            self._worker = threading.Thread(target=self._run, args=(person, source), name="show", daemon=True)
            self._worker.start()
        label = person.name or f"guest -> {person.house}"
        log.info("SORTING %s (source=%s)", label, source)
        return True, "ok"

    def stop(self) -> None:
        with self._lock:
            self._abort_locked()
        self.hat.stop()
        self._set("idle", None, None)

    def _abort_locked(self) -> None:
        self._abort.set()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=3.0)

    def _set(self, state: str, person: Person | None, source: str | None, audio: str | None = None) -> None:
        self.status = ShowStatus(
            state=state,
            person=person.name if person else None,
            house=person.house if person else None,
            source=source,
            audio=audio,
            since=time.time(),
        )
        try:
            self.listener(state, person)
        except Exception:  # noqa: BLE001 - a display hiccup must not kill the show
            log.exception("display listener failed")

    def _wait(self, seconds: float) -> bool:
        """Sleep that wakes on abort. Returns True if aborted."""
        return self._abort.wait(max(0.0, seconds))

    def _run(self, person: Person, source: str) -> None:
        try:
            self._set("thinking", person, source)
            t0 = time.monotonic()
            self.hat.mutter()
            if self._wait(self.cfg.thinking_s - (time.monotonic() - t0)):
                return

            audio = self.hat.play(person.clip)
            if audio == "none":
                log.error("announcement for %s played nowhere (hat down, no fallback clip)", person.clip)
            delay = person.reveal_delay_s if person.reveal_delay_s is not None else self.cfg.reveal_delay_s
            if self._wait(delay):
                return

            self._set("reveal", person, source, audio)
            if self._wait(self.cfg.reveal_s):
                return
            self._set("idle", None, None)
        except Exception:  # noqa: BLE001
            log.exception("show sequence failed")
            self._set("idle", None, None)
