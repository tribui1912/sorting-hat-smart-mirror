"""Recognition service: camera -> detect -> track -> (gate) -> embed -> match -> vote -> trigger.

Standalone, for tuning in front of the mirror:
    python mirror/recognize.py --preview          # live window: boxes, scores, votes
    python mirror/recognize.py --preview --fire   # ...and actually run the hat sequence on a match
    python mirror/recognize.py                    # headless, decisions printed to the log
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from common import Config, People, Person, load_config, setup_logging
from faces import FaceEngine, Gallery
from tracker import IoUTracker, Track

log = logging.getLogger("sortinghat.recognize")

OnMatch = Callable[[Person, dict[str, Any]], bool]  # returns True if the show accepted the trigger


class Recognizer:
    def __init__(self, cfg: Config, people: People, on_match: OnMatch, is_busy: Callable[[], bool] = lambda: False):
        self.cfg = cfg
        self.rc = cfg.recognition
        self.people = people
        self.on_match = on_match
        self.is_busy = is_busy
        self.enabled = self.rc.enabled  # flipped live from the override page
        self.available = False  # True once the model + camera are up
        self.tracker = IoUTracker(stable_iou=self.rc.stable_iou, max_misses=self.rc.track_max_misses,
                                  vote_window=self.rc.vote_window)
        self.cooldown: dict[str, float] = {}
        self._last: tuple[str, float, float] | None = None  # label, mean score, monotonic time
        self._gallery_path = cfg.resolve(cfg.paths.embeddings)
        self._gallery_mtime = 0.0
        self.gallery = Gallery()
        self.engine: FaceEngine | None = None
        self.fps = 0.0

    @property
    def last(self) -> dict[str, Any] | None:
        if not self._last:
            return None
        label, score, t = self._last
        return {"label": label, "score": round(score, 3), "ago_s": round(time.monotonic() - t)}

    def _maybe_reload(self) -> None:
        """Pick up re-enrollment without a restart."""
        try:
            mtime = self._gallery_path.stat().st_mtime
        except FileNotFoundError:
            if self._gallery_mtime == 0.0:
                log.warning("no embeddings at %s yet: run enroll.py. Recognition will match nobody.",
                            self._gallery_path)
                self._gallery_mtime = -1.0
            return
        if mtime != self._gallery_mtime:
            self.gallery = Gallery.load(self._gallery_path)
            self.people.reload()
            self._gallery_mtime = mtime
            missing = [n for n in self.gallery.names if self.people.get(n) is None]
            log.info("gallery loaded: %d people%s", len(self.gallery.names),
                     f" (not in people.json: {missing})" if missing else "")

    # ------------------------------------------------------------------ per-frame logic

    def gate(self, t: Track) -> str | None:
        """None if the face is good enough to embed, else the reason it isn't (shown in the preview)."""
        f = t.face
        if f.height < self.rc.min_face_px:
            return f"small {f.height:.0f}px"
        if f.yaw > self.rc.max_yaw:
            return f"turned {f.yaw:.2f}"
        if t.stable < self.rc.stable_frames:
            return f"settling {t.stable}/{self.rc.stable_frames}"
        return None

    def step(self, frame: np.ndarray) -> list[Track]:
        assert self.engine is not None
        now = time.monotonic()
        faces = self.engine.detect(frame)
        seen = self.tracker.update(faces)
        if not self.enabled:
            for t in seen:
                t.info = "recognition off"
            return seen
        if self.is_busy():
            for t in seen:
                t.info = "show running"
            return seen

        candidates = seen
        if self.rc.only_largest and seen:
            biggest = max(seen, key=lambda t: t.face.height)
            for t in seen:
                if t is not biggest:
                    t.info = "not largest"
            candidates = [biggest]

        for t in candidates:
            if t.done:
                t.info = f"done ({t.label})"
                continue
            if now < t.retry_at:
                t.info = f"unknown, retry in {t.retry_at - now:.0f}s"
                continue
            reason = self.gate(t)
            if reason:
                t.info = reason
                continue
            emb = self.engine.embed(frame, t.face)
            m = self.gallery.match(emb, self.rc.threshold, self.rc.margin, self.rc.match_mode)
            t.votes.append(m.label)
            t.scores.append(m.score)
            t.label = m.best
            t.info = f"{m.best or '-'} {m.score:.2f} m{m.margin:.2f} {'OK' if m.label else 'rej'} " \
                     f"[{sum(v == m.label for v in t.votes)}/{len(t.votes)}]"
            log.debug("track %d: best=%s score=%.3f margin=%.3f -> %s", t.id, m.best, m.score, m.margin,
                      m.label or "UNKNOWN")
            self._decide(t, now)
        return seen

    def _decide(self, t: Track, now: float) -> None:
        counts = Counter(t.votes)
        decided = [lbl for lbl, n in counts.most_common() if n >= self.rc.vote_min]
        if not decided:
            return
        winner = decided[0]
        scores = [s for v, s in zip(t.votes, t.scores) if v == winner]
        score = float(np.mean(scores)) if scores else 0.0

        if winner is None:  # UNKNOWN won the vote
            self._last = ("UNKNOWN", score, now)
            log.info("track %d: UNKNOWN (best guess %s, mean score %.2f)", t.id, t.label, score)
            t.votes.clear()
            t.scores.clear()
            if self.rc.unknown_action == "random_house":
                guest = self.people.guest()
                if self.on_match(guest, {"score": score, "track": t.id, "unknown": True}):
                    t.done = True
                    t.label = f"guest:{guest.house}"
                    return
            t.retry_at = now + self.rc.unknown_retry_s
            return

        person = self.people.get(winner)
        t.label = winner
        self._last = (winner, score, now)
        if person is None:
            log.warning("%s is enrolled but missing from people.json; not sorting", winner)
            t.done = True
            return
        since = now - self.cooldown.get(winner, -1e9)
        if since < self.rc.cooldown_s:
            log.info("track %d: %s recognised but in cooldown (%.0fs left)", t.id, winner,
                     self.rc.cooldown_s - since)
            t.done = True
            return
        log.info("MATCH track %d: %s (mean score %.2f over %d votes)", t.id, winner, score, len(scores))
        if self.on_match(person, {"score": score, "track": t.id}):
            self.cooldown[winner] = now
            t.done = True
        else:  # show got busy between our check and the trigger; try again later
            t.votes.clear()
            t.scores.clear()

    # ------------------------------------------------------------------ loop

    def load(self) -> None:
        self._maybe_reload()
        self.engine = FaceEngine(self.rc)

    def run(self, stop: threading.Event, frame_hook: Callable[[np.ndarray, list[Track]], None] | None = None) -> None:
        from camera import Camera

        if self.engine is None:
            try:
                self.load()
            except ImportError as e:
                log.error("InsightFace unavailable (%s): recognition disabled, use the override page", e)
                return
        with Camera(self.cfg.camera) as cam:
            self.available = True
            frames, t0, last_reload = 0, time.monotonic(), time.monotonic()
            try:
                while not stop.is_set():
                    frame = cam.read()
                    tracks = self.step(frame)
                    if frame_hook:
                        frame_hook(frame, tracks)
                    frames += 1
                    now = time.monotonic()
                    if now - t0 >= 10:
                        self.fps = frames / (now - t0)
                        log.debug("recognition %.1f fps, %d tracks", self.fps, len(self.tracker.tracks))
                        frames, t0 = 0, now
                    if now - last_reload > 5:
                        self._maybe_reload()
                        last_reload = now
            finally:
                self.available = False


# ---------------------------------------------------------------------------- standalone


def preview_hook(cfg: Config, stop: threading.Event) -> Callable[[np.ndarray, list[Track]], None]:
    """pygame debug window: camera image, face boxes, gate reasons, votes."""
    import os

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.display.init()
    pygame.font.init()
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 18)
    screen: list[pygame.Surface] = []

    def hook(frame: np.ndarray, tracks: list[Track]) -> None:
        h, w = frame.shape[:2]
        if not screen:
            scale = min(1.0, 900 / max(w, h))
            screen.append(pygame.display.set_mode((int(w * scale), int(h * scale))))
            pygame.display.set_caption("recognize.py preview (ESC to quit)")
        sc = screen[0]
        k = sc.get_width() / w
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame[:, :, ::-1].swapaxes(0, 1)))
        sc.blit(pygame.transform.scale(surf, sc.get_size()), (0, 0))
        for t in tracks:
            x1, y1, x2, y2 = (t.face.bbox * k).astype(int)
            color = (80, 220, 120) if t.done else (240, 200, 80) if t.votes else (160, 160, 160)
            pygame.draw.rect(sc, color, (x1, y1, x2 - x1, y2 - y1), 2)
            for line_no, text in enumerate([f"#{t.id} {t.info}",
                                            f"{t.face.height:.0f}px yaw {t.face.yaw:.2f}"]):
                img = font.render(text, True, color, (0, 0, 0))
                sc.blit(img, (x1, max(0, y1 - 44 + 22 * line_no)))
        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                stop.set()

    return hook


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--preview", action="store_true", help="show a debug window with boxes and scores")
    ap.add_argument("--fire", action="store_true", help="run the hat sequence on a match (no mirror display)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every per-frame score")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, verbose=args.verbose, name="recognize")
    people = People(cfg.resolve(cfg.paths.people), cfg.houses)
    stop = threading.Event()

    if args.fire:
        from hat import HatClient
        from show import ShowController

        show = ShowController(cfg, HatClient(cfg), lambda s, p: log.info("display -> %s", s))
        rec = Recognizer(cfg, people, lambda p, info: show.trigger(p, "recognition")[0], lambda: show.busy)
    else:
        def announce(p: Person, info: dict[str, Any]) -> bool:
            print(f">>> would sort {p.name or 'guest'} into {p.house}  {info}")
            return True

        rec = Recognizer(cfg, people, announce)

    hook = preview_hook(cfg, stop) if args.preview else None
    try:
        rec.run(stop, hook)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
