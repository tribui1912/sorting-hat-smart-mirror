"""The sorting sequence: idle -> thinking (the hat speaks) -> reveal -> idle.

With a dialogue script (dialogue.yaml) each sorting plays three beats:
    opening (3-5 s) -> optional suspense line (~25%) -> house announcement.
The announcement always ends in the house name; the reveal fires reveal_lead_s
before the clip ends so the crest lands as the name is shouted. Lines whose voice
clips don't exist yet run on the script's timings with the line shown as a
mirror caption (show.captions), so the show works before voice generation.

People with their own recorded announcement (audio/sd/<id>.wav) keep the classic
single-clip flow. One sequence runs at a time. Recognition triggers are refused
while busy; manual override triggers preempt whatever is running (the operator
is always right).
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from common import Config, Person
from hat import HatClient

log = logging.getLogger("sortinghat.show")

StateListener = Callable[[str, Person | None], None]  # ("idle"|"thinking"|"reveal", person)
CaptionSink = Callable[[str], None]

SFX_REVEAL = "sfx_reveal"  # optional sting in audio/sd, played as the crest appears


@dataclass
class ShowStatus:
    state: str = "idle"
    person: str | None = None
    house: str | None = None
    source: str | None = None
    audio: str | None = None  # where the announcement played: hat / local / caption / none
    since: float = 0.0


@dataclass(frozen=True)
class Line:
    id: str    # clip name on the SD card and in audio/sd
    text: str
    personal: bool = False  # {name} was filled in: a <id>__<name> clip takes precedence


class DialogueEngine:
    """Line selection: random opening (no immediate repeat), suspense with a
    probability, house announcements from a shuffled pool that only resets
    once every line of that house has been used.

    Opening/suspense lines containing {name} are personalized; a guest sorted
    without a name gets the *_generic lines instead. The house always comes
    from people.json - the engine never re-rolls it."""

    def __init__(self, data: dict):
        self.openings = [Line(**d) for d in data.get("opening", [])]
        self.suspense = [Line(**d) for d in data.get("suspense", [])]
        self.openings_generic = [Line(**d) for d in data.get("opening_generic", [])]
        self.suspense_generic = [Line(**d) for d in data.get("suspense_generic", [])]
        self.houses: dict[str, list[Line]] = {
            key: [Line(**d) for d in lines] for key, lines in data.get("houses", {}).items()
        }
        self._pools: dict[str, list[Line]] = {}
        self._last_opening: Line | None = None

    @classmethod
    def load(cls, path: Path) -> "DialogueEngine | None":
        try:
            with open(path) as f:
                eng = cls(yaml.safe_load(f) or {})
            if not eng.houses:
                log.warning("%s has no house lines: falling back to single-clip announcements", path)
                return None
            counts = ", ".join(f"{k}={len(v)}" for k, v in eng.houses.items())
            log.info("dialogue: %d openings (+%d generic), %d suspense (+%d generic), announcements %s",
                     len(eng.openings), len(eng.openings_generic),
                     len(eng.suspense), len(eng.suspense_generic), counts)
            return eng
        except FileNotFoundError:
            log.info("no %s: announcements use one clip per person/house", path.name)
        except Exception as e:  # noqa: BLE001
            log.warning("dialogue %s unreadable (%s): single-clip announcements", path, e)
        return None

    def script(self, person: Person, suspense_prob: float) -> list[Line] | None:
        """opening [+ suspense] + one unused announcement for the person's house."""
        if person.house not in self.houses:
            return None
        name = (person.name or "").strip()
        openings = self.openings if name and self.openings else (self.openings_generic or self.openings)
        suspense = self.suspense if name and self.suspense else (self.suspense_generic or self.suspense)
        out: list[Line] = []
        if openings:
            pick = random.choice(openings)
            for _ in range(3):  # reroll an immediate repeat of the last opening
                if pick is not self._last_opening:
                    break
                pick = random.choice(openings)
            self._last_opening = pick
            out.append(self._personalize(pick, name))
        if suspense and random.random() < suspense_prob:
            out.append(self._personalize(random.choice(suspense), name))
        out.append(self._draw_announcement(person.house))
        return out

    @staticmethod
    def _personalize(line: Line, name: str) -> Line:
        if not name or "{name" not in line.text:
            return line
        try:
            return Line(id=line.id, text=line.text.format(name=name), personal=True)
        except (KeyError, IndexError, ValueError):
            return Line(id=line.id, text=line.text)

    def _draw_announcement(self, house: str) -> Line:
        pool = self._pools.get(house)
        if not pool:
            pool = self.houses[house][:]
            random.shuffle(pool)
            self._pools[house] = pool
            log.info("dialogue: %s announcement pool reshuffled (%d lines)", house, len(pool))
        return pool.pop()


class ShowController:
    def __init__(self, cfg: Config, hat: HatClient, listener: StateListener,
                 caption: CaptionSink | None = None):
        self.cfg = cfg.show
        self.hat = hat
        self.listener = listener
        self.caption = caption
        self.dialogue = DialogueEngine.load(cfg.resolve(cfg.show.dialogue))
        self.status = ShowStatus(since=time.time())
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._worker: threading.Thread | None = None
        self._durations: dict[str, float | None] = {}  # clip -> seconds (None = no file)

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

    def _say(self, text: str) -> None:
        if self.cfg.captions and self.caption:
            try:
                self.caption(text)
            except Exception:  # noqa: BLE001
                log.exception("caption sink failed")

    def _wait(self, seconds: float) -> bool:
        """Sleep that wakes on abort. Returns True if aborted."""
        return self._abort.wait(max(0.0, seconds))

    def _clip_duration(self, clip: str) -> float | None:
        """Seconds of audio/sd/<clip>.wav (what the hat plays), None if there is no file.
        Only found files are cached: clips added while the show runs are picked up."""
        if clip in self._durations:
            return self._durations[clip]
        path = self.hat.local.audio_dir / f"{clip}.wav"
        try:
            with wave.open(str(path), "rb") as w:
                self._durations[clip] = w.getnframes() / float(w.getframerate())
        except (OSError, wave.Error):
            return None
        return self._durations[clip]

    def _resolve_clip(self, line: Line, name: str) -> str:
        """Personalized lines prefer a <id>__<name> clip; else the generic <id> clip."""
        if line.personal and name:
            clip = f"{line.id}__{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
            if self._clip_duration(clip) is not None:
                return clip
        return line.id

    def _fallback_duration(self, line: Line, final: bool) -> float:
        """Caption-only timing while the voice clips don't exist yet."""
        return random.uniform(6.0, 9.0) if final else random.uniform(3.0, 5.0)

    # ------------------------------------------------------------------ sequences

    def _run(self, person: Person, source: str) -> None:
        try:
            script = self.dialogue.script(person, self.cfg.suspense_prob) if self.dialogue else None
            personal_clip = person.id is not None and self._clip_duration(person.clip) is not None
            if script is None or personal_clip:
                self._run_single_clip(person, source)
            else:
                self._run_dialogue(person, source, script)
        except Exception:  # noqa: BLE001
            log.exception("show sequence failed")
            self._set("idle", None, None)

    def _run_single_clip(self, person: Person, source: str) -> None:
        """Classic flow: mutter, then one pre-recorded announcement clip."""
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

    def _run_dialogue(self, person: Person, source: str, script: list[Line]) -> None:
        self._set("thinking", person, source)
        audio = "caption"
        for i, line in enumerate(script):
            final = i == len(script) - 1
            self._say(line.text)
            clip = self._resolve_clip(line, person.name)
            duration = self._clip_duration(clip)
            if duration is not None:
                audio = self.hat.play(clip)
                if audio == "none":
                    log.warning("dialogue %s played nowhere (hat down, no local clip)", clip)
            else:
                duration = self._fallback_duration(line, final)

            if final:
                lead = self.cfg.reveal_lead_s if duration and audio != "none" else 0.0
                if self._wait(max(0.0, duration - lead)):
                    return
                self._set("reveal", person, source, audio)
                if lead and self._wait(lead):  # the crest lands as the name is shouted
                    return
                if audio != "none" and self._clip_duration(SFX_REVEAL) is not None:
                    self.hat.play(SFX_REVEAL)  # optional sting, never over the announcement
            elif self._wait(duration):
                return

        if audio == "none":
            log.error("dialogue played nowhere (hat down, no clips generated yet): ran on caption timing")
        if self._wait(self.cfg.reveal_s):
            return
        self._set("idle", None, None)
