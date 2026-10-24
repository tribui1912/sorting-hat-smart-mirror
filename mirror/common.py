"""Shared plumbing: config loading, people registry, logging, crash supervision."""

from __future__ import annotations

import dataclasses
import json
import logging
import logging.handlers
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

log = logging.getLogger("sortinghat")


# --------------------------------------------------------------------------- config schema
# Defaults here mirror config.yaml; the YAML file is the documented source of truth.


@dataclass
class PathsCfg:
    people: str = "people.json"
    embeddings: str = "data/embeddings.npz"
    enroll_images: str = "data/enroll"
    audio: str = "audio/sd"
    assets: str = "assets"
    logs: str = "logs"


@dataclass
class CameraCfg:
    size: tuple[int, int] = (1536, 864)
    rotate: int = 0
    hflip: bool = False
    vflip: bool = False
    lens_position: float | None = 0.8
    webcam_index: int = 0


@dataclass
class RecognitionCfg:
    enabled: bool = True
    model: str = "buffalo_l"
    det_size: tuple[int, int] = (320, 320)
    det_threshold: float = 0.6
    min_face_px: int = 90
    max_yaw: float = 0.5
    stable_frames: int = 3
    stable_iou: float = 0.6
    only_largest: bool = True
    threshold: float = 0.42
    margin: float = 0.08
    match_mode: str = "centroid"
    vote_window: int = 7
    vote_min: int = 5
    unknown_retry_s: float = 3.0
    cooldown_s: float = 30.0
    unknown_action: str = "ignore"
    track_max_misses: int = 5
    target_far: float = 0.01


@dataclass
class EnrollCfg:
    shots: int = 8
    warn_shot_sim: float = 0.50
    warn_mean_sim: float = 0.65
    warn_cross_sim: float = 0.45


@dataclass
class DisplayCfg:
    fullscreen: bool = True
    window_size: tuple[int, int] = (540, 960)
    rotate: int = 0
    focus_y: float = 0.22
    fps: int = 30
    font: str | None = None
    idle_text: str = "Approach the Hat..."
    idle_text_every_s: float = 25.0
    thinking_phrases: list[str] = field(default_factory=lambda: ["Hmm...", "Let me see..."])


@dataclass
class ShowCfg:
    thinking_s: float = 4.0
    reveal_delay_s: float = 2.0
    reveal_s: float = 10.0


@dataclass
class HatCfg:
    host: str = "sortinghat.local"
    port: int = 80
    timeout_s: float = 1.5
    fallback_audio: bool = True
    audio_device: str | None = None
    ping_every_s: float = 5.0


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8000
    token: str = ""


@dataclass
class EnvelopeCfg:
    sample_rate: int = 22050
    normalize_dbfs: float | None = -1.0
    frame_ms: int = 20
    smooth_frames: int = 3
    noise_floor: float = 0.10
    speech_travel: float = 0.65
    gamma: float = 0.8
    lead_ms: int = 40
    closed_deg: int = 90
    open_deg: int = 130


@dataclass
class House:
    key: str
    label: str
    primary: str
    secondary: str
    motto: str = ""  # shown under the house name, with the animal
    animal: str = ""

    @staticmethod
    def rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@dataclass
class Config:
    path: Path
    paths: PathsCfg
    camera: CameraCfg
    recognition: RecognitionCfg
    enroll: EnrollCfg
    display: DisplayCfg
    show: ShowCfg
    hat: HatCfg
    server: ServerCfg
    envelope: EnvelopeCfg
    houses: dict[str, House]

    def resolve(self, rel: str | Path) -> Path:
        """Resolve a path from config relative to the config file's directory."""
        p = Path(rel).expanduser()
        return p if p.is_absolute() else (self.path.parent / p).resolve()


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any] | None, section: str) -> T:
    data = dict(data or {})
    known = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    for k in list(data):
        if k not in known:
            log.warning("config: unknown key %s.%s (typo?) - ignored", section, k)
            data.pop(k)
    for name, f in known.items():
        if name in data and isinstance(data[name], list) and "tuple" in str(f.type):
            data[name] = tuple(data[name])
    return cls(**data)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or DEFAULT_CONFIG).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    houses_raw = raw.get("houses") or {}
    houses = {k: House(key=k, **v) for k, v in houses_raw.items()}
    if not houses:
        raise ValueError(f"{path}: no houses defined")
    cfg = Config(
        path=path,
        paths=_build(PathsCfg, raw.get("paths"), "paths"),
        camera=_build(CameraCfg, raw.get("camera"), "camera"),
        recognition=_build(RecognitionCfg, raw.get("recognition"), "recognition"),
        enroll=_build(EnrollCfg, raw.get("enroll"), "enroll"),
        display=_build(DisplayCfg, raw.get("display"), "display"),
        show=_build(ShowCfg, raw.get("show"), "show"),
        hat=_build(HatCfg, raw.get("hat"), "hat"),
        server=_build(ServerCfg, raw.get("server"), "server"),
        envelope=_build(EnvelopeCfg, raw.get("envelope"), "envelope"),
        houses=houses,
    )
    if cfg.recognition.vote_min > cfg.recognition.vote_window:
        raise ValueError("recognition.vote_min must be <= recognition.vote_window")
    if cfg.recognition.match_mode not in ("centroid", "max"):
        raise ValueError("recognition.match_mode must be 'centroid' or 'max'")
    return cfg


def update_config_values(path: Path, section: str, values: dict[str, Any]) -> None:
    """Rewrite `key: value` lines inside one top-level section, keeping comments and layout.

    PyYAML round-trips would strip every tuning comment, so this edits lines in place.
    """
    lines = path.read_text().splitlines(keepends=True)
    in_section = False
    remaining = dict(values)
    for i, line in enumerate(lines):
        if re.match(r"^\S", line):  # top-level key
            in_section = line.split(":", 1)[0].strip() == section
            continue
        if not in_section:
            continue
        m = re.match(r"^(\s+)([A-Za-z_]+)(:\s*)([^#\n]*?)(\s*)(#.*)?(\n?)$", line)
        if m and m.group(2) in remaining:
            indent, key, sep, old, gap, comment, nl = m.groups()
            text = f"{indent}{key}{sep}{remaining.pop(key)}"
            if comment:  # keep the comment in its original column where possible
                col = len(indent) + len(key) + len(sep) + len(old) + len(gap or "")
                text = text.ljust(col - 1) + " " + comment
            lines[i] = text + (nl or "")
    if remaining:
        raise KeyError(f"keys not found in section '{section}': {sorted(remaining)}")
    path.write_text("".join(lines))


# --------------------------------------------------------------------------- people


@dataclass
class Person:
    """Someone the hat can sort. A guest has no id and uses the generic house clip."""

    name: str
    house: str
    id: int | None = None
    reveal_delay_s: float | None = None  # overrides show.reveal_delay_s for this person's clip

    @property
    def clip(self) -> str:
        """Base filename of this person's announcement on the SD card (and audio/sd)."""
        return str(self.id) if self.id is not None else f"house_{self.house}"

    @property
    def is_guest(self) -> bool:
        return self.id is None


class People:
    """people.json: {"Alice": {"id": 1, "house": "gryffindor"}, ...}"""

    def __init__(self, path: Path, houses: dict[str, House]):
        self.path = path
        self.houses = houses
        self._by_name: dict[str, Person] = {}
        self.reload()

    def reload(self) -> None:
        self._by_name = {}
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text() or "{}")
        for name, entry in data.items():
            if name.startswith("_"):  # allow "_comment" keys
                continue
            p = Person(
                name=name,
                house=entry["house"],
                id=int(entry["id"]),
                reveal_delay_s=entry.get("reveal_delay_s"),
            )
            if p.house not in self.houses:
                raise ValueError(f"{self.path}: {name} has unknown house '{p.house}'")
            self._by_name[name] = p
        ids = [p.id for p in self._by_name.values()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.path}: duplicate ids")

    def save(self) -> None:
        out: dict[str, Any] = {}
        for p in sorted(self._by_name.values(), key=lambda p: p.id or 0):
            entry: dict[str, Any] = {"id": p.id, "house": p.house}
            if p.reveal_delay_s is not None:
                entry["reveal_delay_s"] = p.reveal_delay_s
            out[p.name] = entry
        self.path.write_text(json.dumps(out, indent=2) + "\n")

    def all(self) -> list[Person]:
        return sorted(self._by_name.values(), key=lambda p: p.id or 0)

    def get(self, name: str) -> Person | None:
        return self._by_name.get(name)

    def by_id(self, pid: int) -> Person | None:
        return next((p for p in self._by_name.values() if p.id == pid), None)

    def upsert(self, name: str, house: str | None) -> Person:
        existing = self._by_name.get(name)
        if house is None:
            if existing is None:
                raise ValueError(f"{name} is new: --house is required")
            return existing
        if house not in self.houses:
            raise ValueError(f"unknown house '{house}', choose from {list(self.houses)}")
        if existing:
            existing.house = house
            return existing
        next_id = max((p.id or 0 for p in self._by_name.values()), default=0) + 1
        p = Person(name=name, house=house, id=next_id)
        self._by_name[name] = p
        return p

    def remove(self, name: str) -> None:
        self._by_name.pop(name, None)

    def guest(self, house: str | None = None) -> Person:
        return Person(name="", house=house or random.choice(list(self.houses)))


# --------------------------------------------------------------------------- logging & supervision


def setup_logging(cfg: Config | None = None, verbose: bool = False, name: str = "sortinghat") -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)-10s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if cfg is not None:
        try:
            log_dir = cfg.resolve(cfg.paths.logs)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=3)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as e:
            log.warning("file logging disabled: %s", e)


class Supervisor:
    """Runs worker threads. Dev mode: a crash is fatal and loud. Party mode: log it and restart."""

    def __init__(self, dev: bool):
        self.dev = dev
        self.fatal: BaseException | None = None
        self.stopping = threading.Event()

    def spawn(self, name: str, fn: Callable[[], None], restart_delay_s: float = 3.0) -> threading.Thread:
        def run() -> None:
            while not self.stopping.is_set():
                try:
                    fn()
                    return  # clean exit
                except Exception as e:  # noqa: BLE001 - this is the top-level guard
                    if self.dev:
                        log.exception("%s crashed (dev mode: fatal)", name)
                        self.fatal = e
                        self.stopping.set()
                        return
                    log.exception("%s crashed; restarting in %.0fs", name, restart_delay_s)
                    self.stopping.wait(restart_delay_s)

        t = threading.Thread(target=run, name=name, daemon=True)
        t.start()
        return t


def now() -> float:
    return time.monotonic()
