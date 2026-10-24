"""Minimal IoU tracker: gives each face a stable id across frames so votes accumulate per person."""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from faces import Face


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    id: int
    face: Face
    stable: int = 0  # consecutive frames the box has barely moved
    misses: int = 0
    votes: deque[str | None] = field(default_factory=deque)
    scores: deque[float] = field(default_factory=deque)
    done: bool = False  # already sorted (or in cooldown): no more embedding work
    retry_at: float = 0.0  # monotonic time before which we don't vote (after an UNKNOWN verdict)
    label: str | None = None  # last committed or best-guess label, for debug views
    info: str = ""  # human-readable state for the preview window


class IoUTracker:
    def __init__(self, match_iou: float = 0.3, stable_iou: float = 0.6, max_misses: int = 5, vote_window: int = 7):
        self.match_iou = match_iou
        self.stable_iou = stable_iou
        self.max_misses = max_misses
        self.vote_window = vote_window
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def update(self, faces: list[Face]) -> list[Track]:
        """Associate this frame's detections with existing tracks. Returns tracks seen this frame."""
        pairs = sorted(
            ((iou(t.face.bbox, f.bbox), ti, fi) for ti, t in enumerate(self.tracks) for fi, f in enumerate(faces)),
            reverse=True,
        )
        used_t: set[int] = set()
        used_f: set[int] = set()
        seen: list[Track] = []
        for score, ti, fi in pairs:
            if score < self.match_iou:
                break
            if ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            t = self.tracks[ti]
            t.stable = t.stable + 1 if score >= self.stable_iou else 0
            t.face = faces[fi]
            t.misses = 0
            seen.append(t)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for fi, f in enumerate(faces):
            if fi not in used_f:
                t = Track(id=next(self._ids), face=f, votes=deque(maxlen=self.vote_window),
                          scores=deque(maxlen=self.vote_window))
                self.tracks.append(t)
                seen.append(t)
        return seen
