"""InsightFace wrapper (detection + ArcFace embedding) and the enrolled-face gallery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import RecognitionCfg

log = logging.getLogger("sortinghat.faces")


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5x2 landmarks: left eye, right eye, nose, left mouth, right mouth
    score: float

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def yaw(self) -> float:
        """Rough head-turn proxy: nose offset from the eye midpoint, in eye-distances. 0 = frontal."""
        le, re, nose = self.kps[0], self.kps[1], self.kps[2]
        eye_dist = float(np.linalg.norm(re - le)) or 1.0
        return float(abs(nose[0] - (le[0] + re[0]) / 2) / eye_dist)


class FaceEngine:
    """Detection runs on every frame; embedding is a separate call so it only happens when worth it."""

    def __init__(self, cfg: RecognitionCfg, det_size: tuple[int, int] | None = None):
        from insightface.app import FaceAnalysis  # heavy import: keep it out of module import time

        self.cfg = cfg
        app = FaceAnalysis(name=cfg.model, allowed_modules=["detection", "recognition"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=tuple(det_size or cfg.det_size), det_thresh=cfg.det_threshold)
        self.det = app.det_model
        self.rec = app.models["recognition"]
        log.info("InsightFace %s ready (det_size=%s)", cfg.model, det_size or cfg.det_size)

    def detect(self, img: np.ndarray) -> list[Face]:
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            faces.append(Face(bbox=bboxes[i, :4], kps=kpss[i], score=float(bboxes[i, 4])))
        return faces

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        from insightface.utils import face_align

        crop = face_align.norm_crop(img, landmark=face.kps, image_size=self.rec.input_size[0])
        v = self.rec.get_feat(crop).flatten().astype(np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def largest(self, img: np.ndarray) -> Face | None:
        faces = self.detect(img)
        return max(faces, key=lambda f: f.height) if faces else None


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


@dataclass
class Match:
    label: str | None  # None = UNKNOWN
    score: float  # top-1 cosine
    margin: float  # top1 - top2
    best: str | None  # top-1 name even when rejected (for logs/tuning)


class Gallery:
    """Per-person embeddings plus their averaged centroid. Stored as a flat npz."""

    def __init__(self, samples: dict[str, np.ndarray] | None = None):
        self.samples: dict[str, np.ndarray] = {k: np.asarray(v, np.float32) for k, v in (samples or {}).items()}
        self._rebuild()

    def _rebuild(self) -> None:
        self.names = sorted(n for n, v in self.samples.items() if len(v))
        self.centroids = (normalize(np.stack([self.samples[n].mean(axis=0) for n in self.names]))
                          if self.names else np.zeros((0, 512), np.float32))

    # -- persistence

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        if not path.exists():
            return cls()
        with np.load(path, allow_pickle=False) as z:
            labels, emb = z["labels"], z["embeddings"]
        return cls({str(n): emb[labels == n] for n in np.unique(labels)})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = np.array([n for n in self.names for _ in range(len(self.samples[n]))])
        emb = (np.concatenate([self.samples[n] for n in self.names]) if self.names
               else np.zeros((0, 512), np.float32))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, labels=labels, embeddings=emb, centroid_labels=np.array(self.names), centroids=self.centroids)
        tmp.replace(path)

    # -- editing

    def set(self, name: str, embeddings: np.ndarray) -> None:
        self.samples[name] = normalize(np.asarray(embeddings, np.float32))
        self._rebuild()

    def remove(self, name: str) -> None:
        self.samples.pop(name, None)
        self._rebuild()

    def without_sample(self, name: str, idx: int) -> "Gallery":
        s = dict(self.samples)
        s[name] = np.delete(s[name], idx, axis=0)
        return Gallery(s)

    # -- matching

    def scores(self, emb: np.ndarray, mode: str = "centroid") -> np.ndarray:
        """Cosine similarity of emb (or a batch of them) against every person, in self.names order."""
        if mode == "centroid":
            return emb @ self.centroids.T
        # max over each person's individual shots
        return np.stack([(emb @ self.samples[n].T).max(axis=-1) for n in self.names], axis=-1)

    def match(self, emb: np.ndarray, threshold: float, margin: float, mode: str = "centroid") -> Match:
        if not self.names:
            return Match(None, 0.0, 0.0, None)
        s = self.scores(emb, mode)
        order = np.argsort(s)[::-1]
        top1 = float(s[order[0]])
        top2 = float(s[order[1]]) if len(order) > 1 else -1.0
        best = self.names[order[0]]
        ok = top1 >= threshold and (top1 - top2) >= margin
        return Match(best if ok else None, top1, top1 - top2, best)

    def self_consistency(self, name: str) -> np.ndarray:
        """Each of a person's shots vs their own centroid. Low values = a bad or mislabelled capture."""
        i = self.names.index(name)
        return self.samples[name] @ self.centroids[i]
