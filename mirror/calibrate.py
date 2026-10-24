"""Pick the cosine threshold and top1-top2 margin from data, then write them into config.yaml.

    python mirror/calibrate.py --strangers data/strangers
    python mirror/calibrate.py --strangers data/strangers --genuine data/genuine --dry-run

--strangers  folder (searched recursively) of faces of people who are NOT enrolled. Best source:
             non-attendees captured on the mirror itself (`enroll.py snap data/strangers`), topped up
             with any public face photos. A few hundred faces gives usable numbers.
--genuine    optional folder of held-out shots of enrolled people, one subfolder per name
             (`enroll.py snap data/genuine/Alice`), ideally captured on a different day. Without it,
             genuine scores come from leave-one-out over the enrollment shots, which is optimistic.

The multiple-comparison effect: a stranger is compared against all N enrolled people and only
has to beat the threshold once, so the per-encounter false-accept rate is ~ 1-(1-p)^N ~= N*p where
p is the per-pair rate. With 20 people, a threshold that looks like 0.1% per pair is ~2% per stranger.
This tool reports both and plans against the worse of (measured per-stranger, predicted from per-pair).
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import Config, load_config, setup_logging, update_config_values
from faces import FaceEngine, Gallery

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)
MARGINS = np.array([0.03, 0.05, 0.08, 0.10, 0.12, 0.15])  # 0.03 floor: always keep some lookalike protection
REF_MARGIN = 1  # index into MARGINS used while choosing the threshold (0.05)
# Per-frame genuine rejection we're willing to pay. Voting (k of n frames) absorbs a fair amount,
# so this is the upper end of the threshold search. (+) allows stricter thresholds.
FRR_BUDGET = 0.15


@dataclass
class Probes:
    """For each probe: scores vs every enrolled person, plus the true identity (None for strangers)."""

    scores: np.ndarray  # (n, P)
    truth: list[str | None]
    names: list[str]

    def top2(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(self.scores, axis=1)[:, ::-1]
        rows = np.arange(len(self.scores))
        top1 = self.scores[rows, order[:, 0]]
        top2 = self.scores[rows, order[:, 1]] if self.scores.shape[1] > 1 else np.full(len(rows), -1.0)
        return order[:, 0], top1, top1 - top2


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def embed_folder(engine: FaceEngine, paths: list[Path], cache_path: Path) -> np.ndarray:
    """Embed the largest face in each image. Cached by path+mtime: re-runs are instant."""
    import cv2

    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            cache = dict(zip(z["keys"].tolist(), z["emb"]))
    out, dirty = [], False
    for i, p in enumerate(paths):
        key = f"{p.resolve()}|{p.stat().st_mtime_ns}"
        if key not in cache:
            img = cv2.imread(str(p))
            face = engine.largest(img) if img is not None else None
            cache[key] = engine.embed(img, face) if face is not None else np.zeros(512, np.float32)
            dirty = True
            if (i + 1) % 25 == 0:
                print(f"  embedded {i + 1}/{len(paths)}", flush=True)
        if cache[key].any():
            out.append(cache[key])
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(list(cache)), emb=np.stack(list(cache.values())))
    skipped = len(paths) - len(out)
    if skipped:
        print(f"  ({skipped} images had no detectable face)")
    return np.stack(out) if out else np.zeros((0, 512), np.float32)


def genuine_loo(gallery: Gallery, mode: str) -> Probes:
    """Leave-one-out: each enrollment shot vs a gallery built without it."""
    rows, truth = [], []
    for name in gallery.names:
        shots = gallery.samples[name]
        if len(shots) < 2:
            continue
        for i in range(len(shots)):
            g = gallery.without_sample(name, i)
            rows.append(g.scores(shots[i], mode))
            truth.append(name)
    return Probes(np.array(rows), truth, gallery.names)


def binom_at_least(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--strangers", type=Path, required=True)
    ap.add_argument("--genuine", type=Path)
    ap.add_argument("--target-far", type=float, help="acceptable per-stranger false accept rate "
                                                     "(default recognition.target_far)")
    ap.add_argument("--det-size", type=int, default=640, help="detector size for still images")
    ap.add_argument("--dry-run", action="store_true", help="report only, don't touch config.yaml")
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    setup_logging(name="calibrate")
    rc = cfg.recognition
    target_far = args.target_far if args.target_far is not None else rc.target_far
    gallery = Gallery.load(cfg.resolve(cfg.paths.embeddings))
    P = len(gallery.names)
    if P < 2:
        sys.exit("enroll at least 2 people first")
    engine = FaceEngine(rc, det_size=(args.det_size, args.det_size))
    cache = cfg.resolve("data/calib_cache.npz")

    print(f"Enrolled: {P} people, match_mode={rc.match_mode}")
    print(f"Embedding strangers from {args.strangers} ...")
    S_emb = embed_folder(engine, find_images(args.strangers), cache)
    if len(S_emb) == 0:
        sys.exit("no stranger faces found")
    strangers = Probes(gallery.scores(S_emb, rc.match_mode), [None] * len(S_emb), gallery.names)

    if args.genuine:
        rows, truth = [], []
        for d in sorted(p for p in args.genuine.iterdir() if p.is_dir()):
            if d.name not in gallery.names:
                print(f"  ! {d.name} not enrolled, skipped")
                continue
            e = embed_folder(engine, find_images(d), cache)
            rows += list(gallery.scores(e, rc.match_mode)) if len(e) else []
            truth += [d.name] * len(e)
        genuine = Probes(np.array(rows), truth, gallery.names)
        g_source = f"held-out set {args.genuine}"
    else:
        genuine = genuine_loo(gallery, rc.match_mode)
        g_source = "leave-one-out over enrollment shots (optimistic: same session)"
    if len(genuine.truth) == 0:
        sys.exit("no genuine probes (need >=2 shots per person, or --genuine)")

    # ---- sweep
    g_idx, g_top1, g_margin = genuine.top2()
    g_correct = np.array([genuine.names[i] == t for i, t in zip(g_idx, genuine.truth)])
    _, s_top1, s_margin = strangers.top2()

    T = THRESHOLDS[:, None]
    M = MARGINS[None, :]
    g_acc = (g_top1[None, None, :] >= T[..., None]) & (g_margin[None, None, :] >= M[..., None])  # (t, m, n)
    frr = 1 - (g_acc & g_correct).mean(axis=2)
    misid = (g_acc & ~g_correct).mean(axis=2)
    s_acc = (s_top1[None, None, :] >= T[..., None]) & (s_margin[None, None, :] >= M[..., None])
    far_measured = s_acc.mean(axis=2)
    per_pair = (strangers.scores.ravel()[None, :] >= THRESHOLDS[:, None]).mean(axis=1)
    far_predicted = 1 - (1 - per_pair) ** P
    far_used = np.maximum(far_measured, far_predicted[:, None])

    # ---- choose the threshold: the window between "strangers get through" (too low) and "friends
    # get rejected" (too high). Sit in the middle of it so day-of drift in either direction is survivable.
    notes: list[str] = []
    far_ok = np.nonzero(far_used[:, REF_MARGIN] <= target_far)[0]
    frr_ok = np.nonzero(frr[:, REF_MARGIN] + misid[:, REF_MARGIN] <= FRR_BUDGET)[0]
    if len(far_ok) == 0:
        ti = int(np.argmin(far_used[:, REF_MARGIN]))
        notes.append(f"!! no threshold reaches the target FAR {target_far:.1%}: using the safest found. "
                     "Enrollment quality or lighting needs work, or lean on the override page.")
    elif len(frr_ok) == 0 or frr_ok.max() < far_ok.min():
        ti = int(far_ok.min())
        notes.append(f"!! no window where strangers are kept out AND friends get in: friends will be "
                     f"rejected ~{frr[ti, REF_MARGIN]:.0%} of frames. Re-enroll under party lighting.")
    else:
        lo_i, hi_i = int(far_ok.min()), int(frr_ok.max())
        ti = (lo_i + hi_i) // 2
        print(f"\nSafe window: {THRESHOLDS[lo_i]:.2f} (strangers kept out) .. {THRESHOLDS[hi_i]:.2f} "
              f"(friends still accepted); taking the middle.")
    # ---- choose the margin at that threshold: fewest misidentifications, then the largest margin
    # (better lookalike protection) that costs at most 2 points of extra rejections, capped at 0.10.
    best_misid = misid[ti].min()
    base_frr = frr[ti, 0]
    mi = 0
    for j, m in enumerate(MARGINS):
        if misid[ti, j] == best_misid and m <= 0.10 + 1e-9 and frr[ti, j] <= base_frr + 0.02:
            mi = j
    if misid[ti, mi] > best_misid:
        mi = int(np.argmin(misid[ti]))
    t_best, m_best = float(THRESHOLDS[ti]), float(MARGINS[mi])

    # ---- report
    print(f"\nGenuine probes: {len(genuine.truth)} ({g_source})")
    print(f"Stranger faces: {len(S_emb)}  x {P} people = {len(S_emb) * P} comparisons")
    print(f"Target per-stranger FAR: {target_far:.3%}\n")
    print(f"Sweep at margin {m_best:.2f}:")
    print("  thresh | per-pair FAR | stranger FAR: predicted  measured | genuine FRR  misID")
    lo, hi = max(0, ti - 8), min(len(THRESHOLDS), ti + 9)
    for i in range(lo, hi, 1 if hi - lo < 12 else 2):
        mark = "  <== recommended" if i == ti else ""
        print(f"   {THRESHOLDS[i]:.2f}  |   {per_pair[i]:8.4%}   |       {far_predicted[i]:7.3%}   "
              f"{far_measured[i, mi]:7.3%}  |   {frr[i, mi]:6.1%}   {misid[i, mi]:5.1%}{mark}")
    print(f"\nMargins at threshold {t_best:.2f}:")
    for j, m in enumerate(MARGINS):
        print(f"   margin {m:.2f}: FRR {frr[ti, j]:5.1%}  misID {misid[ti, j]:5.1%}  "
              f"stranger FAR {far_used[ti, j]:6.3%}{'  <==' if j == mi else ''}")

    for n in notes:
        print(n)
    accept = 1 - frr[ti, mi] - misid[ti, mi]
    commit = binom_at_least(rc.vote_min, rc.vote_window, accept)
    print(f"\nRecommended: threshold {t_best:.2f}, margin {m_best:.2f}")
    print(f"  per-frame genuine accept {accept:.1%} -> with {rc.vote_min}-of-{rc.vote_window} voting a friend is "
          f"confirmed ~{commit:.1%} of the time per window (optimistic: frames aren't independent).")
    print("  Voting also suppresses stranger false accepts, but correlated frames mean don't count on it.")
    if len(S_emb) < 200:
        print(f"  ! only {len(S_emb)} strangers: the smallest FAR this can measure is {1 / len(S_emb):.2%}. "
              "Add more stranger faces for tighter numbers.")
    if not args.genuine:
        print("  ! genuine numbers are from enrollment shots; real FRR at the party will be higher. "
              "Capture a held-out set on another day with `enroll.py snap data/genuine/<Name>`.")

    if args.dry_run:
        print("\n(dry run: config.yaml not changed)")
        return
    shutil.copy(cfg.path, cfg.path.with_suffix(".yaml.bak"))
    update_config_values(cfg.path, "recognition", {"threshold": f"{t_best:.2f}", "margin": f"{m_best:.2f}"})
    print(f"\nwrote recognition.threshold={t_best:.2f}, margin={m_best:.2f} to {cfg.path} (backup: config.yaml.bak)")


if __name__ == "__main__":
    main()
